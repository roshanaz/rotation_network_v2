import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2  import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import sys
import time
from transformers import CLIPProcessor, CLIPModel
import argparse
from transformers import TFCLIPModel
import torch


def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    return x_train, y_train, x_test, y_test

class RotationPairGenerator(tf.keras.utils.Sequence):
    def __init__(self, images, batch_size=32, rotation_angles=None, image_size=(96, 96), shuffle=True, augment=True, model_type='mobilenetv2'):
        self.images = images
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.model_type = model_type
        self.augmentation = create_augmentation_layer() if augment else None

        if model_type == 'clip':
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        
        # angles every 15 degrees 
        if rotation_angles is None:
            self.rotation_angles = np.arange(0, 360, 15)
        else:
            self.rotation_angles = rotation_angles
            
        self.indices = np.arange(len(self.images))
        if self.shuffle:
            np.random.shuffle(self.indices)


    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = self.images[batch_indices]

        image_origin_batch = []
        image_rotated_batch = []
        angle_batch = []

        for img in batch_images:
            img_padded = tf.image.resize_with_pad(img, self.image_size[0], self.image_size[1])

            
            angle = np.random.choice(self.rotation_angles)
            normalized_angle = angle / 360.0

            # scipy.ndimage.rotate rotates counterclockwise
            img_rotated = rotate(img, -angle, reshape=False)
            img_rotated_resized = tf.image.resize_with_pad(img_rotated, self.image_size[0], self.image_size[1]) # same preprocessing as original image
                    
            image_origin_batch.append(img_padded)
            image_rotated_batch.append(img_rotated_resized)
            angle_batch.append(normalized_angle)
        

        image_origin_batch = np.array(image_origin_batch)
        image_rotated_batch = np.array(image_rotated_batch)

        if self.augment:
            image_origin_batch = self.augmentation(image_origin_batch, training=True)
            image_rotated_batch = self.augmentation(image_rotated_batch, training=True)

        # Use appropriate preprocessing based on model type
        if self.model_type == 'clip':
            # Normalize to [0,1] for CLIP only
            image_origin_batch = image_origin_batch / 255.0
            image_rotated_batch = image_rotated_batch / 255.0

            # Clip values to ensure they stay in [0,1] range after augmentation
            image_origin_batch = np.clip(image_origin_batch, 0.0, 1.0)
            image_rotated_batch = np.clip(image_rotated_batch, 0.0, 1.0)

            # CLIP's processor handles the normalization
            image_origin_batch = self.processor(images=image_origin_batch, return_tensors="np", do_rescale=False)['pixel_values']
            image_rotated_batch = self.processor(images=image_rotated_batch, return_tensors="np", do_rescale=False)['pixel_values']
        else:
            # MobileNetV2 preprocessing for other models
            image_origin_batch = preprocess_input(image_origin_batch)
            image_rotated_batch = preprocess_input(image_rotated_batch)


        return [image_origin_batch, image_rotated_batch], np.array(angle_batch)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def create_augmentation_layer():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ])

def create_clip_siamese_model(input_shape=(96, 96, 3)):
    input_image1 = layers.Input(shape=input_shape)
    input_image2 = layers.Input(shape=input_shape)

    # # Load both the processor and model
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    class CLIPVisionLayer(layers.Layer):
        def __init__(self, processor, model, **kwargs):
            super().__init__(**kwargs)
            # self.processor = processor
            self.model = model
            # Move model to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            
        def call(self, inputs, training=False):
            def process_batch(x):
                # Images are already preprocessed by the generator
                x_torch = torch.from_numpy(x.numpy())
                # move to GPU if available
                if torch.cuda.is_available():
                    x_torch = x_torch.cuda()
                
                # Process entire batch in one forward pass
                with torch.no_grad():
                    outputs = self.model.get_image_features(x_torch)
                    if torch.cuda.is_available():
                        outputs = outputs.cpu()
                    # move back to CPU for numpy comversion
                    outputs = outputs.numpy()
                    return outputs
            
            processed_tensor = tf.py_function(
                process_batch,
                [inputs],
                tf.float32
            )
            # Set the shape explicitly
            processed_tensor.set_shape((None, 512))  # CLIP's output dimension
            return processed_tensor
        
        def get_config(self):
            config = super().get_config()
            print("\nBase config from parent Layer:", config)
            return config
        
        @classmethod
        def from_config(cls, config):
            print("\nReceived config for rebuilding:", config)
            # When loading, we'll recreate the CLIP model
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            return cls(processor, model, **config)

    
    vision_layer = CLIPVisionLayer(processor, clip)
    
    features1 = vision_layer(input_image1)
    features2 = vision_layer(input_image2)

    concat = layers.Concatenate()([features1, features2])
    x = layers.Dense(512, activation='relu')(concat)
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(1, activation='linear')(x)

    model = Model(inputs=[input_image1, input_image2], outputs=output)
    return model


def create_cnn_subnetwork(input_shape):
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu')
    ])
    return model


def create_siamese_network_cnn(input_shape=(96, 96, 3)):
    input1 = layers.Input(shape=input_shape)
    input2 = layers.Input(shape=input_shape)

    cnn = create_cnn_subnetwork(input_shape=input_shape)

    f1 = cnn(input1)
    f2 = cnn(input2)

    # change this to concat to test the difference
    diff = layers.Subtract()([f1, f2])

    x = layers.Dense(128, activation='relu')(diff)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(1)(x)

    model = Model(inputs=[input1, input2], outputs=output)
    return model





def create_siamese_model_mobilenetv2(input_shape=(96, 96, 3)):
    input_image1 = layers.Input(shape=input_shape)
    input_image2 = layers.Input(shape=input_shape)
    
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    features1 = base_model(input_image1)
    features2 = base_model(input_image2)
    
    flat1 = layers.Flatten()(features1)
    flat2 = layers.Flatten()(features2)
    
    concat = layers.Concatenate()([flat1, flat2])
    
    x = layers.Dense(512, activation='relu')(concat)
    x = layers.Dense(256, activation='relu')(x)
    
    # regression head. output is the normalized angle 0 to 1 
    output = layers.Dense(1, activation='linear')(x)
    
    model = Model(inputs=[input_image1, input_image2], outputs=output)
    
    return model


def train_siamese_model(model_type, epochs, batch_size=32):
    start_time = time.time()

    x_train, y_train, x_test, y_test = load_data()

    # For quick testing, use only a small portion of the data
    test_size = 1000  # or any small number
    x_train = x_train[:test_size]
    x_test = x_test[:test_size]
    test_batch_size = 128
    batch_size = test_batch_size
    
    train_gen = RotationPairGenerator(
        x_train, 
        batch_size=batch_size, 
        image_size=(96, 96),
        shuffle=True,
        augment=True,
        model_type=model_type.lower()
    )
    
    val_gen = RotationPairGenerator(
        x_test, 
        batch_size=batch_size, 
        image_size=(96, 96),
        shuffle=False,
        augment=False,
        model_type=model_type.lower()
    )
    
    if model_type.lower() =='mobilenetv2':
        model = create_siamese_model_mobilenetv2()
    elif model_type.lower() == 'basic_cnn':
        model = create_siamese_network_cnn()
    elif model_type.lower() =='clip':
        model = create_clip_siamese_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}. choose from: mobilenetv2, basic_cnn, clip")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='mse',  
        metrics=['mae']
    )
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            f'best_{model_type}_e{epochs}_b{batch_size}.h5',
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5
        )
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )
    
    model.save(f'{model_type}_e{epochs}_b{batch_size}.keras')

    training_time = time.time()-start_time
    print(f'\nTraining completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)')
    return model, history

def plot_training_history(history, model_type, epochs, batch_size):
    """
    Plot the training and validation loss and save to disk
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if 'mae' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f'training_history_{model_type}_e{epochs}_b{batch_size}.png')
    plt.close()


    
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Siamese network for rotation prediction')
    parser.add_argument('--model-type', type=str, 
                        choices=['mobilenetv2', 'basic_cnn', 'clip'],
                        required=True,
                        help='Type of model architecture to use')
    parser.add_argument('--epochs', type=int,
                        required=True,
                        help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    
    args = parser.parse_args()
    
    print(f"\nTraining Configuration:")
    print(f"Model Type: {args.model_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}\n")

    model, history = train_siamese_model(
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    plot_training_history(
        history, 
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    
    
    

