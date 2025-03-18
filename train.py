import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2  import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate


def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    return x_train, y_train, x_test, y_test

class RotationPairGenerator(tf.keras.utils.Sequence):
    def __init__(self, images, batch_size=32, rotation_angles=None, image_size=(96, 96), shuffle=True):
        self.images = images
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        
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

        image_origin_batch = preprocess_input(image_origin_batch)
        image_rotated_batch = preprocess_input(image_rotated_batch)

        return [image_origin_batch, image_rotated_batch], np.array(angle_batch)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def create_augmentation_layer():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomBrightness(0.2),
    ])


def create_siamese_model(input_shape=(96, 96, 3)):
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


def train_siamese_model(epochs=50, batch_size=32):
    x_train, y_train, x_test, y_test = load_data()
    
    train_gen = RotationPairGenerator(
        x_train, 
        batch_size=batch_size, 
        image_size=(96, 96),
        shuffle=True
    )
    
    val_gen = RotationPairGenerator(
        x_test, 
        batch_size=batch_size, 
        image_size=(96, 96),
        shuffle=False
    )
    
   
    model = create_siamese_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='mse',  
        metrics=['mae']
    )
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
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
    
    model.save('saved_model')
    model.save('final_model.h5')
    
    return model, history

def plot_training_history(history):
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
    plt.savefig('training_history.png')
    plt.close()


    
   
if __name__ == "__main__":
    model, history = train_siamese_model(epochs=1, batch_size=32)
    plot_training_history(history)

    
    
    

