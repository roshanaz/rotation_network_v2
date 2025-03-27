import os
import sys


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import os.path


def predict_rotation_angle(model, image1, image2, input_size=(96, 96)):
    """
    predict the rotation angle between two images using the trained Siamese model.
    """
    image1 = image1.astype("float32")
    image2 = image2.astype("float32")

    #resize to CIFAR-10 size (32x32) to match training distribution
    image1_small = tf.image.resize(image1, (32, 32))
    image2_small = tf.image.resize(image2, (32, 32))

    image1_resized = tf.image.resize_with_pad(
        image1_small, input_size[0], input_size[1]
    ).numpy()
    image2_resized = tf.image.resize_with_pad(
        image2_small, input_size[0], input_size[1]
    ).numpy()

    image1_batch = np.expand_dims(image1_resized, axis=0)
    image2_batch = np.expand_dims(image2_resized, axis=0)

    image1_preprocessed = preprocess_input(image1_batch)
    image2_preprocessed = preprocess_input(image2_batch)

    prediction = model.predict([image1_preprocessed, image2_preprocessed])

    # convert from normalized value [0,1] to degrees [0,360]
    predicted_angle = prediction.flatten() * 360

    return predicted_angle[0]


def load_model(model_path):
    if not os.path.isdir(model_path):
        print(f"Error: Model directory not found: {model_path}")
        print("Please ensure the model file is in the current directory.")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
        return model
    except (OSError, IOError) as e:
        print(f"Error loading model: {e}")
        print("\nThe model file appears to be corrupted or incomplete.")
        print("Please try to obtain a fresh copy of the model file.")
        return None


def load_images(image1_path, image2_path):
    if not os.path.isfile(image1_path):
        print(f"Error: Image file not found: {image1_path}")
        return None, None

    if not os.path.isfile(image2_path):
        print(f"Error: Image file not found: {image2_path}")
        return None, None

    try:
        image1 = np.array(Image.open(image1_path).convert("RGB"))
        image2 = np.array(Image.open(image2_path).convert("RGB"))
        return image1, image2
    except Exception as e:
        print(f"Error loading images: {e}")
        return None, None


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: uv run inference.py <image1_path> <image2_path>")
        print("Example: uv run inference.py image1.jpg image2.png")
        sys.exit(1)

    model_path = 'saved_model'
    model = load_model(model_path)
    if model is None:
        sys.exit(1)

    original_image, rotated_image = load_images(sys.argv[1], sys.argv[2])
    if original_image is None or rotated_image is None:
        sys.exit(1)

    predicted_angle = predict_rotation_angle(model, original_image, rotated_image)
    print(f"Estimated rotation angle is {predicted_angle:.2f} degrees")
