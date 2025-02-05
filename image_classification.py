import tensorflow as tf
import os
import cv2
import numpy as np
from PIL.Image import Image

DATA_DIR = "data"
IMAGE_EXTENSIONS = ["jpg", "jpeg", "bmp", "png"]

if __name__ == "__main__":
    for image_class in os.listdir(DATA_DIR):
        for image in os.listdir(os.path.join(DATA_DIR, image_class)):
            image_path = os.path.join(DATA_DIR, image_class, image)

            try:
                img = cv2.imread(image_path)

                if img is None:
                    print(f"Problem reading image: {image_path}")
                    os.remove(image_path)
                    continue

                file_extension = image.split('.')[-1].lower()

                if file_extension not in IMAGE_EXTENSIONS:
                    print(f"Image {image_path} has invalid extension {file_extension}")
                    os.remove(image_path)
            except Exception as e:
                print(f"Issue with image {image_path}: {e}")

    data = tf.keras.utils.image_dataset_from_directory(DATA_DIR)