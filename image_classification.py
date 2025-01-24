import tensorflow as tf
import os
import cv2
import imghdr

DATA_DIR = "data"
IMAGE_EXTENSIONS = ["jpg", "jpeg", "bmp", "png"]

if __name__ == "__main__":
    for image_class in os.listdir(DATA_DIR):
        for image in os.listdir(os.path.join(DATA_DIR, image_class)):
            image_path = os.path.join(DATA_DIR, image_class, image)
            try:
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in IMAGE_EXTENSIONS:
                    print(f'Image not in extensions list {image_path}')
                    os.remove(image_path)
            except Exception as e:
                print(f"Issue with image {image_path}")