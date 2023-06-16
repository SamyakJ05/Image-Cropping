import os
import cv2
import numpy as np
import random

image_directory = "<Input Training Folder>
savedirectory = "<Output Cropped Image Folder>"

crop_size = (500, 500)
num_crops = 10

def get_random_crop(image, crop_size):
    height, width = image.shape[:2]
    max_x = width - crop_size[1]
    max_y = height - crop_size[0]
    if max_x <= 0 or max_y <= 0:
        return None
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    return image[y:y+crop_size[0], x:x+crop_size[1]]


def calculate_text_percentage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    text_pixels = np.count_nonzero(binary == 0)
    total_pixels = binary.size
    return (text_pixels / total_pixels) * 100



def get_meaningful_crops(image_path, crop_size, num_crops):
    image = cv2.imread(image_path)
    cropped_images = []
    while len(cropped_images) < num_crops:
        crop = get_random_crop(image, crop_size)
        if crop is not None and np.count_nonzero(crop) > 0:
            text_percentage = calculate_text_percentage(crop)
            if text_percentage > 10: 
                cropped_images.append(crop)
    return cropped_images

image_files = os.listdir(image_directory)

for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)

    cropped_images = get_meaningful_crops(image_path, crop_size, num_crops)

    for i, crop in enumerate(cropped_images):
        cv2.imwrite(os.path.join(savedirectory, f"crop_{i+1}_{image_file}"), crop)
