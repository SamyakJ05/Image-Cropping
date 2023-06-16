import os
import random
import cv2
import numpy as np
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'tesseract.exe'


def has_text_ocr(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    denoised = cv2.fastNlMeansDenoising(binary, h=10)

    
    text = pytesseract.image_to_string(denoised)
    return len(text) > 0

def get_random_crops_with_text(directory, crop_size, num_crops):
    image_paths = [os.path.join(directory, file) for file in os.listdir(directory) if
                   file.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_paths)

    crops = []
    for image_path in image_paths:
        if len(crops) >= num_crops:
            break
        if not has_text_ocr(image_path):
            continue

        img = Image.open(image_path)
        width, height = img.size
        if width < crop_size or height < crop_size:
            continue

        for _ in range(10): 
            left = random.randint(0, width - crop_size)
            top = random.randint(0, height - crop_size)
            right = left + crop_size
            bottom = top + crop_size

            crop = img.crop((left, top, right, bottom))
            if crop.getbbox() is None:
                continue

            np_array = np.array(crop)
            if np.mean(np_array) < 240:
                continue

            crops.append(crop)
            break

    return crops[:num_crops]

directory = '<image path>'
crop_size = 500
num_crops = 10

crops = get_random_crops_with_text(directory, crop_size, num_crops)

for i, crop in enumerate(crops):
    crop.save(f'crop_{i+1}.png')
