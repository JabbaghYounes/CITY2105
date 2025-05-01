import os
from PIL import Image
import re
import random

# input folders
folder1 = 'CITY2105-TRAINED-NEW/dataset/images/train/jabba'
folder2 = 'CITY2105-TRAINED-NEW/dataset/images/train/josh'
folder3 = 'CITY2105-TRAINED-NEW/dataset/images/train/kaarina'
folder4 = 'CITY2105-TRAINED-NEW/dataset/images/train/phil'

# output folders
output_regular = 'CITY2105-TRAINED-NEW/dataset/combined_regular'
output_shuffled = 'CITY2105-TRAINED-NEW/dataset/combined_shuffled'
os.makedirs(output_regular, exist_ok=True)
os.makedirs(output_shuffled, exist_ok=True)

# standard image size for consistency
STANDARD_SIZE = (400, 400)

# function to extract leading number from filenames
def extract_number(filename):
    match = re.match(r'(\d+)', filename)
    return int(match.group(1)) if match else None

# function to get numbered image dictionary from a folder
def get_images(folder):
    return {
        extract_number(f): f
        for f in os.listdir(folder)
        if f.lower().endswith('.jpg') and extract_number(f) is not None
    }

# loading image filenames from each folder
images1 = get_images(folder1)
images2 = get_images(folder2)
images3 = get_images(folder3)
images4 = get_images(folder4)

# printing folder info
print(f"Folder1: {len(images1)} | Folder2: {len(images2)} | Folder3: {len(images3)} | Folder4: {len(images4)}")

# pooling numbered images across all folders under common
common_numbers = sorted(set(images1) & set(images2) & set(images3) & set(images4))
if not common_numbers:
    raise ValueError("no common image numbers across all four folders")

# setting up quadrant positions (top-left, top-right, bottom-left, bottom-right)
positions = [(0, 0), (1, 0), (0, 1), (1, 1)]

# function to combine and save image
def combine_images(number, image_paths, shuffle=False, output_folder=''):
    # resize images to 400x400
    loaded_images = [Image.open(path).resize(STANDARD_SIZE) for path in image_paths]

    # shuffle images
    if shuffle:
        random.shuffle(loaded_images)

    w, h = STANDARD_SIZE
    combined = Image.new('RGB', (w * 2, h * 2), color=(255, 255, 255))

    # setting images to 2x2 grid
    for img, (px, py) in zip(loaded_images, positions):
        combined.paste(img, (px * w, py * h))

    # saving file
    filename = f'combined_image{number}.jpg'
    combined.save(os.path.join(output_folder, filename))

# processing all image numbers
for number in common_numbers:
    img_paths = [
        os.path.join(folder1, images1[number]),
        os.path.join(folder2, images2[number]),
        os.path.join(folder3, images3[number]),
        os.path.join(folder4, images4[number]),
    ]
    # creating layouts
    # regular 
    combine_images(number, img_paths, shuffle=False, output_folder=output_regular)
    print(f"[REGULAR] Saved combined_image{number}.jpg")

    # shuffled 
    combine_images(number, img_paths, shuffle=True, output_folder=output_shuffled)
    print(f"[SHUFFLED] Saved combined_image{number}.jpg")

print("\nALL IMAGES COMBINED SUCCESSFULLY!")
