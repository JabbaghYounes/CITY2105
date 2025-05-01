from pathlib import Path
import random
import os
import sys
import shutil
import argparse
import re

# define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--datapath', help='Path to data folder containing image and annotation files', required=True)
parser.add_argument('--train_pct', help='Ratio of images to go to train folder (default 0.8)', default=0.8, type=float)
parser.add_argument('--test_pct', help='Ratio of images to go to test folder (default 0.1)', default=0.1, type=float)
parser.add_argument('--output_path', help='path to output folder', required=True)

args = parser.parse_args()

output_path = args.output_path
data_path = args.datapath
train_percent = args.train_pct
test_percent = args.test_pct
val_percent = 1 - (train_percent + test_percent)

# checking for valid entries
if not os.path.isdir(data_path):
    print('Error: Directory specified by --datapath not found.')
    sys.exit(1)

if not (0.01 <= train_percent <= 0.99) or not (0.01 <= test_percent <= 0.99) or (train_percent + test_percent >= 1.0):
    print('Error: Invalid percentages. Ensure train + test is less than 1, and each is between 0.01 and 0.99.')
    sys.exit(1)

# defining paths to input dataset
input_image_path = os.path.join(data_path, 'images')
input_label_path = os.path.join(data_path, 'labels')

# defining paths to train, validation, and test sets
train_img_path = os.path.join(output_path, 'train/images')
train_txt_path = os.path.join(output_path, 'train/labels')
val_img_path = os.path.join(output_path, 'validation/images')
val_txt_path = os.path.join(output_path, 'validation/labels')
test_img_path = os.path.join(output_path, 'test/images')
test_txt_path = os.path.join(output_path, 'test/labels')

# creating necessary folders
for dir_path in [train_img_path, train_txt_path, val_img_path, val_txt_path, test_img_path, test_txt_path]:
    os.makedirs(dir_path, exist_ok=True)
    print(f'Created folder: {dir_path}')

# function to grab combined images from filenames
def extract_match_key(filename):
    match = re.search(r'combined_image\d+', filename)
    return match.group(0) if match else None

# creating image/text dictionaries to store matching info
img_dict = {extract_match_key(f.name): f for f in Path(input_image_path).rglob('*') if f.suffix in ['.jpg', '.jpeg', '.png']}
txt_dict = {extract_match_key(f.name): f for f in Path(input_label_path).rglob('*') if f.suffix == '.txt'}

# finding and pairing common files using names
common_keys = set(img_dict.keys()) & set(txt_dict.keys())

# extracting valid file pairs
data_pairs = [(img_dict[key], txt_dict[key]) for key in common_keys]

print(f'Found {len(data_pairs)} valid image-label pairs.')

# shuffling dataset
random.shuffle(data_pairs)

# splitting dataset
train_size = int(len(data_pairs) * train_percent)
test_size = int(len(data_pairs) * test_percent)
val_size = len(data_pairs) - (train_size + test_size)

train_pairs = data_pairs[:train_size]
test_pairs = data_pairs[train_size:train_size + test_size]
val_pairs = data_pairs[train_size + test_size:]

print(f'Images for training: {len(train_pairs)}')
print(f'Images for validation: {len(val_pairs)}')
print(f'Images for testing: {len(test_pairs)}')

# function to copy files to respective folders
def copy_files(file_pairs, img_dest, label_dest):
    for img_path, txt_path in file_pairs:
        shutil.copy(img_path, os.path.join(img_dest, img_path.name))
        shutil.copy(txt_path, os.path.join(label_dest, txt_path.name))

copy_files(train_pairs, train_img_path, train_txt_path)
copy_files(val_pairs, val_img_path, val_txt_path)
copy_files(test_pairs, test_img_path, test_txt_path)
# completion message output
print("DATA SPLIT COMPLETE! READY FOR TRAINING.")
