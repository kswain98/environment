import os
from pathlib import Path
import base64

# image_naming_convention = HighresScreenshot00000, HighresScreenshot00001

image_format = ["png", "jpg", "jpeg"]

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def get_image_encode(image_dir, start_idx=0, count=1):
    ret = []

    img_files = [str(Path(image_dir) / file) for file \
        in os.listdir(image_dir) if file.split('.')[1].lower() in image_format]
    
    img_format = img_files[0].split('.')[1].lower() if len(img_files) else ""

    if start_idx + count > len(img_files):
       count = len(img_files) - start_idx

    for file_idx in range(start_idx, count):
        file = img_files[file_idx]
        base64_image = encode_image(file)
        ret.append(base64_image)

    return ret, img_format