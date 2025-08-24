import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

# Input and output directories
input_dir = Path("/rds/general/user/mb1024/home/SA/extracted_1000_tiffs/")
output_dir = Path("/rds/general/user/mb1024/home/SA/TIFF_processed_200x200/")
output_dir.mkdir(parents=True, exist_ok=True)

# Target size
target_width = 200
target_height = 200

def process_and_save_tiff(src_path, dst_path):
    with Image.open(src_path) as img:
        # Convert to 8-bit grayscale
        img = img.point(lambda x: x * (255.0 / 65535)).convert("L")

        # Downsample proportionally first
        new_width = int(img.width / 2.25)
        new_height = int(img.height / 2.25)
        img = img.resize((new_width, new_height), Image.BILINEAR)

        # Padding if needed
        pad_w = max(target_width - img.width, 0)
        pad_h = max(target_height - img.height, 0)
        if pad_w > 0 or pad_h > 0:
            padding = (
                pad_w // 2, pad_h // 2,
                pad_w - pad_w // 2, pad_h - pad_h // 2
            )
            mean_val = int(np.mean(np.array(img)))
            img = ImageOps.expand(img, padding, fill=mean_val)

        # Final center crop to 200x200
        left = (img.width - target_width) // 2
        top = (img.height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        img_cropped = img.crop((left, top, right, bottom))

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img_cropped.save(dst_path)

if __name__ == "__main__":
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith((".tiff", ".tif")):
                src_path = Path(root) / fname
                rel_path = src_path.relative_to(input_dir)
                dst_path = output_dir / rel_path
                process_and_save_tiff(src_path, dst_path)
