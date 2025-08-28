import os
from PIL import Image, ImageOps
from pathlib import Path

"""
GeoTIFF Preprocessing Script
============================

This script preprocesses a labelled GeoTIFF dataset for downstream 
machine learning tasks (e.g., SSL transfer learning).
"""

# Directories
# Path to labelled GeoTIFF dataset (input images)
input_dir = Path('/path/to/GeoTIFF/')
# Path to store processed dataset
output_dir = Path('/path/to/store/TIFF_processed/') 
output_dir.mkdir(parents=True, exist_ok=True)

target_width = 200
target_height = 200

for subdir, _, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith('.tiff'):
            src_path = Path(subdir) / file
            rel_path = src_path.relative_to(input_dir)
            dst_path = output_dir / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                with Image.open(src_path) as img:
                    # Convert 16-bit to 8-bit grayscale
                    img = img.point(lambda x: x * (255.0 / 65535)).convert('L')

                    # Downsample
                    new_width = int(img.width / 2.25)
                    new_height = int(img.height / 2.25)
                    img = img.resize((new_width, new_height), Image.BILINEAR)

                    # Pad if needed
                    pad_w = max(target_width - img.width, 0)
                    pad_h = max(target_height - img.height, 0)
                    if pad_w > 0 or pad_h > 0:
                        padding = (
                            pad_w // 2, pad_h // 2,  # left, top
                            pad_w - pad_w // 2, pad_h - pad_h // 2  # right, bottom
                        )
                        mean_val = int(np.mean(np.array(img)))

                        # Apply padding with the mean value
                        img = ImageOps.expand(img, padding, fill=mean_val)  

                    # Center crop
                    left = (img.width - target_width) // 2
                    top = (img.height - target_height) // 2
                    right = left + target_width
                    bottom = top + target_height
                    img_cropped = img.crop((left, top, right, bottom))

                    img_cropped.save(dst_path)

            except Exception as e:
                print(f"Error processing {src_path}: {e}")

if __name__ == "__main__":
    for subdir, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.tiff'):
                src_path = Path(subdir) / file
                rel_path = src_path.relative_to(input_dir)
                dst_path = output_dir / rel_path

                try:
                    process_and_save_tiff(src_path, dst_path)
                except Exception as e:
                    print(f"Error processing {src_path}: {e}")