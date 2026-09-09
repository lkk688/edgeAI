#!/usr/bin/env python3
import os
import argparse
from PIL import Image

def compress_image(file_path, max_size=1000, quality=85, in_place=True, out_dir=None):
    try:
        orig_size = os.path.getsize(file_path)
        with Image.open(file_path) as img:
            # Handle EXIF orientation
            try:
                exif = img._getexif()
                if exif is not None:
                    # 0x0112 is the Orientation tag
                    orientation = exif.get(0x0112)
                    if orientation == 3:
                        img = img.rotate(180, expand=True)
                    elif orientation == 6:
                        img = img.rotate(270, expand=True)
                    elif orientation == 8:
                        img = img.rotate(90, expand=True)
            except Exception:
                pass

            # Calculate target dimensions maintaining aspect ratio
            width, height = img.size
            if max(width, height) > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(f"• Resizing {os.path.basename(file_path)} from {width}x{height} to {new_width}x{new_height}")
            
            # Determine output path
            if in_place:
                save_path = file_path
            else:
                os.makedirs(out_dir, exist_ok=True)
                save_path = os.path.join(out_dir, os.path.basename(file_path))

            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.jpg', '.jpeg']:
                img.save(save_path, 'JPEG', quality=quality, optimize=True)
            elif ext == '.png':
                # Check for transparency/alpha channel
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    img.save(save_path, 'PNG', optimize=True)
                else:
                    img.save(save_path, 'PNG', optimize=True)
                    
        new_size = os.path.getsize(save_path)
        reduction = (orig_size - new_size) / orig_size * 100
        print(f"Compressed {os.path.basename(file_path)}: {orig_size/1024:.1f}KB -> {new_size/1024:.1f}KB ({reduction:.1f}% reduction)")
    except Exception as e:
        print(f"Error compressing {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="General purpose image compression and resizing script using Pillow.")
    parser.add_argument("--dir", default=".", help="Directory containing images to compress (default: current directory)")
    parser.add_argument("--file", help="Path to a single image file to compress")
    parser.add_argument("--max-size", type=int, default=1000, help="Maximum dimension (width or height) of resized images (default: 1000)")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality factor (default: 85)")
    parser.add_argument("--out-dir", help="Output directory (if not compressing in-place)")
    
    args = parser.parse_args()
    
    in_place = args.out_dir is None
    
    if args.file:
        if os.path.exists(args.file):
            compress_image(args.file, args.max_size, args.quality, in_place, args.out_dir)
        else:
            print(f"File not found: {args.file}")
    else:
        if not os.path.exists(args.dir):
            print(f"Directory not found: {args.dir}")
            return
            
        extensions = ('.jpg', '.jpeg', '.png')
        for f in os.listdir(args.dir):
            if f.lower().endswith(extensions):
                full_path = os.path.join(args.dir, f)
                compress_image(full_path, args.max_size, args.quality, in_place, args.out_dir)

if __name__ == "__main__":
    main()
