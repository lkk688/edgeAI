#!/usr/bin/env python3
"""
Video to Animated WebP Converter Tool
Converts input videos (e.g., .mp4, .webm, .mov) to highly compressed, animated WebP files.
Perfect for embedding silent screen recordings in slides and documentations.

Usage:
    python3 convert_to_webp.py <input_video_path> [output_webp_path] [fps] [scale_width] [quality] [speed]
"""

import sys
import os
import subprocess
import tempfile
import shutil
from PIL import Image

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: 'ffmpeg' is not installed or not in PATH.", file=sys.stderr)
        print("Please install ffmpeg before running this script.", file=sys.stderr)
        sys.exit(1)

def has_ffmpeg_webp_encoder():
    try:
        result = subprocess.run(["ffmpeg", "-encoders"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return "libwebp" in result.stdout
    except Exception:
        return False

def convert_to_webp(input_path, output_path=None, fps=10, scale_width=800, quality=75, speed=1.0):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Determine output path if not provided
    if not output_path:
        base, _ = os.path.splitext(input_path)
        output_path = base + ".webp"

    print(f"🎬 Converting: {input_path} -> {output_path}")
    print(f"📈 Settings: FPS={fps}, Scale Width={scale_width}px, Quality={quality}, Speed={speed}x")

    use_native = has_ffmpeg_webp_encoder()
    
    if use_native:
        print("🚀 Using native ffmpeg libwebp encoder...")
        # Adjust presentation timestamps (setpts) for speedup, then output at target fps
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-vf", f"setpts=PTS/{speed},fps={fps},scale={scale_width}:-1:flags=lanczos",
            "-loop", "0",
            "-vcodec", "libwebp",
            "-lossless", "0",
            "-compression_level", "4",
            "-q:v", str(quality),
            "-an",
            "-y",
            output_path
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error during native conversion: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("⚠️  ffmpeg lacks 'libwebp' encoder. Falling back to Pillow (PIL) frame compiler...")
        temp_dir = tempfile.mkdtemp()
        try:
            # Extract frames as PNG using ffmpeg at target FPS
            print("🎥 Extracting video frames...")
            extract_cmd = [
                "ffmpeg",
                "-i", input_path,
                "-vf", f"fps={fps},scale={scale_width}:-1",
                "-y",
                os.path.join(temp_dir, "frame_%05d.png")
            ]
            subprocess.run(extract_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            # Read extracted frames
            frame_files = sorted([
                os.path.join(temp_dir, f) 
                for f in os.listdir(temp_dir) 
                if f.endswith(".png")
            ])
            
            if not frame_files:
                raise RuntimeError("No frames could be extracted from the video.")
                
            # Sample frames based on speed
            step = float(speed)
            selected_files = []
            idx = 0.0
            while int(idx) < len(frame_files):
                selected_files.append(frame_files[int(idx)])
                idx += step
                
            print(f"🖼️  Compiling {len(selected_files)} frames (sampled from {len(frame_files)} total) into animated WebP...")
            frames = [Image.open(f) for f in selected_files]
            
            # Save as animated webp
            duration = int(1000 / fps)
            frames[0].save(
                output_path,
                format="WEBP",
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,
                quality=quality,
                method=4
            )
            
            # Close images
            for img in frames:
                img.close()
                
            print("🧹 Cleaning up temporary files...")
        except Exception as e:
            print(f"❌ Error during fallback conversion: {e}", file=sys.stderr)
            sys.exit(1)
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    # Success output & comparison
    if os.path.exists(output_path):
        print(f"✅ Success! Animated WebP created at: {output_path}")
        orig_size = os.path.getsize(input_path) / (1024 * 1024)
        new_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"📦 Size comparison: {orig_size:.2f} MB -> {new_size:.2f} MB")
        return output_path
    else:
        print("❌ Error: WebP file was not created.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 convert_to_webp.py <input_video_path> [output_webp_path] [fps] [scale_width] [quality] [speed]")
        sys.exit(1)

    check_ffmpeg()
    
    in_path = sys.argv[1]
    
    # Parse arguments
    out_path = None
    fps = 10
    scale_width = 800
    quality = 75
    speed = 1.0
    
    # Check if the second argument is a path
    args = sys.argv[2:]
    if args and (args[0].endswith('.webp') or '/' in args[0] or '\\' in args[0]):
        out_path = args.pop(0)
        
    # Now args contains only numerical parameters: [fps, scale_width, quality, speed]
    if len(args) >= 1:
        fps = int(args[0])
    if len(args) >= 2:
        scale_width = int(args[1])
    if len(args) >= 3:
        quality = int(args[2])
    if len(args) >= 4:
        speed = float(args[3])
        
    convert_to_webp(in_path, out_path, fps, scale_width, quality, speed)

