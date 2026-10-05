#!/usr/bin/env python3
"""
Video Converter Tool
Converts input videos (e.g., .webm, .mov, .avi) to highly compatible, web-optimized MP4 (H.264) format.
Automatically scales the video to even dimensions (required for H.264 encoding) and handles audio/no-audio inputs.

Usage:
    python3 convert_video.py <input_video_path> [output_video_path]
"""

import sys
import os
import subprocess
import json

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: 'ffmpeg' or 'ffprobe' is not installed or not in PATH.", file=sys.stderr)
        print("Please install ffmpeg before running this script.", file=sys.stderr)
        sys.exit(1)

def has_audio_stream(input_path):
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-select_streams", "a", 
        "-show_entries", "stream=codec_name", 
        "-of", "json", 
        input_path
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        info = json.loads(result.stdout)
        return len(info.get("streams", [])) > 0
    except Exception as e:
        print(f"Warning: Failed to probe audio streams ({e}). Assuming no audio.")
        return False

def convert_video(input_path, output_path=None):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Determine output path if not provided
    if not output_path:
        base, _ = os.path.splitext(input_path)
        output_path = base + ".mp4"

    print(f"🎬 Probing video: {input_path}")
    has_audio = has_audio_stream(input_path)
    print(f"🔊 Audio track detected: {has_audio}")

    # Build ffmpeg command
    # Scale filter ensures width and height are even numbers (trunc(iw/2)*2)
    # format=yuv420p ensures compatibility with standard browser players
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-vf", "scale='2*trunc(iw/2)':'2*trunc(ih/2)',format=yuv420p",
        "-profile:v", "high",
        "-level:v", "4.0",
        "-movflags", "+faststart", # Optimize for streaming / quick load in browser
        "-y" # Overwrite output if exists
    ]

    if has_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "128k"])
    else:
        cmd.extend(["-an"]) # Disable audio channel if no audio in source

    cmd.append(output_path)

    print(f"🚀 Running conversion command:\n{' '.join(cmd)}")
    try:
        # Run command and pipe output to show progress
        subprocess.run(cmd, check=True)
        print(f"✅ Success! Web-optimized MP4 created at: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during conversion: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 convert_video.py <input_video_path> [output_video_path]")
        sys.exit(1)

    check_ffmpeg()
    
    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_video(in_path, out_path)
