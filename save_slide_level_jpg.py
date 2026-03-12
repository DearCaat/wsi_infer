#!/usr/bin/env python3
"""
Save the complete whole-slide image at a specified level as JPG.

This script reads a WSI (Whole Slide Image) file and saves the entire image
at a specified pyramid level as a JPG file.

Uses OpenSlide for maximum compatibility and no GPU dependency.

Usage:
    python save_slide_level_jpg.py --slide_path <path_to_slide> --level <level_number> --output_dir <output_directory>

Example:
    python save_slide_level_jpg.py --slide_path ./slide.svs --level 2 --output_dir ./output
    python save_slide_level_jpg.py --slide_path ./slide.svs --level 3 --quality 90
"""

import argparse
import os
from pathlib import Path
from PIL import Image
import sys

try:
    import openslide
except ImportError:
    print("Error: openslide-python is required. Install it with:")
    print("  pip install openslide-python")
    sys.exit(1)


def get_slide_name(slide_path: str) -> str:
    """Extract slide name from path (filename without extension)."""
    return os.path.splitext(os.path.basename(slide_path))[0]


def save_slide_level_as_jpg(
    slide_path: str,
    level: int,
    output_dir: str = None,
    output_filename: str = None,
    quality: int = 90,
):
    """
    Save the complete WSI at a specified pyramid level as JPG.

    Parameters
    ----------
    slide_path : str
        Path to the WSI file (e.g., .svs, .ndpi, .tif).
    level : int
        Pyramid level to extract (0 = highest resolution).
    output_dir : str, optional
        Directory to save the output JPG file.
        If None, saves to the same directory as the input file.
    output_filename : str, optional
        Name for the output JPG file.
        If None, uses the slide name + "_level{level}.jpg".
    quality : int, optional
        JPEG quality (1-95, higher = better). Default is 90.

    Returns
    -------
    str
        Path to the saved JPG file.
    
    Raises
    ------
    FileNotFoundError
        If the slide file does not exist.
    ValueError
        If the level is invalid or the slide cannot be opened.
    """
    
    # Check if file exists
    if not os.path.exists(slide_path):
        raise FileNotFoundError(f"Slide file not found: {slide_path}")
    
    # Load slide with OpenSlide
    print(f"Loading WSI: {slide_path}")
    try:
        slide = openslide.OpenSlide(slide_path)
    except Exception as e:
        raise ValueError(f"Failed to open slide file: {e}")
    
    slide_name = get_slide_name(slide_path)
    
    # Get slide information
    print(f"Slide name: {slide_name}")
    print(f"Dimensions at level 0 (highest resolution): {slide.dimensions}")
    print(f"Number of levels: {slide.level_count}")
    print(f"Level dimensions: {slide.level_dimensions}")
    print(f"Level downsamples: {slide.level_downsamples}")
    
    # Validate level
    if level < 0 or level >= slide.level_count:
        raise ValueError(
            f"Invalid level {level}. WSI has {slide.level_count} levels "
            f"(valid range: 0-{slide.level_count - 1})"
        )
    
    # Set output path
    if output_dir is None:
        output_dir = os.path.dirname(slide_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if output_filename is None:
        output_filename = f"{slide_name}_level{level}.jpg"
    
    output_path = os.path.join(output_dir, output_filename)
    
    # Get dimensions at the specified level
    level_width, level_height = slide.level_dimensions[level]
    print(f"\nExtracting level {level}")
    print(f"  Dimensions: {level_width} x {level_height}")
    print(f"  Downsample factor: {slide.level_downsamples[level]:.2f}x")
    
    # Read the entire level
    print(f"Reading the complete image at level {level}...")
    image = slide.read_region(
        location=(0, 0),
        level=level,
        size=(level_width, level_height)
    )
    
    # Convert to RGB (in case of RGBA)
    if image.mode == 'RGBA':
        rgb_img = Image.new('RGB', image.size, (255, 255, 255))
        rgb_img.paste(image, mask=image.split()[3])
        image = rgb_img
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Save as JPG
    print(f"Saving to: {output_path}")
    image.save(output_path, 'JPEG', quality=quality, optimize=False)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Successfully saved! File size: {file_size_mb:.2f} MB")
    
    # Close slide
    slide.close()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Save complete WSI at specified pyramid level as JPG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Save level 2 to default output directory
  python save_slide_level_jpg.py --slide_path ./slide.svs --level 2
  
  # Save level 1 to custom output directory with custom filename
  python save_slide_level_jpg.py --slide_path ./slide.svs --level 1 \\
      --output_dir ./output --output_filename slide_level1.jpg
  
  # Save level 3 with custom JPEG quality
  python save_slide_level_jpg.py --slide_path ./slide.svs --level 3 --quality 85
        """
    )
    
    parser.add_argument(
        '--slide_path',
        required=True,
        help='Path to the WSI file (e.g., .svs, .ndpi, .tif)'
    )
    parser.add_argument(
        '--level',
        type=int,
        required=True,
        help='Pyramid level to extract (0 = highest resolution)'
    )
    parser.add_argument(
        '--output_dir',
        default=None,
        help='Directory to save the output JPG file (default: same as input)'
    )
    parser.add_argument(
        '--output_filename',
        default=None,
        help='Name for the output file (default: {slide_name}_level{level}.jpg)'
    )
    parser.add_argument(
        '--quality',
        type=int,
        default=90,
        help='JPEG quality 1-95 (default: 90)'
    )
    
    args = parser.parse_args()
    
    try:
        output_path = save_slide_level_as_jpg(
            slide_path=args.slide_path,
            level=args.level,
            output_dir=args.output_dir,
            output_filename=args.output_filename,
            quality=args.quality,
        )
        print(f"\n✓ Complete! Saved to: {output_path}")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
