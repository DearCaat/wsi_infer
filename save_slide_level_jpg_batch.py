#!/usr/bin/env python3
"""
Advanced script to save WSI slides at specified levels with batch processing support.

Uses OpenSlide for maximum compatibility and no GPU dependency.
Provides batch processing capabilities, automatic level detection, and flexible output options.

Usage:
    # Single slide
    python save_slide_level_jpg_batch.py --slide_path slide.svs --level 2 --output_dir ./output
    
    # Batch processing with glob pattern
    python save_slide_level_jpg_batch.py --input_dir ./slides --pattern "*.svs" --level 1 --output_dir ./output
    
    # Save all levels
    python save_slide_level_jpg_batch.py --slide_path slide.svs --save_all_levels --output_dir ./output
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Dict
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import openslide
except ImportError:
    print("Error: openslide-python is required. Install it with:")
    print("  pip install openslide-python")
    sys.exit(1)

from PIL import Image


def get_slide_name(slide_path: str) -> str:
    """Extract slide name from path (filename without extension)."""
    return os.path.splitext(os.path.basename(slide_path))[0]


def save_slide_level_as_jpg(
    slide_path: str,
    level: int,
    output_dir: str = None,
    output_filename: str = None,
    quality: int = 90,
    verbose: bool = True,
) -> Dict:
    """
    Save the complete WSI at a specified pyramid level as JPG.
    
    Parameters
    ----------
    slide_path : str
        Path to the WSI file.
    level : int
        Pyramid level to extract.
    output_dir : str, optional
        Output directory for the JPG file.
    output_filename : str, optional
        Custom output filename.
    quality : int, optional
        JPEG quality (default: 90).
    verbose : bool, optional
        Print progress information (default: True).
        
    Returns
    -------
    dict
        Result dictionary with keys: 'success', 'input_path', 'output_path', 'message', 'error'
    """
    result = {
        'success': False,
        'input_path': slide_path,
        'output_path': None,
        'message': '',
        'error': None,
    }
    
    try:
        # Load WSI
        if verbose:
            print(f"[Loading] {os.path.basename(slide_path)}")
        
        slide = openslide.OpenSlide(slide_path)
        slide_name = get_slide_name(slide_path)
        
        # Validate level
        if level < 0 or level >= slide.level_count:
            slide.close()
            raise ValueError(
                f"Invalid level {level}. WSI has {slide.level_count} levels"
            )
        
        # Set output path
        if output_dir is None:
            output_dir = os.path.dirname(slide_path)
        
        os.makedirs(output_dir, exist_ok=True)
        
        if output_filename is None:
            output_filename = f"{slide_name}_level{level}.jpg"
        
        output_path = os.path.join(output_dir, output_filename)
        
        # Check if output already exists
        if os.path.exists(output_path):
            slide.close()
            if verbose:
                print(f"  [Skip] Output already exists: {output_filename}")
            result['message'] = 'Output already exists'
            return result
        
        # Get level dimensions
        level_width, level_height = slide.level_dimensions[level]
        
        if verbose:
            print(f"  [Reading] Level {level} ({level_width}x{level_height})")
        
        # Read the entire level
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
        if verbose:
            print(f"  [Saving] {output_filename}")
        
        image.save(output_path, 'JPEG', quality=quality, optimize=False)
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        result['success'] = True
        result['output_path'] = output_path
        result['message'] = f"Saved {level_width}x{level_height} ({file_size_mb:.1f}MB)"
        
        if verbose:
            print(f"  ✓ {result['message']}")
        
        slide.close()
        return result
        
    except Exception as e:
        result['error'] = str(e)
        result['message'] = f"Error: {str(e)}"
        if verbose:
            print(f"  ✗ {result['message']}")
        return result


def save_all_levels(
    slide_path: str,
    output_dir: str = None,
    quality: int = 90,
    verbose: bool = True,
) -> List[Dict]:
    """
    Save all pyramid levels of a WSI as separate JPG files.
    
    Parameters
    ----------
    slide_path : str
        Path to the WSI file.
    output_dir : str, optional
        Output directory.
    quality : int, optional
        JPEG quality (default: 90).
    verbose : bool, optional
        Print progress (default: True).
        
    Returns
    -------
    list of dict
        List of result dictionaries for each level.
    """
    results = []
    
    try:
        slide = openslide.OpenSlide(slide_path)
        slide_name = get_slide_name(slide_path)
        
        if verbose:
            print(f"\nProcessing: {os.path.basename(slide_path)}")
            print(f"Total levels: {slide.level_count}")
            print(f"Level dimensions: {slide.level_dimensions}")
            print(f"Level downsamples: {[f'{d:.2f}x' for d in slide.level_downsamples]}\n")
        
        for level in range(slide.level_count):
            output_filename = f"{slide_name}_level{level}.jpg"
            result = save_slide_level_as_jpg(
                slide_path=slide_path,
                level=level,
                output_dir=output_dir,
                output_filename=output_filename,
                quality=quality,
                verbose=verbose,
            )
            results.append(result)
        
        slide.close()
        return results
        
    except Exception as e:
        print(f"Error processing {slide_path}: {e}")
        return []


def find_slide_files(input_dir: str, pattern: str = "*.svs") -> List[str]:
    """
    Find slide files matching the given pattern.
    
    Parameters
    ----------
    input_dir : str
        Input directory.
    pattern : str, optional
        File pattern (default: "*.svs").
        
    Returns
    -------
    list of str
        List of matching slide file paths.
    """
    input_path = Path(input_dir)
    slides = sorted(input_path.glob(pattern))
    return [str(s) for s in slides]


def batch_process_slides(
    slide_paths: List[str],
    level: int,
    output_dir: str,
    quality: int = 90,
    num_workers: int = 1,
    verbose: bool = True,
):
    """
    Process multiple slide files in parallel.
    
    Parameters
    ----------
    slide_paths : list of str
        List of slide file paths.
    level : int
        Pyramid level to extract.
    output_dir : str
        Output directory.
    quality : int, optional
        JPEG quality (default: 95).
    num_workers : int, optional
        Number of parallel workers (default: 1).
    verbose : bool, optional
        Print progress (default: True).
        
    Returns
    -------
    list of dict
        List of result dictionaries.
    """
    results = []
    
    if verbose:
        print(f"\nProcessing {len(slide_paths)} slides...")
        print(f"Using {num_workers} worker(s)\n")
    
    if num_workers == 1:
        # Sequential processing
        for slide_path in slide_paths:
            result = save_slide_level_as_jpg(
                slide_path=slide_path,
                level=level,
                output_dir=output_dir,
                quality=quality,
                verbose=verbose,
            )
            results.append(result)
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    save_slide_level_as_jpg,
                    slide_path=slide_path,
                    level=level,
                    output_dir=output_dir,
                    quality=quality,
                    verbose=verbose,
                ): slide_path
                for slide_path in slide_paths
            }
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
    
    # Summary
    if verbose:
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        print(f"\n{'='*60}")
        print(f"Summary: {successful}/{total} files processed successfully")
        print(f"{'='*60}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Advanced WSI level extraction with batch processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single slide
  python save_slide_level_jpg_batch.py --slide_path ./slide.svs --level 2 --output_dir ./output
  
  # Batch processing
  python save_slide_level_jpg_batch.py --input_dir ./slides --pattern "*.svs" \\
      --level 1 --output_dir ./output --num_workers 4
  
  # Save all levels
  python save_slide_level_jpg_batch.py --slide_path ./slide.svs --save_all_levels \\
      --output_dir ./output
        """
    )
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--slide_path', help='Path to a single WSI file')
    input_group.add_argument('--input_dir', help='Directory containing WSI files')
    
    parser.add_argument('--pattern', default='*.svs', 
                        help='File pattern for batch processing (default: *.svs)')
    parser.add_argument('--level', type=int, help='Pyramid level to extract')
    parser.add_argument('--save_all_levels', action='store_true',
                        help='Save all pyramid levels (ignores --level)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--quality', type=int, default=90,
                        help='JPEG quality 1-95 (default: 90)')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    
    args = parser.parse_args()
    
    # Validation
    if args.input_dir and not args.level and not args.save_all_levels:
        print("Error: When using --input_dir, specify either --level or --save_all_levels")
        return 1
    
    if args.slide_path and not args.save_all_levels and args.level is None:
        print("Error: When using --slide_path, specify either --level or --save_all_levels")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    start_time = time.time()
    
    try:
        # Single slide processing
        if args.slide_path:
            if not os.path.exists(args.slide_path):
                print(f"Error: File not found: {args.slide_path}")
                return 1
            
            if args.save_all_levels:
                results = save_all_levels(
                    slide_path=args.slide_path,
                    output_dir=args.output_dir,
                    quality=args.quality,
                    verbose=True,
                )
            else:
                result = save_slide_level_as_jpg(
                    slide_path=args.slide_path,
                    level=args.level,
                    output_dir=args.output_dir,
                    quality=args.quality,
                    verbose=True,
                )
                results = [result]
        
        # Batch processing
        else:
            slide_paths = find_slide_files(args.input_dir, args.pattern)
            
            if not slide_paths:
                print(f"No files found matching pattern '{args.pattern}' in '{args.input_dir}'")
                return 1
            
            print(f"Found {len(slide_paths)} slide(s)")
            
            if args.save_all_levels:
                results = []
                for slide_path in slide_paths:
                    slide_results = save_all_levels(
                        slide_path=slide_path,
                        output_dir=args.output_dir,
                        quality=args.quality,
                        verbose=True,
                    )
                    results.extend(slide_results)
            else:
                results = batch_process_slides(
                    slide_paths=slide_paths,
                    level=args.level,
                    output_dir=args.output_dir,
                    quality=args.quality,
                    num_workers=args.num_workers,
                    verbose=True,
                )
        
        # Final summary
        successful = sum(1 for r in results if r['success'])
        elapsed = time.time() - start_time
        
        print(f"\n✓ Done! Processed {successful} file(s) in {elapsed:.1f} seconds")
        return 0 if successful == len(results) else 1
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
