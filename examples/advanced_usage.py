#!/usr/bin/env python3
"""
Advanced Circle Grid Detection Example

This example shows object-oriented usage and batch processing.
"""

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from circular_grid_detector import GridDetector

def batch_detection_example():
    """Example of processing multiple images with the same detector."""
    
    # Initialize detector once for multiple images
    detector = GridDetector(
        n_x=11,                    # 11 circles horizontally
        n_y=7,                     # 7 circles vertically  
        is_asymmetric_grid=False   # Symmetric grid
    )
    
    # Example image paths (replace with your own)
    image_paths = [
        "path/to/image1.jpg",
        "path/to/image2.jpg", 
        "path/to/image3.jpg",
    ]
    
    results = []
    total_time = 0
    
    print("Processing batch of images...")
    print("=" * 50)
    
    for i, image_path in enumerate(image_paths, 1):
        if not os.path.exists(image_path):
            print(f"[{i}/{len(image_paths)}] ⚠️  Image not found: {image_path}")
            continue
            
        start_time = time.time()
        
        try:
            success, centers = detector.detect(image_path, debug=False)
            
            elapsed = time.time() - start_time
            total_time += elapsed
            
            if success:
                print(f"[{i}/{len(image_paths)}] ✅ {os.path.basename(image_path)}")
                print(f"               Detected: {len(centers)} centers")
                print(f"               Time: {elapsed:.2f}s")
                
                results.append({
                    'path': image_path,
                    'success': True,
                    'centers': centers,
                    'time': elapsed
                })
            else:
                print(f"[{i}/{len(image_paths)}] ❌ {os.path.basename(image_path)}")
                print(f"               Detection failed")
                print(f"               Time: {elapsed:.2f}s")
                
                results.append({
                    'path': image_path,
                    'success': False,
                    'centers': [],
                    'time': elapsed
                })
                
        except Exception as e:
            print(f"[{i}/{len(image_paths)}] 💥 Error processing {image_path}: {e}")
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    print("\n" + "=" * 50)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 50)
    print(f"Total images: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print(f"Success rate: {successful/len(results)*100:.1f}%")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time: {total_time/len(results):.2f}s per image")
    
    return results

def asymmetric_grid_example():
    """Example of detecting asymmetric grids."""
    
    print("\nAsymmetric Grid Detection Example")
    print("=" * 50)
    
    # For asymmetric grids (checkerboard-like patterns)
    detector = GridDetector(
        n_x=8,                     # 8 circles horizontally
        n_y=6,                     # 6 circles vertically
        is_asymmetric_grid=True    # Asymmetric pattern
    )
    
    image_path = "path/to/asymmetric_grid.jpg"
    
    if os.path.exists(image_path):
        success, centers = detector.detect(image_path, debug=True)
        
        if success:
            print(f"✅ Asymmetric grid detected: {len(centers)} centers")
        else:
            print("❌ Asymmetric grid detection failed")
    else:
        print(f"⚠️  Example image not found: {image_path}")

def performance_comparison():
    """Compare detection performance with different settings."""
    
    print("\nPerformance Comparison")
    print("=" * 50)
    
    test_image = "path/to/test_image.jpg"
    
    if not os.path.exists(test_image):
        print(f"⚠️  Test image not found: {test_image}")
        return
    
    # Different grid sizes to test
    test_configs = [
        (7, 5, "Small grid (7x5)"),
        (11, 7, "Medium grid (11x7)"), 
        (15, 11, "Large grid (15x11)")
    ]
    
    for n_x, n_y, description in test_configs:
        detector = GridDetector(n_x, n_y)
        
        # Warm up
        detector.detect(test_image, debug=False)
        
        # Timed runs
        times = []
        for _ in range(3):
            start = time.time()
            success, centers = detector.detect(test_image, debug=False)
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        print(f"{description}: {avg_time:.3f}s average")

def main():
    """Run all examples."""
    print("Circular Grid Detector - Advanced Usage Examples")
    print("=" * 60)
    
    # Run examples
    try:
        batch_detection_example()
        asymmetric_grid_example() 
        performance_comparison()
        
    except KeyboardInterrupt:
        print("\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"💥 Error running examples: {e}")

if __name__ == "__main__":
    main()
