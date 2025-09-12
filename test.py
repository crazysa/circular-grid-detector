#!/usr/bin/env python3

import os
import sys
import time
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circular_grid_detector import detect_grid


def test_single_image(image_path: str, grid_width: int, grid_height: int, debug: bool = False) -> Tuple[bool, int, float]:
    print(f"Testing: {os.path.basename(image_path)}")
    
    start_time = time.time()
    try:
        success, centers = detect_grid(image_path, grid_width, grid_height, 
                                     is_asymmetric=False, debug=debug)
        processing_time = time.time() - start_time
        
        num_detected = len(centers)
        expected_count = grid_width * grid_height
        
        print(f"  Result: {'SUCCESS' if success else 'PARTIAL'}")
        print(f"  Detected: {num_detected}/{expected_count}")
        print(f"  Time: {processing_time:.2f}s")
        
        return success, num_detected, processing_time
            
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"  ERROR: {str(e)}")
        return False, 0, processing_time


def test_grid_detection():
    test_folder = "/home/shub/Downloads/circular_grid"
    grid_width = 11
    grid_height = 7
    
    print(f"Grid Detection Test - {grid_width}x{grid_height}")
    print(f"Test folder: {test_folder}")
    print("=" * 60)
    
    if not os.path.exists(test_folder):
        print(f"Error: Test folder '{test_folder}' does not exist!")
        return
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    original_images = []
    
    for file in os.listdir(test_folder):
        if not ('_detection' in file or '_circle_grid' in file or '_ultra_fast' in file):
            file_path = os.path.join(test_folder, file)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(file.lower())
                if ext in image_extensions:
                    original_images.append(file_path)
    
    if not original_images:
        print("No original image files found!")
        return
    
    print(f"Found {len(original_images)} original images")
    print()
    
    successful_detections = 0
    total_images = len(original_images)
    total_time = 0
    results = []
    
    for i, image_path in enumerate(original_images, 1):
        print(f"[{i}/{total_images}] ", end="")
        success, num_detected, processing_time = test_single_image(image_path, grid_width, grid_height, debug=True)
        
        results.append({
            'image': os.path.basename(image_path),
            'success': success,
            'detected': num_detected,
            'time': processing_time
        })
        
        if success:
            successful_detections += 1
        
        total_time += processing_time
        print()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Images tested: {total_images}")
    print(f"Successful detections: {successful_detections}")
    print(f"Success rate: {successful_detections/total_images*100:.1f}%")
    print(f"Average time: {total_time/total_images:.2f}s")
    print(f"Processing speed: {1/(total_time/total_images):.1f} images/second")
    print()
    
    if successful_detections >= 5:
        print("✓ REQUIREMENT MET: Detected grid in at least 5 files!")
    else:
        print(f"✗ REQUIREMENT NOT MET: Only detected in {successful_detections} files (need at least 5)")
    
    print("\nResults:")
    print("-" * 60)
    for result in results:
        status = "✓" if result['success'] else "○"
        print(f"{status} {result['image']:40} {result['detected']:2d}/{grid_width*grid_height:2d} {result['time']:5.2f}s")
    
    print(f"\nVisualization saved in: {test_folder}/visualization/")


if __name__ == "__main__":
    test_grid_detection()
