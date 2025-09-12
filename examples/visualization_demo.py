#!/usr/bin/env python3
"""
Visualization Demo

This example demonstrates the debug visualization features of the detector.
"""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from circular_grid_detector import detect_grid, GridDetector

def create_synthetic_grid(width=800, height=600, grid_x=11, grid_y=7):
    """Create a synthetic circular grid for testing."""
    
    # Create blank image
    img = np.ones((height, width), dtype=np.uint8) * 240
    
    # Calculate grid spacing
    margin_x = width * 0.1
    margin_y = height * 0.1
    
    spacing_x = (width - 2 * margin_x) / (grid_x - 1)
    spacing_y = (height - 2 * margin_y) / (grid_y - 1)
    
    # Draw circles
    radius = min(spacing_x, spacing_y) * 0.3
    
    for row in range(grid_y):
        for col in range(grid_x):
            center_x = int(margin_x + col * spacing_x)
            center_y = int(margin_y + row * spacing_y)
            
            # Draw filled circle
            cv2.circle(img, (center_x, center_y), int(radius), 0, -1)
    
    return img

def visualization_comparison_demo():
    """Compare detection with and without debug visualization."""
    
    print("Visualization Demo")
    print("=" * 50)
    
    # Create synthetic test image
    print("Creating synthetic grid image...")
    synthetic_img = create_synthetic_grid(800, 600, 11, 7)
    test_image_path = "synthetic_grid_test.jpg"
    cv2.imwrite(test_image_path, synthetic_img)
    print(f"Saved synthetic image: {test_image_path}")
    
    # Detection without visualization
    print("\n1. Detection without debug visualization...")
    success1, centers1 = detect_grid(
        test_image_path, 
        grid_width=11, 
        grid_height=7, 
        debug=False
    )
    
    if success1:
        print(f"✅ Detected {len(centers1)} centers (no visualization)")
    else:
        print("❌ Detection failed (no visualization)")
    
    # Detection with visualization  
    print("\n2. Detection with debug visualization...")
    success2, centers2 = detect_grid(
        test_image_path,
        grid_width=11,
        grid_height=7, 
        debug=True
    )
    
    if success2:
        print(f"✅ Detected {len(centers2)} centers (with visualization)")
        print("📁 Check 'visualization/' folder for debug images")
    else:
        print("❌ Detection failed (with visualization)")
        print("📁 Check 'visualization/' folder for failure analysis")
    
    # Show what visualization includes
    print("\n🎨 Debug Visualization Features:")
    print("   • Green circles with numbers: Successfully detected grid points")
    print("   • Blue contours: Candidate blob contours (on failure)")
    print("   • Yellow dots: Candidate blob centers (on failure)")
    print("   • Status text: SUCCESS/FAIL with point count")
    print("   • Candidate count: Total blobs found before filtering")

def different_grid_sizes_demo():
    """Test visualization with different grid sizes."""
    
    print("\nTesting Different Grid Sizes")
    print("=" * 50)
    
    test_configs = [
        (5, 4, "Small grid"),
        (11, 7, "Standard grid"),
        (9, 6, "Medium grid")
    ]
    
    for i, (grid_x, grid_y, description) in enumerate(test_configs):
        print(f"\n{i+1}. {description} ({grid_x}x{grid_y}):")
        
        # Create synthetic image for this grid size
        img = create_synthetic_grid(800, 600, grid_x, grid_y)
        test_path = f"test_grid_{grid_x}x{grid_y}.jpg"
        cv2.imwrite(test_path, img)
        
        # Detect with visualization
        success, centers = detect_grid(
            test_path,
            grid_width=grid_x,
            grid_height=grid_y,
            debug=True
        )
        
        expected_points = grid_x * grid_y
        if success and len(centers) == expected_points:
            print(f"   ✅ Perfect detection: {len(centers)}/{expected_points}")
        elif success:
            print(f"   ⚠️  Partial detection: {len(centers)}/{expected_points}")
        else:
            print(f"   ❌ Detection failed: 0/{expected_points}")
        
        print(f"   📁 Visualization: visualization/{os.path.splitext(os.path.basename(test_path))[0]}_detection.jpg")

def failed_detection_demo():
    """Demonstrate what happens when detection fails."""
    
    print("\nFailed Detection Visualization Demo")
    print("=" * 50)
    
    # Create an image that should fail detection
    print("Creating challenging test image...")
    
    # Image with too few circles
    img = np.ones((600, 800), dtype=np.uint8) * 240
    
    # Add only a few random circles (not in grid pattern)
    for _ in range(8):
        x = np.random.randint(100, 700)
        y = np.random.randint(100, 500)
        radius = np.random.randint(15, 35)
        cv2.circle(img, (x, y), radius, 0, -1)
    
    fail_test_path = "failing_test_image.jpg"
    cv2.imwrite(fail_test_path, img)
    
    # Try to detect 11x7 grid (should fail)
    print("Attempting detection on challenging image...")
    success, centers = detect_grid(
        fail_test_path,
        grid_width=11,
        grid_height=7,
        debug=True
    )
    
    if not success:
        print("❌ Detection failed as expected")
        print("📁 Check visualization to see:")
        print("   • Blue contours around detected blob candidates")
        print("   • Yellow dots at candidate centers")
        print("   • 'DETECTION FAIL' status with candidate count")
    else:
        print(f"⚠️  Unexpected success: {len(centers)} points detected")

def cleanup_test_files():
    """Clean up generated test files."""
    
    test_files = [
        "synthetic_grid_test.jpg",
        "test_grid_5x4.jpg",
        "test_grid_11x7.jpg", 
        "test_grid_9x6.jpg",
        "failing_test_image.jpg"
    ]
    
    print("\nCleaning up test files...")
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"   Removed: {file}")

def main():
    """Run visualization demo."""
    
    print("🎨 Circular Grid Detector - Visualization Demo")
    print("=" * 60)
    
    try:
        visualization_comparison_demo()
        different_grid_sizes_demo()
        failed_detection_demo()
        
        print("\n" + "=" * 60)
        print("Demo completed! Check the 'visualization/' folder for debug images.")
        
    except Exception as e:
        print(f"💥 Error in demo: {e}")
    
    finally:
        cleanup_test_files()

if __name__ == "__main__":
    main()
