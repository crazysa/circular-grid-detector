#!/usr/bin/env python3
"""
Basic Circle Grid Detection Example

This example demonstrates the simplest usage of the circular grid detector.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from circular_grid_detector import detect_grid

def main():
    # Example image path (replace with your own)
    image_path = "path/to/your/circular_grid_image.jpg"
    
    # Basic detection with visualization
    print("Detecting circular grid pattern...")
    success, centers = detect_grid(
        image_path=image_path,
        grid_width=11,    # 11 circles horizontally
        grid_height=7,    # 7 circles vertically
        debug=True        # Enable visualization
    )
    
    if success:
        print(f"✅ Success! Detected {len(centers)} circle centers")
        
        # Print first few centers
        print("\nFirst 5 detected centers:")
        for i, (x, y) in enumerate(centers[:5]):
            print(f"  Circle {i}: ({x:.2f}, {y:.2f})")
        
        print(f"\nVisualization saved to: visualization/ folder")
        
    else:
        print("❌ Detection failed - no grid pattern found")
        
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
