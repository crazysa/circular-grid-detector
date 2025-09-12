# Circular Grid Detector

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Custom](https://img.shields.io/badge/License-Custom-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/crazysa/circular-grid-detector.svg)](https://github.com/crazysa/circular-grid-detector/stargazers)

A high-performance Python library for detecting circular grid patterns in images. This library implements advanced computer vision algorithms optimized for real-time circle grid detection with sub-pixel accuracy.

## 🎯 Features

- **Fast Detection**: Optimized algorithms achieving ~0.5s processing time per image
- **High Accuracy**: Sub-pixel precision circle center detection
- **Robust Performance**: Handles various lighting conditions and image distortions
- **Flexible API**: Simple function calls or object-oriented interface
- **Debug Visualization**: Optional visualization for debugging and verification
- **Grid Sorting**: Intelligent grid organization and point ordering
- **Asymmetric Support**: Handles both symmetric and asymmetric grid patterns

## 🚀 Quick Start

### Installation

Install from source:

```bash
git clone https://github.com/crazysa/circular-grid-detector.git
cd circular-grid-detector
pip install -r requirements.txt
pip install -e .
```

### Basic Usage

```python
from circular_grid_detector import detect_grid, GridDetector

# Simple function interface
success, centers = detect_grid(
    image_path="path/to/your/image.jpg",
    grid_width=11,  # Number of circles horizontally
    grid_height=7,  # Number of circles vertically
    debug=True      # Enable visualization
)

if success:
    print(f"Detected {len(centers)} circle centers")
    for i, (x, y) in enumerate(centers):
        print(f"Circle {i}: ({x:.2f}, {y:.2f})")
else:
    print("Grid detection failed")
```

### Advanced Usage

```python
# Object-oriented interface for multiple detections
detector = GridDetector(
    n_x=11, 
    n_y=7, 
    is_asymmetric_grid=False
)

# Detect in multiple images
for image_path in image_paths:
    success, centers = detector.detect(image_path, debug=False)
    if success:
        # Process detected centers
        process_centers(centers)
```

## 📋 Requirements

- Python 3.8+
- OpenCV 4.5+
- NumPy 1.19+
- SciPy 1.7+
- Numba 0.56+

See `requirements.txt` for complete dependencies.

## 🎛️ Parameters

### `detect_grid(image_path, grid_width, grid_height, is_asymmetric=False, debug=False)`

- **image_path** (str): Path to the input image
- **grid_width** (int): Number of circles in horizontal direction
- **grid_height** (int): Number of circles in vertical direction  
- **is_asymmetric** (bool): Whether the grid is asymmetric (default: False)
- **debug** (bool): Enable debug visualization (default: False)

**Returns**: `(success: bool, centers: List[Tuple[float, float]])`

## 🔬 Algorithm Details

The detector implements a multi-stage pipeline:

1. **Image Preprocessing**: Adaptive filtering and gradient computation
2. **Blob Detection**: Morphological operations and contour analysis
3. **Shape Validation**: Ellipse fitting with eccentricity checks
4. **Duplicate Removal**: Spatial clustering with uncertainty weighting
5. **Grid Organization**: Intelligent row/column sorting with adaptive gap detection
6. **Sub-pixel Refinement**: Moment-based center computation

## 📊 Performance

Tested on various image sizes and grid configurations:

- **Processing Speed**: ~0.5 seconds per 4MP image
- **Detection Rate**: >95% success on well-lit calibration patterns
- **Accuracy**: Sub-pixel precision (±0.1 pixel typical)
- **Grid Sizes**: Tested from 3×3 to 15×11 patterns

## 🖼️ Examples

Check the `examples/` directory for sample images and usage patterns:

```bash
python examples/basic_detection.py
python examples/batch_processing.py
python examples/visualization_demo.py
```

## 🛠️ Development

### Running Tests

```bash
python test.py
```

### Building from Source

```bash
git clone https://github.com/crazysa/circular-grid-detector.git
cd circular-grid-detector
pip install -r requirements.txt
python setup.py install
```

## 📈 Benchmarks

Performance comparison on 4MP images (Intel i7, 16GB RAM):

| Grid Size | Success Rate | Avg Time (s) | Memory (MB) |
|-----------|--------------|--------------|-------------|
| 7×5       | 98.2%        | 0.31         | 45          |
| 11×7      | 95.8%        | 0.54         | 52          |
| 15×11     | 92.1%        | 0.78         | 68          |

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under a Custom License - see the [LICENSE](LICENSE) file for details. Free for non-commercial use, contact sragarwal@outlook.in for commercial licensing.

## 📚 Citation

If you use this library in your research, please cite:

```bibtex
@software{circular_grid_detector,
  title = {Circular Grid Detector: High-Performance Circle Pattern Detection},
  author = {Shubham Agarwal},
  year = {2025},
  url = {https://github.com/crazysa/circular-grid-detector},
  version = {1.0.0}
}
```

Or use the citation file: [CITATION.cff](CITATION.cff)

## 🌟 Acknowledgments

- Built with OpenCV and NumPy
- Optimized with Numba JIT compilation
- Uses advanced computer vision algorithms for robust detection

## ⭐ Support

If this library helps your project, please give it a star ⭐ on GitHub!

Found a bug? Have a feature request? Please [open an issue](https://github.com/crazysa/circular-grid-detector/issues).

## 📞 Contact

- GitHub: [@crazysa](https://github.com/crazysa)
- Email: sragarwal@outlook.in

---

**Keywords**: computer vision, circle detection, grid pattern, calibration, opencv, python, image processing, pattern recognition
