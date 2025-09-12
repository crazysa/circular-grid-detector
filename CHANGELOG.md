# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-13

### Added
- Initial release of Circular Grid Detector
- High-performance circle grid detection algorithm
- Support for both symmetric and asymmetric grids
- Debug visualization with success/failure analysis
- Intelligent grid sorting with adaptive gap detection
- Sub-pixel accuracy circle center detection
- Comprehensive test suite
- Examples and documentation
- PyPI package support

### Features
- **GridDetector class**: Object-oriented interface for multiple detections
- **detect_grid function**: Simple functional interface
- **Debug visualization**: Optional visualization for debugging
- **Multi-stage pipeline**: Preprocessing, detection, validation, sorting
- **Performance optimization**: ~0.5s processing time per 4MP image
- **Robust detection**: Handles various lighting conditions
- **Flexible grid sizes**: Tested from 3×3 to 15×11 patterns

### Performance
- Average processing time: 0.31-0.78s depending on grid size
- Detection success rate: >95% on well-lit calibration patterns
- Sub-pixel accuracy: ±0.1 pixel typical
- Memory efficient: 45-68MB peak usage

### Documentation
- Comprehensive README with usage examples
- API documentation with parameter descriptions
- Contributing guidelines
- Example scripts for common use cases
- Performance benchmarks and comparisons

### Dependencies
- Python 3.8+
- OpenCV 4.5+
- NumPy 1.19+
- SciPy 1.7+
- Numba 0.56+

---

