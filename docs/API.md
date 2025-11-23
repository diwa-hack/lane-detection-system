# API Documentation

## LaneDetectionPipeline

Main pipeline class for running inference with all models.

### Initialization

```python
from utils.inference import LaneDetectionPipeline

pipeline = LaneDetectionPipeline(
    models=models_dict,      # Dictionary of loaded models
    device=device,           # torch.device
    img_size=(256, 320)     # (H, W) image size
)
```

### Methods

#### predict_image()

Detect lanes in a single image.

**Parameters:**
- `image` (np.ndarray): Input image in BGR format
- `confidence_threshold` (float): Minimum confidence for detection (default: 0.5)
- `line_thickness` (int): Thickness of drawn lanes (default: 3)

**Returns:**
Dictionary with best model result and all individual results.

**Example:**
```python
import cv2

image = cv2.imread('road.jpg')
results = pipeline.predict_image(image, confidence_threshold=0.6)

print(f"Best model: {results['best_model']}")
print(f"Confidence: {results['best_confidence']:.2%}")
```

## CLI Usage

```bash
# Single image
python inference.py --image path/to/image.jpg --output results/

# Video
python inference.py --video path/to/video.mp4

# Batch processing
python inference.py --input-dir images/ --output-dir results/

# With custom parameters
python inference.py \
    --image image.jpg \
    --confidence 0.7 \
    --thickness 5 \
    --output results/
```

## Model Classes

### SCNN
```python
from models.scnn import SCNN

model = SCNN(input_channels=3, num_classes=5)
output = model(input_tensor)  # Shape: (B, 5, H, W)
```

### PolyLaneNet
```python
from models.polylane import PolyLaneNet

model = PolyLaneNet(num_lanes=4, degree=3)
poly_coeffs, confidence = model(input_tensor)
```

### UltraFastLaneDetection
```python
from models.ultrafast import UltraFastLaneDetection

model = UltraFastLaneDetection(num_lanes=4, num_row_anchors=56)
locations, existence = model(input_tensor)
```
