import torch
import cv2
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

class LaneDetectionPipeline:
    """
    Unified pipeline for lane detection - Updated for your model architectures
    All models output segmentation masks of shape (B, 1, H, W)
    """
    def __init__(self, models, device, img_size=(256, 320)):
        self.models = models
        self.device = device
        self.img_size = img_size  # (H, W)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

        self.colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
        ]

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)

    def predict_image(self, image, confidence_threshold=0.5, line_thickness=3):
        """
        Run inference on a single image with all models
        Returns best model result and all individual results
        """
        original_size = (image.shape[1], image.shape[0])  # (W, H)
        input_tensor = self.preprocess_image(image)

        results = {}
        all_results = {}

        with torch.no_grad():
            for model_name, model in self.models.items():
                # All models output segmentation masks (B, 1, H, W)
                output = model(input_tensor)

                # Process the segmentation output
                lanes, confidence = self._process_segmentation_output(
                    output, original_size, confidence_threshold
                )

                # Draw lanes on image
                result_image = self._draw_lanes(
                    image.copy(), lanes, line_thickness
                )

                all_results[model_name] = {
                    'image': cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB),
                    'lanes': lanes,
                    'confidence': confidence,
                    'num_lanes': len(lanes)
                }

        # Select best model based on confidence and number of lanes detected
        best_model = max(all_results.items(), 
                        key=lambda x: (x[1]['confidence'], x[1]['num_lanes']))

        results = {
            'best_model': best_model[0],
            'best_result': best_model[1]['image'],
            'best_confidence': best_model[1]['confidence'],
            'all_results': all_results
        }

        return results

    def _process_segmentation_output(self, output, original_size, threshold):
        """
        Process segmentation mask output to extract lane lines
        Works for all three models (SCNN, PolyLaneNet, UltraFast)
        """
        # Apply sigmoid to get probabilities
        output = torch.sigmoid(output)
        output = output.squeeze(0).squeeze(0).cpu().numpy()  # (H, W)

        # Threshold the mask
        binary_mask = (output > threshold).astype(np.uint8)

        H, W = self.img_size
        orig_W, orig_H = original_size

        # Find connected components (lanes)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        lanes = []
        confidences = []

        # Process each connected component (skip background label 0)
        for label_idx in range(1, num_labels):
            # Get mask for this lane
            lane_mask = (labels == label_idx).astype(np.uint8)

            # Skip if too small
            if stats[label_idx, cv2.CC_STAT_AREA] < 50:
                continue

            # Extract lane points
            lane_points = []

            # Sample along height
            for y in range(0, H, 2):  # Sample every 2 pixels
                # Get all x positions for this y
                x_positions = np.where(lane_mask[y, :] > 0)[0]

                if len(x_positions) > 0:
                    # Use median x position
                    x = int(np.median(x_positions))

                    # Scale to original size
                    x_orig = int(x * orig_W / W)
                    y_orig = int(y * orig_H / H)
                    lane_points.append((x_orig, y_orig))

            if len(lane_points) > 10:  # Need minimum points for a valid lane
                lanes.append(lane_points)

                # Calculate confidence based on mask intensity
                lane_region_confidence = output[lane_mask > 0].mean()
                confidences.append(float(lane_region_confidence))

        avg_confidence = np.mean(confidences) if confidences else 0.0
        return lanes, avg_confidence

    def _draw_lanes(self, image, lanes, thickness=3):
        """Draw detected lanes on image"""
        for idx, lane_points in enumerate(lanes):
            color = self.colors[idx % len(self.colors)]

            # Draw lines connecting points
            for i in range(len(lane_points) - 1):
                cv2.line(image, lane_points[i], lane_points[i+1], color, thickness)

            # Optionally draw points
            for point in lane_points[::5]:  # Draw every 5th point
                cv2.circle(image, point, 2, color, -1)

        return image
