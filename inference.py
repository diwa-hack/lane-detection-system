#!/usr/bin/env python3
"""
Standalone inference script for lane detection
"""

import argparse
import cv2
import torch
from pathlib import Path
import sys

from models.scnn import SCNN
from models.polylane import PolyLaneNet
from models.ultrafast import UltraFastLaneDetection
from utils.inference import LaneDetectionPipeline

def load_models(checkpoint_dir, device):
    """Load all models from checkpoints"""
    models = {}

    model_configs = {
        'SCNN': (SCNN, 'scnn_best.pth', {'input_channels': 3, 'num_classes': 5}),
        'PolyLaneNet': (PolyLaneNet, 'polylane_best.pth', {'num_lanes': 4, 'degree': 3}),
        'UltraFast': (UltraFastLaneDetection, 'ultrafast_best.pth', {'num_lanes': 4})
    }

    for name, (model_class, ckpt_file, kwargs) in model_configs.items():
        ckpt_path = checkpoint_dir / ckpt_file

        if not ckpt_path.exists():
            print(f"Warning: {ckpt_file} not found, skipping {name}")
            continue

        try:
            model = model_class(**kwargs)
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            models[name] = model
            print(f"✓ Loaded {name}")
        except Exception as e:
            print(f"✗ Error loading {name}: {e}")

    return models

def process_image(pipeline, image_path, output_dir, confidence, thickness):
    """Process a single image"""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return

    results = pipeline.predict_image(
        image,
        confidence_threshold=confidence,
        line_thickness=thickness
    )

    # Save result
    output_path = output_dir / f"{image_path.stem}_result.jpg"
    result_bgr = cv2.cvtColor(results['best_result'], cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), result_bgr)

    print(f"✓ Processed {image_path.name}")
    print(f"  Best model: {results['best_model']}")
    print(f"  Confidence: {results['best_confidence']:.2%}")
    print(f"  Output: {output_path}")

def process_directory(pipeline, input_dir, output_dir, confidence, thickness):
    """Process all images in a directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in input_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]

    print(f"\nFound {len(image_files)} images")

    for image_path in image_files:
        process_image(pipeline, image_path, output_dir, confidence, thickness)

def main():
    parser = argparse.ArgumentParser(description='Lane Detection Inference')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--input-dir', type=str, help='Path to input directory')
    parser.add_argument('--output-dir', type=str, default='results', 
                       help='Output directory')
    parser.add_argument('--checkpoints', type=str, default='checkpoints',
                       help='Checkpoints directory')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--thickness', type=int, default=3,
                       help='Line thickness')
    parser.add_argument('--model', type=str, choices=['scnn', 'polylane', 'ultrafast', 'best'],
                       default='best', help='Model to use')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    checkpoint_dir = Path(args.checkpoints)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load models
    print("\nLoading models...")
    models = load_models(checkpoint_dir, device)

    if not models:
        print("Error: No models loaded!")
        sys.exit(1)

    pipeline = LaneDetectionPipeline(models, device)

    # Process input
    if args.image:
        image_path = Path(args.image)
        process_image(pipeline, image_path, output_dir, args.confidence, args.thickness)

    elif args.video:
        from utils.video_processor import process_video
        print(f"\nProcessing video: {args.video}")
        output_paths = process_video(
            args.video,
            pipeline,
            confidence_threshold=args.confidence,
            line_thickness=args.thickness,
            show_fps=True
        )
        print(f"✓ Video processed: {output_paths['best']}")

    elif args.input_dir:
        input_dir = Path(args.input_dir)
        process_directory(pipeline, input_dir, output_dir, args.confidence, args.thickness)

    else:
        print("Error: Please specify --image, --video, or --input-dir")
        sys.exit(1)

    print("\n✓ Done!")

if __name__ == '__main__':
    main()
