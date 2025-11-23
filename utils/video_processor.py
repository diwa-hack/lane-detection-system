import cv2
import os
from pathlib import Path
import tempfile
import time

def process_video(input_path, pipeline, confidence_threshold=0.5, 
                 line_thickness=3, best_only=False, skip_frames=0, 
                 show_fps=True, progress_callback=None):
    """
    Process video with lane detection

    Args:
        input_path: Path to input video
        pipeline: LaneDetectionPipeline instance
        confidence_threshold: Confidence threshold for detection
        line_thickness: Thickness of lane lines
        best_only: If True, only process with best model
        skip_frames: Number of frames to skip (for faster processing)
        show_fps: Whether to display FPS on video
        progress_callback: Callback function for progress updates

    Returns:
        Dictionary with paths to output videos
    """
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output directory
    output_dir = Path(tempfile.gettempdir()) / 'lane_detection_output'
    output_dir.mkdir(exist_ok=True)

    # Prepare video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    best_output_path = str(output_dir / 'best_model_output.mp4')
    best_writer = cv2.VideoWriter(best_output_path, fourcc, fps, (width, height))

    individual_writers = {}
    individual_paths = {}

    if not best_only:
        for model_name in pipeline.models.keys():
            path = str(output_dir / f'{model_name.lower()}_output.mp4')
            writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
            individual_writers[model_name] = writer
            individual_paths[model_name] = path

    frame_count = 0
    processed_count = 0
    start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Skip frames if requested
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                # Write original frame
                best_writer.write(frame)
                for writer in individual_writers.values():
                    writer.write(frame)
                continue

            # Process frame
            results = pipeline.predict_image(
                frame, 
                confidence_threshold=confidence_threshold,
                line_thickness=line_thickness
            )

            # Add FPS text if requested
            if show_fps:
                elapsed = time.time() - start_time
                current_fps = processed_count / elapsed if elapsed > 0 else 0

                # Add FPS to best result
                best_frame = cv2.cvtColor(results['best_result'], cv2.COLOR_RGB2BGR)
                cv2.putText(best_frame, f"FPS: {current_fps:.1f} | Model: {results['best_model']}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                best_writer.write(best_frame)

                # Add FPS to individual results
                if not best_only:
                    for model_name, result_data in results['all_results'].items():
                        frame_with_fps = cv2.cvtColor(result_data['image'], cv2.COLOR_RGB2BGR)
                        cv2.putText(frame_with_fps, 
                                  f"FPS: {current_fps:.1f} | Conf: {result_data['confidence']:.2%}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        individual_writers[model_name].write(frame_with_fps)
            else:
                best_frame = cv2.cvtColor(results['best_result'], cv2.COLOR_RGB2BGR)
                best_writer.write(best_frame)

                if not best_only:
                    for model_name, result_data in results['all_results'].items():
                        frame_rgb = cv2.cvtColor(result_data['image'], cv2.COLOR_RGB2BGR)
                        individual_writers[model_name].write(frame_rgb)

            processed_count += 1

            # Update progress
            if progress_callback:
                progress = int((frame_count / total_frames) * 100)
                status = f"Processing frame {frame_count}/{total_frames}"
                progress_callback(progress, status)

    finally:
        cap.release()
        best_writer.release()
        for writer in individual_writers.values():
            writer.release()

    return {
        'best': best_output_path,
        'individual': individual_paths if not best_only else {}
    }
