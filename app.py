import streamlit as st
import cv2
import torch
import numpy as np
from pathlib import Path
import tempfile
from PIL import Image
import time
from models.scnn import SCNN
from models.polylane import PolyLaneNet
from models.ultrafast import UltraFastLaneDetection
from utils.inference import LaneDetectionPipeline
from utils.video_processor import process_video
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Lane Detection System",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all three lane detection models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = {}
    model_paths = {
        'SCNN': 'checkpoints/scnn_best.pth',
        'PolyLaneNet': 'checkpoints/polylane_best.pth',
        'UltraFast': 'checkpoints/ultrafast_best.pth'
    }

    with st.spinner('Loading models...'):
        for name, path in model_paths.items():
            try:
                if name == 'SCNN':
                    model = SCNN(input_channels=3, num_classes=5)
                elif name == 'PolyLaneNet':
                    model = PolyLaneNet(num_lanes=4, degree=3)
                else:
                    model = UltraFastLaneDetection(num_lanes=4)

                if Path(path).exists():
                    checkpoint = torch.load(path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.to(device)
                    model.eval()
                    models[name] = model
                    st.sidebar.success(f"✓ {name} loaded")
                else:
                    st.sidebar.warning(f"⚠ {name} checkpoint not found")
            except Exception as e:
                st.sidebar.error(f"✗ Error loading {name}: {str(e)}")

    return models, device

def main():
    # Header
    st.markdown('<h1 class="main-header">🛣️ Lane Detection System</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    st.sidebar.title("⚙️ Configuration")

    # Load models
    models, device = load_models()

    if not models:
        st.error("❌ No models loaded. Please ensure model checkpoints are in the 'checkpoints/' directory.")
        st.info("📝 Expected files: scnn_best.pth, polylane_best.pth, ultrafast_best.pth")
        return

    # Create pipeline
    pipeline = LaneDetectionPipeline(models, device)

    # Sidebar options
    st.sidebar.markdown("### Input Options")
    input_type = st.sidebar.radio("Select Input Type", ["Image", "Video"])

    st.sidebar.markdown("### Model Selection")
    show_individual = st.sidebar.checkbox("Show Individual Model Results", value=True)

    st.sidebar.markdown("### Visualization Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    line_thickness = st.sidebar.slider("Lane Line Thickness", 1, 10, 3)

    # Main content
    if input_type == "Image":
        process_image_input(pipeline, show_individual, confidence_threshold, line_thickness)
    else:
        process_video_input(pipeline, show_individual, confidence_threshold, line_thickness)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>🎓 Campus Placement Project | Deep Learning for Autonomous Driving</p>
            <p>Models: SCNN • PolyLaneNet • UltraFast Lane Detection</p>
        </div>
    """, unsafe_allow_html=True)

def process_image_input(pipeline, show_individual, confidence_threshold, line_thickness):
    st.markdown('<h2 class="sub-header">📷 Image Lane Detection</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            st.image(image_rgb, caption="Uploaded Image", use_container_width=True)

            if st.button("🚀 Detect Lanes", type="primary"):
                with st.spinner("Processing..."):
                    start_time = time.time()

                    # Run inference
                    results = pipeline.predict_image(
                        image, 
                        confidence_threshold=confidence_threshold,
                        line_thickness=line_thickness
                    )

                    inference_time = time.time() - start_time

                    # Display results in col2
                    with col2:
                        display_image_results(results, inference_time, show_individual)

def display_image_results(results, inference_time, show_individual):
    st.markdown("### 🎯 Detection Results")

    # Best model result
    st.markdown(f"**Best Model: {results['best_model']}**")
    st.image(results['best_result'], caption=f"Best Result - {results['best_model']}", use_container_width=True)

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Model", results['best_model'])
    with col2:
        st.metric("Confidence", f"{results['best_confidence']:.2%}")
    with col3:
        st.metric("Processing Time", f"{inference_time:.2f}s")

    # Individual results
    if show_individual:
        st.markdown("---")
        st.markdown("### 📊 All Model Comparisons")

        cols = st.columns(len(results['all_results']))
        for idx, (model_name, result_data) in enumerate(results['all_results'].items()):
            with cols[idx]:
                st.markdown(f"**{model_name}**")
                st.image(result_data['image'], use_container_width=True)
                st.metric("Confidence", f"{result_data['confidence']:.2%}")
                st.metric("Lanes Detected", result_data['num_lanes'])

def process_video_input(pipeline, show_individual, confidence_threshold, line_thickness):
    st.markdown('<h2 class="sub-header">🎥 Video Lane Detection</h2>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov'])

    if uploaded_file is not None:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()

        st.video(tfile.name)

        col1, col2, col3 = st.columns(3)
        with col1:
            process_best_only = st.checkbox("Process with Best Model Only", value=False)
        with col2:
            skip_frames = st.number_input("Skip Frames (for faster processing)", 0, 10, 0)
        with col3:
            show_fps = st.checkbox("Show FPS", value=True)

        if st.button("🚀 Process Video", type="primary"):
            with st.spinner("Processing video... This may take a while."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Process video
                output_paths = process_video(
                    tfile.name,
                    pipeline,
                    confidence_threshold=confidence_threshold,
                    line_thickness=line_thickness,
                    best_only=process_best_only,
                    skip_frames=skip_frames,
                    show_fps=show_fps,
                    progress_callback=lambda p, s: update_progress(progress_bar, status_text, p, s)
                )

                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")

                # Display results
                st.markdown("---")
                st.markdown("### 🎬 Processed Videos")

                if process_best_only:
                    st.markdown(f"**Best Model Result**")
                    st.video(output_paths['best'])
                else:
                    tabs = st.tabs(["Best Model"] + list(output_paths['individual'].keys()))

                    with tabs[0]:
                        st.video(output_paths['best'])

                    for idx, (model_name, video_path) in enumerate(output_paths['individual'].items(), 1):
                        with tabs[idx]:
                            st.video(video_path)

def update_progress(progress_bar, status_text, progress, status):
    progress_bar.progress(progress)
    status_text.text(status)

if __name__ == "__main__":
    main()
