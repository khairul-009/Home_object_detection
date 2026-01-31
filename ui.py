"""
YOLO Home Items Object Detection - Streamlit Web Application
==============================================================
A modern, user-friendly web interface for detecting home items in images
using YOLOv8 object detection model.

Features:
- Upload images (JPG, JPEG, PNG)
- Real-time object detection with YOLOv8
- Visual display with bounding boxes and labels
- Download annotated results
- Adjustable confidence threshold
- Detection statistics

Author: Your Name
Date: 2026-01-31
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import io
import tempfile
from pathlib import Path

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="YOLO Home Items Detector",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR MODERN UI
# ============================================================================

st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    /* Card styling for sections */
    .card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Metric styling */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        border-radius: 8px;
        padding: 0.75rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #764ba2;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING FUNCTION
# ============================================================================

@st.cache_resource
def load_yolo_model(model_path="yolov8n.pt"):
    """
    Load and cache the YOLO model to avoid reloading on every interaction.
    
    Args:
        model_path (str): Path to the YOLO model weights file.
                         Default is 'yolov8n.pt' (YOLOv8 Nano).
                         Other options: 'yolov8s.pt', 'yolov8m.pt', etc.
    
    Returns:
        YOLO: Loaded YOLO model instance
    
    Note:
        @st.cache_resource decorator ensures the model is loaded only once
        and reused across all user sessions, improving performance.
    """
    try:
        # Load the YOLO model
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("💡 The model will be downloaded automatically on first run.")
        return None

# ============================================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================================

def process_image(model, image, confidence_threshold=0.25):
    """
    Run YOLO object detection on the uploaded image.
    
    Args:
        model (YOLO): Loaded YOLO model
        image (PIL.Image): Input image from user
        confidence_threshold (float): Minimum confidence for detections (0.0-1.0)
    
    Returns:
        tuple: (annotated_image_array, results_object)
               - annotated_image_array: NumPy array with bounding boxes drawn
               - results_object: YOLO results containing detection data
    """
    # Ensure image is in RGB format (3 channels)
    # This handles RGBA, grayscale, and other formats
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert PIL Image to numpy array for processing
    image_array = np.array(image)
    
    # Verify we have 3 channels (RGB)
    if len(image_array.shape) == 2:
        # Grayscale image - convert to RGB
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif image_array.shape[2] == 4:
        # RGBA image - remove alpha channel
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    
    # Run YOLO inference on the image
    # conf parameter sets the confidence threshold
    # verbose=False suppresses console output
    results = model(image_array, conf=confidence_threshold, verbose=False)
    
    # Get the annotated image with bounding boxes and labels
    # plot() method draws boxes, labels, and confidence scores
    annotated_image = results[0].plot()
    
    # Convert from BGR (OpenCV format) to RGB (display format)
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    return annotated_image_rgb, results[0]

def extract_detection_details(results):
    """
    Extract detailed information about detected objects.
    
    Args:
        results: YOLO results object from inference
    
    Returns:
        list: List of dictionaries containing detection information
              Each dict has: class_name, confidence, bbox_coordinates
    """
    detections = []
    
    # Check if any objects were detected
    if results.boxes is not None and len(results.boxes) > 0:
        # Iterate through each detected object
        for box in results.boxes:
            # Extract class ID and get class name from model
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            
            # Extract confidence score
            confidence = float(box.conf[0])
            
            # Extract bounding box coordinates [x1, y1, x2, y2]
            bbox = box.xyxy[0].tolist()
            
            # Store detection information
            detection_info = {
                'class': class_name,
                'confidence': confidence,
                'bbox': bbox
            }
            detections.append(detection_info)
    
    return detections

def convert_image_to_bytes(image_array, format='PNG'):
    """
    Convert numpy array image to bytes for download.
    
    Args:
        image_array (numpy.ndarray): Image as numpy array
        format (str): Image format (PNG, JPEG, etc.)
    
    Returns:
        bytes: Image data as bytes
    """
    # Convert numpy array to PIL Image
    image_pil = Image.fromarray(image_array)
    
    # Create a bytes buffer
    buffer = io.BytesIO()
    
    # Save image to buffer in specified format
    image_pil.save(buffer, format=format)
    
    # Get the bytes value
    return buffer.getvalue()

# ============================================================================
# STATISTICS CALCULATION
# ============================================================================

def calculate_statistics(detections):
    """
    Calculate detection statistics for display.
    
    Args:
        detections (list): List of detection dictionaries
    
    Returns:
        dict: Statistics including counts, average confidence, etc.
    """
    if not detections:
        return {
            'total_objects': 0,
            'unique_classes': 0,
            'avg_confidence': 0,
            'class_counts': {}
        }
    
    # Count objects by class
    class_counts = {}
    total_confidence = 0
    
    for det in detections:
        class_name = det['class']
        confidence = det['confidence']
        
        # Update class count
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1
        
        # Sum confidence for average calculation
        total_confidence += confidence
    
    # Calculate statistics
    stats = {
        'total_objects': len(detections),
        'unique_classes': len(class_counts),
        'avg_confidence': total_confidence / len(detections),
        'class_counts': class_counts
    }
    
    return stats

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main application function containing the Streamlit UI and logic.
    """
    
    # ========================================================================
    # HEADER SECTION
    # ========================================================================
    
    st.markdown("""
        <div class="header-container">
            <h1>🏠 YOLO Home Items Detector</h1>
            <p style="font-size: 1.2rem; margin-top: 0.5rem;">
                Upload an image to detect and identify home items using advanced AI
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # SIDEBAR CONFIGURATION
    # ========================================================================
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("Model Settings")
        
        # Add option for custom model
        model_type = st.radio(
            "Model Type",
            options=["Pre-trained Models", "Custom Model"],
            help="Choose between pre-trained YOLOv8 models or upload your own"
        )
        
        if model_type == "Pre-trained Models":
            model_choice = st.selectbox(
                "Select YOLO Model",
                options=[
                    "YOLOv8 Nano (Fastest, ~6MB)",
                    "YOLOv8 Small (Balanced, ~22MB)",
                    "YOLOv8 Medium (Accurate, ~52MB)",
                    "YOLOv8 Large (Very Accurate, ~88MB)"
                ],
                index=0,
                help="Larger models are more accurate but slower"
            )
            
            # Map selection to model file
            model_map = {
                "YOLOv8 Nano (Fastest, ~6MB)": "yolov8n.pt",
                "YOLOv8 Small (Balanced, ~22MB)": "yolov8s.pt",
                "YOLOv8 Medium (Accurate, ~52MB)": "yolov8m.pt",
                "YOLOv8 Large (Very Accurate, ~88MB)": "yolov8l.pt"
            }
            model_path = model_map[model_choice]
            custom_model_file = None
            
        else:  # Custom Model
            st.info("📤 Upload your custom YOLOv8 model (.pt file)")
            custom_model_file = st.file_uploader(
                "Choose a model file",
                type=['pt'],
                help="Upload a custom trained YOLO model (.pt format)"
            )
            
            if custom_model_file is not None:
                # Save uploaded model to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                    tmp_file.write(custom_model_file.read())
                    model_path = tmp_file.name
                
                st.success(f"✅ Custom model uploaded: {custom_model_file.name}")
                
                # Display model info
                with st.expander("📋 Model Information"):
                    st.write(f"**Filename:** {custom_model_file.name}")
                    st.write(f"**Size:** {custom_model_file.size / (1024*1024):.2f} MB")
            else:
                st.warning("⚠️ Please upload a custom model file to proceed")
                model_path = None
        
        # Confidence threshold slider
        st.subheader("Detection Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum confidence score for displaying detections. Lower values show more objects but may include false positives."
        )
        
        # Display options
        st.subheader("Display Options")
        show_statistics = st.checkbox("Show Statistics", value=True)
        show_detection_list = st.checkbox("Show Detection List", value=True)
        
        # Information section
        st.markdown("---")
        st.subheader("ℹ️ Information")
        
        if model_type == "Pre-trained Models":
            st.info("""
                **Supported Objects:**
                - Furniture (chair, sofa, bed, table)
                - Electronics (TV, laptop, phone)
                - Kitchen items (refrigerator, microwave)
                - And 70+ more COCO dataset classes
                
                **Tips:**
                - Use well-lit, clear images
                - Larger models are more accurate
                - Adjust confidence for sensitivity
            """)
        else:
            st.info("""
                **Custom Model Guidelines:**
                - Upload a trained YOLOv8 model (.pt file)
                - Model should be compatible with Ultralytics format
                - Detection classes depend on your training
                
                **Tips:**
                - Ensure model is properly trained
                - Use appropriate confidence threshold
                - Test with sample images first
            """)
        
        # About section
        with st.expander("📖 About"):
            st.markdown("""
                This application uses **YOLOv8** (You Only Look Once) 
                from Ultralytics for real-time object detection.
                
                **Model Information:**
                - Architecture: YOLOv8
                - Framework: Ultralytics
                - Supports: Pre-trained & Custom models
                
                **Version:** 2.0.0
            """)
    
    # ========================================================================
    # MAIN CONTENT AREA
    # ========================================================================
    
    # Load the YOLO model
    if model_path is not None:
        with st.spinner(f"Loading model... Please wait..."):
            model = load_yolo_model(model_path)
        
        if model is not None:
            if model_type == "Custom Model":
                st.success(f"✅ Custom model loaded successfully!")
                
                # Display custom model details
                try:
                    model_info_col1, model_info_col2 = st.columns(2)
                    with model_info_col1:
                        st.info(f"**Model Type:** Custom YOLO")
                    with model_info_col2:
                        # Get number of classes if available
                        if hasattr(model, 'names'):
                            st.info(f"**Classes:** {len(model.names)}")
                except:
                    pass
            else:
                st.success(f"✅ Model loaded successfully: {model_choice}")
        else:
            st.error("Failed to load the model. Please check your file and try again.")
            st.stop()
    else:
        st.warning("⚠️ Please select or upload a model to continue.")
        st.info("👈 Use the sidebar to choose a pre-trained model or upload your custom model.")
        st.stop()
    
    # ========================================================================
    # IMAGE UPLOAD SECTION
    # ========================================================================
    
    st.markdown("### 📤 Upload Image")
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # File uploader widget
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a JPG, JPEG, or PNG image"
        )
    
    with col2:
        # Instructions
        st.info("""
            **How to use:**
            1. Upload an image using the file uploader
            2. Wait for detection to complete
            3. View results with bounding boxes
            4. Download the annotated image
        """)
    
    # ========================================================================
    # IMAGE PROCESSING AND RESULTS
    # ========================================================================
    
    if uploaded_file is not None:
        try:
            # Read the uploaded image
            image = Image.open(uploaded_file)
            
            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if image.mode not in ['RGB', 'L']:
                st.info(f"ℹ️ Converting image from {image.mode} to RGB format...")
                image = image.convert('RGB')
            elif image.mode == 'L':
                st.info("ℹ️ Converting grayscale image to RGB format...")
                image = image.convert('RGB')
            
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")
            st.stop()
        
        # Display original image
        st.markdown("---")
        st.markdown("### 🖼️ Results")
        
        # Create columns for before/after comparison
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.markdown("#### 📷 Original Image")
            st.image(image, use_container_width=True, caption="Uploaded Image")
            
            # Display image info
            st.caption(f"📐 Size: {image.size[0]} x {image.size[1]} pixels")
            st.caption(f"🎨 Mode: {image.mode}")
        
        with result_col2:
            st.markdown("#### 🎯 Detection Results")
            
            # Process image with loading indicator
            try:
                with st.spinner("🔍 Detecting objects... Please wait..."):
                    annotated_image, results = process_image(
                        model, 
                        image, 
                        confidence_threshold
                    )
                
                # Display annotated image
                st.image(
                    annotated_image, 
                    use_container_width=True, 
                    caption="Objects Detected"
                )
                
            except Exception as e:
                st.error(f"❌ Error during detection: {str(e)}")
                st.error("**Possible solutions:**")
                st.markdown("""
                - Ensure your custom model is compatible with the image
                - Try converting the image to JPG format
                - Check if your model was trained properly
                - Try with a pre-trained model first to verify the image is valid
                """)
                st.stop()
        
        # Extract detection details
        detections = extract_detection_details(results)
        
        # ====================================================================
        # STATISTICS SECTION
        # ====================================================================
        
        if show_statistics and detections:
            st.markdown("---")
            st.markdown("### 📊 Detection Statistics")
            
            # Calculate statistics
            stats = calculate_statistics(detections)
            
            # Display metrics in columns
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric(
                    label="Total Objects",
                    value=stats['total_objects'],
                    delta=None
                )
            
            with metric_col2:
                st.metric(
                    label="Unique Classes",
                    value=stats['unique_classes'],
                    delta=None
                )
            
            with metric_col3:
                st.metric(
                    label="Avg. Confidence",
                    value=f"{stats['avg_confidence']:.1%}",
                    delta=None
                )
            
            # Class distribution chart
            if stats['class_counts']:
                st.markdown("#### 📈 Objects by Class")
                
                # Create a simple bar chart using Streamlit
                import pandas as pd
                chart_data = pd.DataFrame({
                    'Class': list(stats['class_counts'].keys()),
                    'Count': list(stats['class_counts'].values())
                }).sort_values('Count', ascending=False)
                
                st.bar_chart(chart_data.set_index('Class'))
        
        # ====================================================================
        # DETECTION LIST SECTION
        # ====================================================================
        
        if show_detection_list and detections:
            st.markdown("---")
            st.markdown("### 📋 Detailed Detection List")
            
            # Show detected class names for custom models
            if model_type == "Custom Model":
                with st.expander("🏷️ Available Classes in Custom Model"):
                    if hasattr(model, 'names'):
                        class_list = list(model.names.values())
                        st.write(f"**Total Classes:** {len(class_list)}")
                        st.write("**Class Names:**")
                        # Display in columns for better readability
                        cols = st.columns(3)
                        for idx, class_name in enumerate(class_list):
                            cols[idx % 3].write(f"- {class_name}")
            
            # Create a formatted table of detections
            for i, det in enumerate(detections, 1):
                with st.expander(
                    f"🔍 Detection #{i}: {det['class'].upper()} - {det['confidence']:.1%} confidence"
                ):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.write(f"**Class:** {det['class']}")
                        st.write(f"**Confidence:** {det['confidence']:.2%}")
                    
                    with col_b:
                        bbox = det['bbox']
                        st.write(f"**Bounding Box:**")
                        st.write(f"Top-left: ({bbox[0]:.0f}, {bbox[1]:.0f})")
                        st.write(f"Bottom-right: ({bbox[2]:.0f}, {bbox[3]:.0f})")
        
        # ====================================================================
        # DOWNLOAD SECTION
        # ====================================================================
        
        st.markdown("---")
        st.markdown("### 💾 Download Results")
        
        # Convert annotated image to bytes
        image_bytes = convert_image_to_bytes(annotated_image, format='PNG')
        
        # Create download columns
        download_col1, download_col2, download_col3 = st.columns([2, 2, 2])
        
        with download_col1:
            # Download button for PNG
            st.download_button(
                label="📥 Download as PNG",
                data=image_bytes,
                file_name=f"detected_{uploaded_file.name.split('.')[0]}.png",
                mime="image/png",
                use_container_width=True
            )
        
        with download_col2:
            # Download button for JPEG
            image_bytes_jpg = convert_image_to_bytes(annotated_image, format='JPEG')
            st.download_button(
                label="📥 Download as JPG",
                data=image_bytes_jpg,
                file_name=f"detected_{uploaded_file.name.split('.')[0]}.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
        
        with download_col3:
            # Option to clear and start over
            if st.button("🔄 Process Another Image", use_container_width=True):
                st.rerun()
        
        # Display success message
        if detections:
            st.success(f"✅ Successfully detected {len(detections)} object(s)!")
        else:
            st.warning("⚠️ No objects detected. Try lowering the confidence threshold.")
    
    else:
        # ====================================================================
        # WELCOME SCREEN (No image uploaded)
        # ====================================================================
        
        st.markdown("---")
        st.markdown("### 👋 Welcome!")
        
        # Create columns for welcome content
        welcome_col1, welcome_col2 = st.columns(2)
        
        with welcome_col1:
            st.markdown("""
                #### 🚀 Get Started
                
                To begin detecting objects in your images:
                
                1. **Upload an Image** - Click the file uploader above
                2. **Adjust Settings** - Use the sidebar to configure detection
                3. **View Results** - See detected objects with bounding boxes
                4. **Download** - Save your annotated image
                
                **Supported Formats:** JPG, JPEG, PNG
            """)
        
        with welcome_col2:
            st.markdown("""
                #### 💡 Tips for Best Results
                
                - Use **clear, well-lit** images
                - Ensure objects are **visible and unobstructed**
                - Start with **YOLOv8 Nano** for speed
                - Adjust **confidence threshold** if needed:
                  - Lower (0.15-0.20): More detections
                  - Higher (0.50-0.70): Only confident detections
                - Try **larger models** for better accuracy
            """)
        
        # Sample use cases
        st.markdown("---")
        st.markdown("#### 🎯 What Can You Detect?")
        
        use_case_col1, use_case_col2, use_case_col3 = st.columns(3)
        
        with use_case_col1:
            st.markdown("""
                **🪑 Furniture**
                - Chairs
                - Sofas
                - Tables
                - Beds
            """)
        
        with use_case_col2:
            st.markdown("""
                **📱 Electronics**
                - TVs
                - Laptops
                - Phones
                - Keyboards
            """)
        
        with use_case_col3:
            st.markdown("""
                **🍽️ Kitchen Items**
                - Refrigerators
                - Microwaves
                - Ovens
                - Utensils
            """)
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray; padding: 2rem 0;'>
            <p style='margin: 0;'>
                Built with ❤️ using <strong>Streamlit</strong> and <strong>YOLOv8</strong>
            </p>
            <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
                Powered by <strong>Ultralytics YOLO</strong> | 
                <a href='https://docs.ultralytics.com/' target='_blank' style='color: #667eea;'>Documentation</a>
            </p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()