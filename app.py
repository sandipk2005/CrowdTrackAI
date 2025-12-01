import streamlit as st
import cv2
import numpy as np
from PIL import Image
# We import the updated detection logic
from detection import detect_people, load_yolo_model
from utils import generate_heatmap, save_video
from config import (
    YOLO_MODELS,
    ENABLE_FPS_DISPLAY,
    ENABLE_HEATMAP,
    ENABLE_OVERCROWD_ALERT,
    ALERT_SOUND
)
import tempfile
import time
import os
import csv
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ✅ Optimize OpenCV performance
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

# 🌐 Streamlit App Config
st.set_page_config(page_title="CrowdTrackAI – Smart Detection", page_icon="🤖", layout="wide")

# 💠 --- Custom Cyber Style ---
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 30% 30%, #00111a, #000a0f, #000);
    color: #ffffff;
    font-family: 'Orbitron', sans-serif;
}
h1, h2, h3 {
    text-shadow: 0 0 20px #00ffff;
    text-align:center;
}
div.stButton > button:first-child {
    background: linear-gradient(90deg,#00ffff,#0080ff);
    color:white;
    border:none;
    border-radius:12px;
    padding:10px 24px;
    font-weight:bold;
    box-shadow:0 0 15px #00ffff;
    transition:0.3s;
}
div.stButton > button:hover {
    transform:scale(1.1);
    box-shadow:0 0 35px #00ffff;
}
@keyframes blink {50%{opacity:0;}}
.ai-active {
    color:#00ff00;
    font-weight:bold;
    animation:blink 1s infinite;
}
.scan-line {
    height:4px;
    width:60%;
    margin:auto;
    background:linear-gradient(90deg,#00ffff,#0080ff,#00ffff);
    animation:move 2s infinite;
}
@keyframes move {0%{background-position:0%;}100%{background-position:100%;}}
.confidence-gauge {
    background: linear-gradient(90deg, #ff0000, #ffff00, #00ff00);
    border-radius:10px;
    padding:3px;
    margin:10px 0;
}
.confidence-value {
    text-align:center;
    font-weight:bold;
    font-size:16px;
    margin:5px 0;
}
</style>
""", unsafe_allow_html=True)

# 🔷 Header
st.markdown("""
<h1>🤖 CrowdTrackAI</h1>
<h3>AI-Powered Real-Time People Detection System</h3>
<div class="scan-line"></div>
""", unsafe_allow_html=True)

# ------------------------------------------------ SIDEBAR SETTINGS ------------------------------------------------
st.sidebar.header("⚙️ System Settings")

st.sidebar.markdown("---")

# 2. NEW FEATURE: Range Adjustment
st.sidebar.header("👥 Detection Range & Alerts")
st.sidebar.info("Set the acceptable range of people. Alerts will trigger if counts are outside this range.")

# Range Slider
min_detected_range, max_detected_range = st.sidebar.slider(
    "Set Person Count Range:",
    min_value=0,
    max_value=1000,
    value=(10, 50),
    step=1
)

st.sidebar.write(f"**Alert Trigger:** > {max_detected_range} or < {min_detected_range}")

# Mode selector
option = st.radio("Select Input Source:", ["🖼️ Image", "🎥 Video", "📷 Live Camera", "📊 Data Insights"], horizontal=True)

# AI Status
st.markdown('<div style="text-align:center;"><span>🧠 AI Status: </span><span class="ai-active">ACTIVE 🔵</span></div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

count_placeholder = st.empty()
alert_placeholder = st.empty()
confidence_placeholder = st.empty()
chart_placeholder = st.empty()

# Logging setup
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "detections_log.csv")
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        csv.writer(f).writerow(["Timestamp", "People_Count", "Confidence(%)", "Mode"])

def log_detection(people_count, avg_conf, mode):
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), people_count, avg_conf, mode])

# --- CONFIDENCE VISUALIZATION FUNCTION ---
def display_confidence_gauge(confidence):
    """Display a beautiful confidence gauge with color coding"""
    confidence = max(0, min(100, confidence))  # Ensure between 0-100
    
    # Color coding
    if confidence >= 80:
        color = "#00ff00"  # Green
        status = "Excellent"
    elif confidence >= 60:
        color = "#ffff00"  # Yellow
        status = "Good"
    elif confidence >= 40:
        color = "#ffa500"  # Orange
        status = "Fair"
    else:
        color = "#ff0000"  # Red
        status = "Low"
    
    # Create gauge using Plotly
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Model Confidence: {status}", 'font': {'color': 'white', 'size': 16}},
        delta = {'reference': 50, 'increasing': {'color': color}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white", 'tickfont': {'color': 'white'}},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(255,0,0,0.3)'},
                {'range': [40, 60], 'color': 'rgba(255,165,0,0.3)'},
                {'range': [60, 80], 'color': 'rgba(255,255,0,0.3)'},
                {'range': [80, 100], 'color': 'rgba(0,255,0,0.3)'}],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': confidence}}))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Arial"},
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

# --- FIXED DUPLICATE REMOVAL FUNCTION ---
def remove_duplicate_detections(detections, iou_threshold=0.45):
    """
    Remove duplicate detections using NMS.
    Handles both nested lists (Image Mode) and flat lists (Video/Live Mode).
    """
    if detections is None or len(detections) == 0:
        return []
    
    boxes = []
    confidences = []
    
    for det in detections:
        # CHECK 1: Image Mode (List/Tuple)
        if isinstance(det[0], (list, tuple, np.ndarray)):
            bbox = det[0]
            conf = det[1]
            if len(bbox) == 4:
                x, y, w, h = bbox
                if (w * h) >= 500: # Lowered minimum area to detect small people
                    boxes.append([x, y, x + w, y + h])
                    confidences.append(float(conf))
                    
        # CHECK 2: Video/Live Mode (Numbers)
        elif isinstance(det[0], (int, float, np.integer, np.floating)):
            if len(det) >= 4:
                x1, y1, x2, y2 = det[:4]
                conf = det[4] if len(det) > 4 else 0.5 
                area = (x2 - x1) * (y2 - y1)
                if area >= 500: # Lowered minimum area
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(float(conf))
    
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, iou_threshold)
        
        final_detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                if i < len(detections):
                    final_detections.append(detections[i])
        return final_detections
    
    return []

# --- HELPER FUNCTION FOR ALERTS ---
def check_range_and_alert(count, min_r, max_r):
    if count > max_r:
        alert_placeholder.error(f"🚨 ALERT: High Density! {count} People detected (Limit: {max_r})")
        if ENABLE_OVERCROWD_ALERT and os.path.exists(ALERT_SOUND):
            st.audio(ALERT_SOUND, format="audio/mp3", start_time=0)
    elif count < min_r:
        alert_placeholder.warning(f"⚠️ Notice: Low Count. {count} People detected (Target Min: {min_r})")
    else:
        alert_placeholder.success(f"✅ Status Normal: {count} People detected (Range: {min_r}-{max_r})")

# --- SIMPLE TRACKING FALLBACK ---
class SimpleTracker:
    def __init__(self):
        self.next_id = 1
        self.tracks = {}
        
    def update(self, detections):
        current_tracks = {}
        results = []
        
        for det in detections:
            if isinstance(det[0], (list, tuple, np.ndarray)):
                bbox = det[0]
                conf = det[1]
                x, y, w, h = bbox
                center_x, center_y = x + w/2, y + h/2
                x1, y1, x2, y2 = x, y, x+w, y+h
            else:
                x1, y1, x2, y2 = det[:4]
                conf = det[4] if len(det) > 4 else 0.5
                center_x, center_y = (x1 + x2)/2, (y1 + y2)/2
                w, h = x2 - x1, y2 - y1
                
            track_id = self._find_closest_track(center_x, center_y)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
                
            current_tracks[track_id] = (center_x, center_y)
            results.append([x1, y1, x2, y2, track_id, conf])
            
        self.tracks = current_tracks
        return results
    
    def _find_closest_track(self, x, y, max_distance=100):
        for track_id, (tx, ty) in self.tracks.items():
            distance = np.sqrt((x - tx)**2 + (y - ty)**2)
            if distance < max_distance:
                return track_id
        return None

simple_tracker = SimpleTracker()

# ------------------------------------------------ IMAGE MODE ------------------------------------------------
if option == "🖼️ Image":
    load_yolo_model('yolov8x.pt') 
    st.success("✅ Loaded: YOLOv8x (Extra Large) - High Accuracy")
    st.caption("ℹ️ **Ultra Detection Mode**: Detecting all people.")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        frame = np.array(image)
        
        # UPDATE: Optimized settings for maximum detection
        detection_config = {
            'conf_threshold': 0.15,  # Lower confidence to catch more people
            'iou_threshold': 0.45,   # Better separation
            'max_det': 1000,
        }
        
        start_time = time.time()

        # UPDATE: Use same detection mode as video for consistency
        detections, people_count, avg_conf = detect_people(
            frame, 
            is_video=True,  # Changed to True for better detection
            conf_threshold=detection_config['conf_threshold'],
            iou_threshold=detection_config['iou_threshold'],
            max_detections=detection_config['max_det']
        )
        
        # UPDATE: Use same duplicate removal as video mode
        clean_detections = remove_duplicate_detections(detections, iou_threshold=0.40)
        accurate_count = len(clean_detections)
        
        fps = 1.0 / (time.time() - start_time)
        log_detection(accurate_count, avg_conf, "Image")

        if people_count != accurate_count:
            st.info(f"🔍 Optimized: {accurate_count} unique people found.")

        count_placeholder.markdown(
            f"<h2 style='text-align:center;color:#00ffaa;'>✅ People Detected: {accurate_count}</h2>", 
            unsafe_allow_html=True
        )
        
        check_range_and_alert(accurate_count, min_detected_range, max_detected_range)

        # Display confidence gauge
        confidence_fig = display_confidence_gauge(avg_conf)
        confidence_placeholder.plotly_chart(confidence_fig, use_container_width=True)

        for det in clean_detections:
            if isinstance(det[0], (list, tuple, np.ndarray)):
                bbox = det[0]
                conf = det[1]
                
                if len(bbox) == 4:
                    x, y, w, h = map(int, bbox)
                    
                    if conf > 0.7: color = (0, 255, 0)
                    elif conf > 0.4: color = (0, 255, 255)
                    else: color = (0, 165, 255)  # Changed to orange for better visibility
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    label = f"Person {conf:.2f}"
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x, y - 20), (x + text_w, y), color, -1)
                    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        if ENABLE_HEATMAP:
            heatmap = generate_heatmap(frame, clean_detections)
            frame = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

        if ENABLE_FPS_DISPLAY:
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        st.image(frame, channels="BGR", caption=f"🎯 ACCURATE SCAN: {accurate_count} people detected", width='stretch')
        
        if accurate_count > 0:
            st.success(f"🎉 ACCURATE COUNT! Detected {accurate_count} unique people!")
            
# ------------------------------------------------ VIDEO MODE ------------------------------------------------
elif option == "🎥 Video":
    # UPDATE: Using VisDrone Model for perfect aerial/crowd detection
    load_yolo_model('yolov8m-visdrone.pt')
    st.success("✅ Loaded: YOLOv8-VisDrone (Best for Crowds & Aerial Views)")
    st.caption("ℹ️ **High Precision Mode**: Detecting small and crowded objects.")

    video_file = st.file_uploader("📹 Upload Video", type=["mp4", "avi", "mov", "mkv"])
    if video_file:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(video_file.read())
        temp_video.close()
        cap = cv2.VideoCapture(temp_video.name)
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        st.info(f"📊 Video Info: {frame_width}x{frame_height} @ {original_fps:.1f}FPS")
        
        stframe = st.empty()
        frames = []

        # UPDATE: Perfect Count Settings
        detection_config = {
            'conf_threshold': 0.25, # Low confidence to catch everyone
            'iou_threshold': 0.45,  # Separates people standing close
            'max_det': 1000,        # Allows counting large crowds
        }

        with st.spinner("🔍 ACCURATE SCAN: Detecting all people..."):
            frame_count = 0
            total_people_detected = 0
            max_people_in_frame = 0
            confidence_values = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                status_text.text(f"🔍 Scanning frame {frame_count}...")
                progress_bar.progress(min(frame_count / 500, 1.0))

                start_time = time.time()
                
                detections, people_count, avg_conf = detect_people(
                    frame, 
                    is_video=True,
                    conf_threshold=detection_config['conf_threshold'],
                    iou_threshold=detection_config['iou_threshold'],
                    max_detections=detection_config['max_det']
                )
                
                confidence_values.append(avg_conf)
                
                # Robust duplicate removal
                clean_detections = remove_duplicate_detections(detections, iou_threshold=0.45)
                accurate_count = len(clean_detections)
                
                tracked_objects = simple_tracker.update(clean_detections)
                
                max_people_in_frame = max(max_people_in_frame, accurate_count)
                fps = 1.0 / (time.time() - start_time)
                log_detection(accurate_count, avg_conf, "Video")

                for obj in tracked_objects:
                    if len(obj) >= 5:
                        x1, y1, x2, y2, obj_id = map(int, obj[:5])
                        
                        color_hash = (obj_id * 50) % 255
                        color = (color_hash, (color_hash + 85) % 255, (color_hash + 170) % 255)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        label = f"ID:{obj_id}"
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w + 5, y1), color, -1)
                        cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                cv2.putText(frame, f"TOTAL PEOPLE: {accurate_count}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                
                count_placeholder.markdown(f"""
                <div style="text-align:center;">
                    <h2 style="color:#00ffaa; margin:0;">👥 Total People: {accurate_count}</h2>
                    <p style="color:#00ffff; margin:0;">📈 Max Crowd: {max_people_in_frame}</p>
                    <p style="color:#ffff00; margin:0;">🎯 Model: VisDrone (High Accuracy)</p>
                </div>
                """, unsafe_allow_html=True)
                
                check_range_and_alert(accurate_count, min_detected_range, max_detected_range)

                stframe.image(frame, channels="BGR", width='stretch')
                frames.append(frame)

            cap.release()
            
            st.success("✅ SCAN COMPLETE!")
            
            # Display average confidence for the entire video
            avg_video_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
            st.subheader("📊 Video Analysis Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Frames Processed", frame_count)
            with col2:
                st.metric("Max People Detected", max_people_in_frame)
            with col3:
                st.metric("Average Confidence", f"{avg_video_confidence:.1f}%")
            
            # Confidence gauge for the video
            confidence_fig = display_confidence_gauge(avg_video_confidence)
            st.plotly_chart(confidence_fig, use_container_width=True)
            
            if frames:
                # FIX: Use predefined output path and don't rely on save_video return value
                output_dir = "output"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, "accurate_scan_output.mp4")
                
                # Save video - we'll use the path we defined regardless of return value
                save_video(frames, output_path, original_fps)
                
                # Check if the file was actually created
                if os.path.exists(output_path):
                    st.subheader("🎬 Processed Video")
                    st.video(output_path)
                    
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Processed Video",
                            data=file,
                            file_name="crowd_visdrone_scan.mp4",
                            mime="video/mp4"
                        )
                    st.success(f"✅ Video saved successfully: {output_path}")
                else:
                    st.error("❌ Failed to save processed video. Please check the output directory permissions.")
                    st.info("The video processing completed successfully, but there was an issue saving the output file.")

        progress_bar.empty()
        status_text.empty()

# ------------------------------------------------ LIVE CAMERA MODE ------------------------------------------------
elif option == "📷 Live Camera":
    # Using Standard Nano model for Speed
    load_yolo_model('yolov8n.pt')
    st.caption("ℹ️ Using **YOLOv8n** for fast Real-Time detection.")

    st.markdown("### 🎥 Live Feed (Webcam)")
    run = st.checkbox("✅ Start Camera Stream")

    cap = None
    if run:
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW]:
            temp_cap = cv2.VideoCapture(0, backend)
            if temp_cap.isOpened():
                cap = temp_cap
                st.success(f"✅ Camera initialized using backend: {backend}")
                break
        if not cap or not cap.isOpened():
            st.error("❌ Could not access webcam.")

    if cap and cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

    FRAME_WINDOW = st.empty()
    confidence_col1, confidence_col2 = st.columns(2)

    while run and cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.5)
            continue

        start_time = time.time()
        
        # Lower threshold for live camera too
        detections, people_count, avg_conf = detect_people(frame, is_video=True, conf_threshold=0.30)
        
        clean_detections = remove_duplicate_detections(detections, iou_threshold=0.45)
        accurate_count = len(clean_detections)
        
        tracked_objects = simple_tracker.update(clean_detections)
        
        fps = 1.0 / (time.time() - start_time)
        log_detection(accurate_count, avg_conf, "Live Camera")

        count_placeholder.markdown(f"""
        <div style="text-align:center;">
            <h2 style="color:#00ffaa; margin:0;">👥 Live: {accurate_count}</h2>
            <p style="color:#00ffff; margin:0;">⚡ FPS: {fps:.1f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        check_range_and_alert(accurate_count, min_detected_range, max_detected_range)

        # Update confidence display every 10 frames for performance
        if frame_count % 10 == 0:
            with confidence_col1:
                confidence_fig = display_confidence_gauge(avg_conf)
                st.plotly_chart(confidence_fig, use_container_width=True)

        for obj in tracked_objects:
            if len(obj) >= 5:
                x1, y1, x2, y2, obj_id = map(int, obj[:5])
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{obj_id}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        FRAME_WINDOW.image(frame, channels="BGR", width='stretch')

    if cap:
        cap.release()
    cv2.destroyAllWindows()

# ------------------------------------------------ DATA INSIGHTS ------------------------------------------------
elif option == "📊 Data Insights":
    st.header("📊 Analytics & Confidence Trends")
    try:
        df = pd.read_csv(LOG_FILE)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", len(df))
        col2.metric("Avg Count", f"{df['People_Count'].mean():.1f}")
        col3.metric("Max Count", df['People_Count'].max())
        col4.metric("Avg Conf", f"{df['Confidence(%)'].mean():.1f}%")
        
        # People Count Over Time
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df['Timestamp'], y=df['People_Count'], 
                               mode='lines+markers', name='People Count',
                               line=dict(color='#00ffaa', width=3)))
        fig1.update_layout(
            title="People Count Over Time",
            xaxis_title="Time",
            yaxis_title="People Count",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Confidence Distribution
        fig2 = px.histogram(df, x='Confidence(%)', title='Confidence Distribution',
                          color_discrete_sequence=['#00ffaa'])
        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis_title="Confidence (%)",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        with st.expander("View Raw Data"):
            st.dataframe(df)
    except Exception as e:
        st.info("No data available yet. Process some images or videos to see analytics here.")
        st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown("""<div style="text-align:center;"><p>🤖 <strong>CrowdTrackAI</strong> - Advanced People Detection Analytics</p></div>""", unsafe_allow_html=True)