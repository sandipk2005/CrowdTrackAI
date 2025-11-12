import streamlit as st
import cv2
import numpy as np
from PIL import Image
from detection import detect_people, load_yolo_model
from utils import draw_tracks, generate_heatmap, overcrowding_alert, save_video
from config import (
    MAX_PEOPLE,
    YOLO_MODELS,
    ENABLE_FPS_DISPLAY,
    ENABLE_DENSITY_METER,
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
</style>
""", unsafe_allow_html=True)

# 🔷 Header
st.markdown("""
<h1>🤖 CrowdTrackAI</h1>
<h3>AI-Powered Real-Time People Detection System</h3>
<div class="scan-line"></div>
""", unsafe_allow_html=True)

# Sidebar model selector
st.sidebar.header("⚙️ YOLO Model Settings")
model_choice = st.sidebar.selectbox(
    "Select YOLO Model",
    options=list(YOLO_MODELS.keys()),
    index=0,
    help="⚡ Speed Mode (yolov8n) or 🎯 Accuracy Mode (yolov8l)"
)
load_yolo_model(model_choice)

# Mode selector
option = st.radio("Select Input Source:", ["🖼️ Image", "🎥 Video", "📷 Live Camera", "📊 Data Insights"], horizontal=True)

# AI Status
st.markdown('<div style="text-align:center;"><span>🧠 AI Status: </span><span class="ai-active">ACTIVE 🔵</span></div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

count_placeholder = st.empty()
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

# ------------------------------------------------ IMAGE MODE ------------------------------------------------
if option == "🖼️ Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        frame = np.array(image)
        start_time = time.time()

        detections, people_count, avg_conf = detect_people(frame)
        fps = 1.0 / (time.time() - start_time)
        log_detection(people_count, avg_conf, "Image")

        count_placeholder.markdown(f"<h2 style='text-align:center;color:#00ffaa;'>🧍 People Detected: {people_count}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align:center;color:#00ffff;'>Detection Accuracy: {avg_conf}%</h4>", unsafe_allow_html=True)

        for det in detections:
            bbox = det[0]
            if len(bbox) == 4:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if ENABLE_HEATMAP:
            heatmap = generate_heatmap(frame, detections)
            frame = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)

        if ENABLE_FPS_DISPLAY:
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if ENABLE_OVERCROWD_ALERT and people_count > MAX_PEOPLE:
            st.warning("🚨 Overcrowding Alert! Area exceeds safe limit.")
            if os.path.exists(ALERT_SOUND):
                st.audio(ALERT_SOUND, format="audio/mp3", start_time=0)

        st.image(frame, channels="BGR", caption=f"Detected: {people_count}")

# ------------------------------------------------ VIDEO MODE ------------------------------------------------
elif option == "🎥 Video":
    video_file = st.file_uploader("📹 Upload Video", type=["mp4", "avi", "mov"])
    if video_file:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(video_file.read())
        temp_video.close()
        cap = cv2.VideoCapture(temp_video.name)
        stframe = st.image([])
        frames = []

        with st.spinner("🧠 AI analyzing video..."):
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                start_time = time.time()
                detections, people_count, avg_conf = detect_people(frame)
                fps = 1.0 / (time.time() - start_time)
                log_detection(people_count, avg_conf, "Video")

                for det in detections:
                    bbox = det[0]
                    if len(bbox) == 4:
                        x, y, w, h = map(int, bbox)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if ENABLE_FPS_DISPLAY:
                    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                count_placeholder.markdown(f"<h2 style='text-align:center;color:#00ffaa;'>🧍 People Detected: {people_count}</h2>", unsafe_allow_html=True)
                st.markdown(f"<h4 style='text-align:center;color:#00ffff;'>Detection Accuracy: {avg_conf}%</h4>", unsafe_allow_html=True)
                stframe.image(frame, channels="BGR", width="stretch")
                frames.append(frame)

            cap.release()
            save_video(frames, "output/output_video.mp4")
        st.success("✅ Video processed successfully!")
        st.video("output/output_video.mp4")

# ------------------------------------------------ LIVE CAMERA MODE ------------------------------------------------
elif option == "📷 Live Camera":
    st.markdown("### 🎥 Choose your camera source:")
    camera_type = st.radio(
        "Select Camera Type:",
        ["💻 Laptop/PC Camera", "📱 Android Camera (IP Webcam)"],
        horizontal=True
    )

    run = st.checkbox("✅ Start Camera Stream")

    # Initialize camera safely
    cap = None
    if camera_type == "💻 Laptop/PC Camera":
        # Try all Windows-compatible backends automatically
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW]:
            temp_cap = cv2.VideoCapture(0, backend)
            if temp_cap.isOpened():
                cap = temp_cap
                st.success(f"✅ Camera initialized using backend: {backend}")
                break
        if not cap or not cap.isOpened():
            st.error("❌ Could not access webcam. Try closing other camera apps or restarting your PC.")
    else:
        ip_url = st.text_input("📱 Enter Android camera stream URL (e.g. http://192.168.1.5:8080/video):")
        if ip_url.strip():
            cap = cv2.VideoCapture(ip_url)
            st.info("📶 Connected to Android IP camera stream.")
        else:
            st.warning("Please enter your Android camera URL to start the feed.")
            cap = None

    # Configure capture properties for smooth performance
    if cap and cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

    FRAME_WINDOW = st.image([])

    while run and cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            st.error("⚠️ Frame not received. Check camera connection.")
            time.sleep(0.5)
            continue

        # YOLOv8 detection
        start_time = time.time()
        detections, people_count, avg_conf = detect_people(frame)
        fps = 1.0 / (time.time() - start_time)
        log_detection(people_count, avg_conf, "Live Camera")

        # Draw detections
        for det in detections:
            bbox = det[0]
            if len(bbox) == 4:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Person", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show FPS
        if ENABLE_FPS_DISPLAY:
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        FRAME_WINDOW.image(frame, channels="BGR", width="stretch")
        count_placeholder.markdown(
            f"<h2 style='text-align:center;color:#00ffaa;'>🧍 People Detected: {people_count}</h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<h4 style='text-align:center;color:#00ffff;'>Detection Accuracy: {avg_conf}%</h4>",
            unsafe_allow_html=True
        )

        time.sleep(0.02)

    # Safe release
    if cap:
        cap.release()
    st.info("🛑 Stream stopped.")


# ------------------------------------------------ INSIGHTS DASHBOARD ------------------------------------------------
elif option == "📊 Data Insights":
    st.markdown("## 📈 Detection Insights Dashboard")

    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        st.warning("⚠️ No log data found. Run detections first.")
    else:
        df = pd.read_csv(LOG_FILE)
        total_frames = len(df)
        total_people = df["People_Count"].sum()
        max_crowd = df["People_Count"].max()
        avg_conf = round(df["Confidence(%)"].mean(), 2)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🧍 Total People", f"{total_people}")
        col2.metric("📸 Frames", f"{total_frames}")
        col3.metric("👥 Max Crowd", f"{max_crowd}")
        col4.metric("🎯 Avg. Confidence", f"{avg_conf}%")

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df["People_Count"], mode="lines+markers", name="People Count", line=dict(color="lime", width=3)))
        fig.add_trace(go.Scatter(y=df["Confidence(%)"], mode="lines+markers", name="Confidence (%)", line=dict(color="cyan", width=2, dash="dot")))
        fig.update_layout(title="🧠 Detection Trends", xaxis_title="Frame Index", yaxis_title="Value", template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df.tail(20), use_container_width=True)
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Export Full Log as CSV", data=csv_data, file_name="CrowdTrackAI_Log.csv", mime="text/csv")

# ------------------------------------------------ Footer ------------------------------------------------
st.markdown("""
<hr>
<h5 style='text-align:center;color:#00ffff;'>Made with 🧠 by VisAI Labs | Powered by YOLOv8 + Streamlit</h5>
""", unsafe_allow_html=True)
