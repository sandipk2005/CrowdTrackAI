import streamlit as st
import cv2
import numpy as np
from PIL import Image
from detection import detect_people
from tracking import update_tracks, tracker
from utils import draw_tracks, generate_heatmap, overcrowding_alert, save_video
from config import MAX_PEOPLE

# 🌐 Streamlit Page Config
st.set_page_config(page_title="CrowdTrackAI – Smart Detection", page_icon="🤖", layout="wide")

# 💠 --- CUSTOM AI STYLE ---
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

# 🔷 AI Header
st.markdown("""
<h1>🤖 CrowdTrackAI</h1>
<h3>AI-Powered Real-Time People Detection System</h3>
<div class="scan-line"></div>
""", unsafe_allow_html=True)

# 🧩 Mode Selection
option = st.radio("Select Input Source:", ["🖼️ Image", "🎥 Video", "📷 Live Camera"], horizontal=True)

# 🟢 AI Status
st.markdown('<div style="text-align:center;"><span>🧠 AI Status: </span><span class="ai-active">ACTIVE 🔵</span></div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Live Counter Placeholder
count_placeholder = st.empty()

# ---------------- IMAGE MODE -----------------
if option == "🖼️ Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        frame = np.array(image)
        detections = detect_people(frame)
        count_placeholder.markdown(f"<h2 style='text-align:center;color:#00ffaa;'>🧍 People Detected: {len(detections)}</h2>", unsafe_allow_html=True)
        tracks = [(i, [x,y,x+w,y+h], "person") for i, ([x,y,w,h],_,_) in enumerate(detections)]
        frame = draw_tracks(frame, tracks)
        st.image(frame, channels="BGR", caption=f"Detected: {len(detections)}")

# ---------------- VIDEO MODE -----------------
elif option == "🎥 Video":
    video_file = st.file_uploader("Upload Video", type=["mp4","avi","mov"])
    if video_file:
        tfile = open("temp.mp4","wb")
        tfile.write(video_file.read())
        cap = cv2.VideoCapture("temp.mp4")
        frames=[]
        total=0
        with st.spinner("🧠 AI analyzing video..."):
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                detections = detect_people(frame)
                total = len(detections)
                tracks = update_tracks(detections)
                frame = draw_tracks(frame, tracks)
                frames.append(frame)
            cap.release()
            save_video(frames, "output.mp4")
        count_placeholder.markdown(f"<h2 style='text-align:center;color:#00ffaa;'>🧍 People Detected: {total}</h2>", unsafe_allow_html=True)
        st.video("output.mp4")

# ---------------- LIVE CAMERA -----------------
elif option == "📷 Live Camera":
    st.info("🕶️ Live camera streaming feature coming soon...")

# Footer
st.markdown("""
<hr>
<h5 style='text-align:center;color:#00ffff;'>Made with 🧠 by VisAI Labs | Powered by YOLOv8 + Streamlit</h5>
""", unsafe_allow_html=True)
