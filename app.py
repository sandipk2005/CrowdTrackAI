import streamlit as st
import cv2
import numpy as np
from PIL import Image
from detection import detect_people
from tracking import update_tracks, tracker
from utils import draw_tracks, generate_heatmap, overcrowding_alert, save_video
from config import MAX_PEOPLE
import tempfile
import time


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
    import tempfile
    import numpy as np
    import cv2
    import streamlit as st
    from ultralytics import YOLO

    # Load YOLO model
    model = YOLO("yolov8n.pt")

    # Function to detect people
    def detect_people(frame):
        results = model(frame)
        detections = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:  # class 0 = person
                    detections.append(box.xyxy[0].tolist())
        return detections

    # Save processed video
    def save_video(frames, output_path):
        if not frames:
            return
        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()

    # Upload video
    video_file = st.file_uploader("📹 Upload Video", type=["mp4", "avi", "mov"])
    if video_file:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(video_file.read())
        temp_video.close()

        cap = cv2.VideoCapture(temp_video.name)

        # Streamlit placeholders
        stframe = st.empty()             # for showing video frames
        count_placeholder = st.empty()   # for real-time count text
        frames = []

        with st.spinner("🧠 AI analyzing video..."):
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detection
                detections = detect_people(frame)
                valid_boxes = []

                for det in detections:
                    try:
                        x1, y1, x2, y2 = map(int, np.array(det[:4]).flatten())
                        valid_boxes.append((x1, y1, x2, y2))
                    except Exception as e:
                        print("⚠️ Detection error:", e)

                # Draw boxes
                for (x1, y1, x2, y2) in valid_boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Person", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Count
                total_people = len(valid_boxes)

                # ✅ Styled real-time display
                count_placeholder.markdown(
                    f"""
                    <div style='text-align:center; margin-top:15px;'>
                        <h2 style='color:#00ffaa; font-size:28px; text-shadow:0 0 15px #00ffaa;'>
                            🧍 People Detected: {total_people}
                        </h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Show video frame
                stframe.image(frame, channels="BGR", use_container_width=True)
                frames.append(frame)

            cap.release()
            save_video(frames, "output/output_video.mp4")

        st.success("✅ Video processed successfully!")
        st.video("output/output_video.mp4")


# ---------------- LIVE CAMERA -----------------
elif option == "📷 Live Camera":
    st.warning("🎥 Live Camera Mode – Press 'Stop' to end streaming.")
    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Unable to access camera")
            break

        detections = detect_people(frame)
        if detections is None or len(detections) == 0:
            tracks = []
        else:
            tracks = update_tracks(detections, frame)

        # ✅ FIXED: draw bounding boxes safely
        for det in detections:
            if isinstance(det[0], (list, tuple, np.ndarray)):
                det = det[0]
            x, y, w, h = [int(v) for v in det[:4]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        frame = draw_tracks(frame, tracks)

        count_placeholder.markdown(
            f"<h2 style='text-align:center;color:#00ffaa;'>🧍 People Detected: {len(detections)}</h2>",
            unsafe_allow_html=True,
        )

        FRAME_WINDOW.image(frame, channels="BGR")
        time.sleep(0.02)

    else:
        cap.release()
        st.info("🛑 Camera stopped.")



# Footer
st.markdown("""
<hr>
<h5 style='text-align:center;color:#00ffff;'>Made with 🧠 by VisAI Labs | Powered by YOLOv8 + Streamlit</h5>
""", unsafe_allow_html=True)

