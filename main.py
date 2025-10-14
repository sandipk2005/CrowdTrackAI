import streamlit as st
import cv2
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from detection import detect_people
from tracking import update_tracks, tracker
from utils import draw_tracks, generate_heatmap, overcrowding_alert, save_video
from config import MAX_PEOPLE, VIDEO_OUTPUT

st.title("CrowdTrackAI – Advanced People Counting & Analytics")

option = st.radio("Select Input", ["Image", "Video", "Live Camera"])
frames_list = []

# ================= IMAGE MODE =================
if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        detections = detect_people(frame)

        # ✅ Safety check before tracking
        if detections is None or len(detections) == 0:
            st.error("No people detected! Please upload a clearer image.")
        else:
            try:
                tracks = update_tracks(detections, frame)
            except Exception as e:
                st.error(f"Tracking Error: {e}")
                tracks = []

            frame = draw_tracks(frame, tracks)
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"People: {len(tracks)}")

            # HEATMAP
            heatmap = generate_heatmap(tracks, frame.shape)
            plt.imshow(heatmap, cmap='hot', interpolation='nearest')
            st.pyplot(plt)

            # ALERT
            alert = overcrowding_alert(len(tracks))
            if alert:
                st.warning(alert)


# ================= VIDEO MODE =================
elif option == "Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = detect_people(frame)

            if detections is None or len(detections) == 0:
                tracks = []
            else:
                try:
                    tracks = update_tracks(detections, frame)
                except Exception as e:
                    st.error(f"Tracking Error: {e}")
                    tracks = []

            frame = draw_tracks(frame, tracks)
            frames_list.append(frame)

            alert = overcrowding_alert(len(tracks))
            if alert:
                st.warning(alert)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
        save_video(frames_list, VIDEO_OUTPUT)
        st.success(f"Processed video saved: {VIDEO_OUTPUT}")

        # HEATMAP for entire video
        if frames_list:
            all_tracks = tracker.tracks
            heatmap = generate_heatmap(all_tracks, frames_list[0].shape)
            plt.imshow(heatmap, cmap='hot', interpolation='nearest')
            st.pyplot(plt)


# ================= LIVE CAMERA =================
elif option == "Live Camera":
    st.info("Live camera works only locally. Run the app using:  streamlit run main.py")