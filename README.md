# 👁️‍🗨️ **CrowdTrackAI**

### *AI-Powered Real-Time People Detection & Analytics System*

> 🚀 **Smart Surveillance | Live Crowd Counting | AI Safety Monitoring**

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-orange?logo=ai" alt="YOLOv8">
  <img src="https://img.shields.io/badge/Streamlit-1.50.0-red?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/OpenCV-4.12.0-green?logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/DeepSort-Realtime-yellow?logo=github" alt="DeepSort">
  <img src="https://img.shields.io/badge/License-MIT-brightgreen" alt="License">
</p>

---

## 🌟 **Overview**

**CrowdTrackAI** is a next-gen AI system for **real-time crowd detection and analytics**.  
It uses **YOLOv8 + Streamlit** to detect, count, and monitor people in live feeds from **CCTV, webcams, and Android IP cameras** — ideal for **smart cities, event safety, and public surveillance**.

---

## ⚙️ **Core Features**

| Feature | Description |
|--------|-------------|
| 🚶 **Real-Time Detection** | YOLOv8 detects people frame-by-frame instantly |
| 📊 **Crowd Counting Dashboard** | Live people counter with accuracy metrics |
| 🎥 **Multi-Camera Input** | Works with webcam, CCTV, or Android IP Webcam |
| ⚡ **Speed / Accuracy Modes** | YOLOv8-Small (Fast) / YOLOv8-Large (Accurate) |
| 🚨 **Overcrowding Alerts** | Automatic sound + visual warnings |
| 📈 **Data Logging & Insights** | Generates CSV logs & analytical charts |
| 🌐 **Cross-Platform Ready** | Runs on PC, Web, and Android (via IP camera) |

---

## 🧩 **Architecture**

📷 Camera Feed / IP Stream
↓
🎯 YOLOv8 Detector → 🧮 People Counter
↓
📊 Streamlit Dashboard (FPS, Accuracy, Crowd Trend)
↓
🧾 CSV Log + Crowd Alerts + Live Graphs


---

## ⚙️ **Quick Setup**

```bash
# 1️⃣ Clone Repository
git clone https://github.com/sandipk2005/CrowdTrackAI.git
cd CrowdTrackAI

# 2️⃣ Install Dependencies (CUDA 12.1 supported)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# 3️⃣ Run Application
streamlit run app.py

📱 Use on Android (IP Webcam)

Install IP Webcam app on Android

Tap Start Server

Copy the streaming URL (example: http://192.168.1.8:8080/video)

In CrowdTrackAI → Choose 📷 Live Camera → Android IP Webcam

Paste the URL → ✅ Start Detection


🖥️ Tech Stack
Layer	Technology
🔍 Detection	YOLOv8 (Ultralytics)
🧠 Tracking	DeepSort-Realtime
🖥️ Frontend	Streamlit
🧮 Vision	OpenCV + NumPy
📊 Analytics	Pandas + Plotly
🔊 Alerts	Audio via Streamlit
📊 Sample Output
Frame	People Count	Confidence	Crowd Density
🧍 Frame 1	3	92.5%	🟢 Low
🧍 Frame 120	28	89.8%	🟡 Medium
🧍 Frame 300	56	91.1%	🔴 High (Alert Triggered)
🔮 Future Enhancements

☁️ Cloud dashboard for multi-camera monitoring

📱 Full Android/iOS mobile app

🛰️ IoT + CCTV hardware integration

🤖 Transformer-based crowd prediction models

🔔 Smart AI automation alerts


**👨‍💻 Developer**

Sandip A. Khamkar
🎓 Department of Technology, Savitribai Phule Pune University
📧 sandipkhamkar564@gmail.com

**🪪 License**

MIT License © 2025 — Free for research & innovation.

🧠 Empowering Safety Through Artificial Intelligence.

⭐ Star this repository if you find it helpful!
💡 Build safer, smarter spaces with CrowdTrackAI.