# 👁️‍🗨️ **CrowdTrackAI**

### *AI-Powered Real-Time People Detection & Analytics System*

> 🚀 **Smart Surveillance | Live Crowd Counting | AI Safety Monitoring**

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-orange" alt="YOLOv8">
  <img src="https://img.shields.io/badge/Streamlit-1.50.0-red" alt="Streamlit">
  <img src="https://img.shields.io/badge/OpenCV-4.12.0-green" alt="OpenCV">
  <img src="https://img.shields.io/badge/DeepSort-Realtime-yellow" alt="DeepSort">
  <img src="https://img.shields.io/badge/License-MIT-brightgreen" alt="License">
</p>

---

## 🌟 **Overview**

**CrowdTrackAI** is a next-gen AI system for **real-time crowd detection and analytics**.  
It uses **YOLOv8 + Streamlit** to detect, count, and monitor people in live feeds from **CCTV, webcams, and Android IP cameras**, ideal for **smart cities, event safety, and public surveillance**.

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

## 📱 **Use on Android (IP Webcam)**

1. Install **IP Webcam** from Google Play Store  
2. Tap **Start Server**  
3. Copy the stream URL ([https://crowdtrackai-cby2qerx58lzv2ghkrccrf.streamlit.app/])  
4. In **CrowdTrackAI** → choose **📷 Live Camera → Android IP Webcam**  
5. Paste the URL → ✔ **Start Detection**

---

## 🖥️ **Tech Stack**

| Layer | Technology |
|-------|------------|
| 🔍 Detection | YOLOv8 (Ultralytics) |
| 🧠 Tracking | DeepSort-Realtime |
| 🖥️ Frontend | Streamlit |
| 🧮 Vision | OpenCV + NumPy |
| 📊 Analytics | Pandas + Plotly |
| 🔊 Alerts | Streamlit Audio |

---

## 📊 **Sample Output**

| Frame | People Count | Confidence | Crowd Density |
|-------|--------------|------------|----------------|
| 🧍 Frame 1 | 3 | 92.5% | 🟢 Low |
| 🧍 Frame 120 | 28 | 89.8% | 🟡 Medium |
| 🧍 Frame 300 | 56 | 91.1% | 🔴 High (Alert Triggered) |

---

## 🔮 **Future Enhancements**

- ☁️ Cloud dashboard for multi-camera monitoring  
- 📱 Full Android/iOS mobile app  
- 🛰️ IoT + CCTV integration  
- 🤖 Transformer-based crowd prediction models  
- 🔔 Smart AI alert automation system  

---

## 👨‍💻 **Author**

**Sandip Khamkar**

Department of Technology, Savitribai Phule Pune University  
📧 **sandipkhamkar564@gmail.com**

---

## 🪪 **License**

MIT License

This project is open-source and Free for academic & research purposes with proper attribution.

**Empowering Safety Through Artificial Intelligence.**
