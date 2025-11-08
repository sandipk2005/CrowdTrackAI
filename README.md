# 👁️ CrowdTrackAI – AI-Powered Real-Time People Detection System  

> 🎯 *Smart Surveillance | Real-Time Analytics | Crowd Safety*  

---

## 🧭 Overview  
**CrowdTrackAI** is an **AI-driven crowd monitoring and analysis system** designed to detect, track, and count people in **real-time video streams**.  
Using **Computer Vision** and **Deep Learning (YOLOv8)**, it enables **intelligent crowd management** in public areas like **malls, stations, and events**.  

This project enhances **public safety**, **event planning**, and **smart city infrastructure** by providing **live analytics and crowd density insights**.

---

## ⚙️ Key Features  

| 🌟 Feature | 🔍 Description |
|-------------|----------------|
| 🧠 **Real-Time Detection** | Detects and tracks people using YOLOv8 deep learning model. |
| 📊 **Automatic Crowd Counting** | Counts people accurately in dense crowds. |
| 🎥 **Live Video Feed Analysis** | Processes CCTV, webcam, or video file inputs. |
| 📈 **Analytics Dashboard** | Displays visual crowd metrics and trends. |
| ⚙️ **Streamlit Web App** | Simple, interactive interface for live monitoring. |
| 🧾 **Downloadable Reports** | Generate CSV/PDF summaries of crowd data. |
| 🔐 **Scalable & Efficient** | Optimized for CPU/GPU performance. |
| 💡 **Custom Model Training** | Supports re-training with new crowd datasets. |

---

## 🧩 Dataset Information  

| Attribute | Details |
|------------|----------|
| **Name** | CrowdHuman Dataset |
| **Source** | [CrowdHuman.org](https://www.crowdhuman.org/) |
| **Format** | `.jpg` images with `.json` bounding box annotations |
| **Purpose** | Human detection and crowd density estimation |
| **Size** | ~15,000 labeled crowd images |
| **Environment** | Real-world crowded public spaces |

📦 *Used for training YOLOv8 model to recognize and count humans accurately in complex, crowded environments.*

---

## 💻 Tech Stack  

| Component | Technology Used |
|------------|----------------|
| 🖥️ **Frontend** | Streamlit |
| ⚙️ **Backend / Model** | Python, YOLOv8 (Ultralytics) |
| 🎞️ **Computer Vision** | OpenCV |
| 📊 **Visualization** | Matplotlib, Seaborn |
| 📂 **Data Handling** | NumPy, Pandas |
| 💾 **Development Environment** | Jupyter Notebook / VS Code |

---

## 🤖 Model Summary  

| Attribute | Description |
|------------|--------------|
| **Model Type** | Object Detection (YOLOv8) |
| **Objective** | Detect and count people in real-time |
| **Input** | Live video frames or images |
| **Output** | Bounding boxes, labels, person count |
| **Performance Metrics** | mAP (mean Average Precision), FPS |
| **Training Framework** | Ultralytics YOLO Library |

### 🧠 Model Workflow  
1️⃣ **Data Preprocessing** → Resize, normalize, and label images  
2️⃣ **Model Training** → Train YOLOv8 on annotated dataset  
3️⃣ **Validation** → Evaluate accuracy and precision  
4️⃣ **Integration** → Deploy model within Streamlit app  
5️⃣ **Live Detection** → Perform real-time people detection & analytics  

---

## 🗓️ Project Timeline (2 Weeks)  

| Week | Milestone | Status |
|------|------------|---------|
| **Week 1** | Data collection, preprocessing, model selection, EDA | ✅ Completed |
| **Week 2** | Model training, Streamlit dashboard integration, testing, deployment | ✅ Completed |

---

## 🚀 Future Enhancements  

- 🌐 Integration with multi-angle CCTV systems  
- ☁️ Cloud storage for live analytics and logs  
- 📱 Mobile app for remote crowd monitoring  
- 🧩 Deep learning upgrades (Hybrid CNN + Transformer models)  
- 🛰️ IoT & smart camera network support  
- 🔔 Real-time alerts for overcrowding or abnormal activity  
- 🌿 Energy-efficient model optimization  

---

## 👨‍💻 Author Information  

| Field | Details |
|--------|----------|
| **Name** | Sandip A. Khamkar |
| **University** | Department of Technology, Savitribai Phule Pune University |
| **Email** | [sandipkhamkar564@gmail.com]|

---

## 📜 License  

This project is released under the **MIT License**, allowing anyone to **use, modify, and distribute** the software for **academic, research, or personal development purposes** with proper attribution.  

By keeping it open-source, we encourage **collaboration, transparency, and learning** in the field of **AI and Computer Vision**.  

> 💡 *Build. Learn. Innovate. Together, we make technology smarter and safer.*

---

## 🏁 Conclusion  

**CrowdTrackAI** showcases the potential of **AI and Computer Vision** in improving **public safety** and **smart surveillance**.  
With real-time detection, analytics, and alerting capabilities, it’s a step toward **intelligent crowd management** for safer and smarter cities. 🌍  

---

⭐ **If you find this project useful, don’t forget to star the repository and share it!**  
💡 *Empowering Safety Through AI – CrowdTrackAI*  
