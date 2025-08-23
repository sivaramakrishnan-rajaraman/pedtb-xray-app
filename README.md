# 🫁 Explainable Pediatric Chest X-ray Classifier (PedTB)

**Live app:**
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pedtb-xray-app.streamlit.app/)

This app demonstrates an end-to-end pipeline on pediatric chest X-rays:

1. **YOLOv8-s (custom)** detects the **lung region** and we **crop** the lungs from the original image.  
2. **DPN-68 (fine-tuned)** classifies the crop as **Normal** or **TB-related (normal_not)**.  
3. **Explainability (Grad-CAM family)** overlays heatmaps on the cropped lungs *only when the prediction is TB-related.*

All model weights are hosted on **Hugging Face Hub** (downloaded at runtime).

---

## 🧭 What is Streamlit? What is Streamlit Cloud?

- **Streamlit** is a Python framework for building interactive web apps in a few lines of code. You write a `streamlit_app.py`, and Streamlit renders the UI (widgets, images, charts) in the browser.
- **Streamlit Cloud** is a hosted service by Streamlit where you point to your **GitHub repository**, and it *automatically* builds and runs your Streamlit app on their servers.  
  - It uses your repo’s `requirements.txt` to install dependencies.
  - It runs your app’s main file (e.g., `streamlit_app.py`).
  - You get a public URL like `https://<your-username>.streamlit.app/`.

## 🧱 Repository Structure
```bash
pneumonia-xray-app/
├─ streamlit_app.py # Main Streamlit entry point
├─ requirements.txt
├─ runtime.txt # Python version pin for Streamlit Cloud ("3.11")
├─ README.md
├─ .gitignore
└─ src/
├─ hf_utils.py # Helper to download from Hugging Face Hub
├─ tb_model.py 
└─ cam_utils.py # Grad-CAM utilities (uses jacobgil/pytorch-grad-cam)
```
## 🖥️ Local Setup & Run

```bash
# 1) Clone repo
git clone https://github.com/sivaramakrishnan-rajaraman/pedtb-xray-app.git
cd pedtb-xray-app

# 2) (Optional) Create a clean Python env (recommend Python 3.11)
#    e.g., using conda or venv

# 3) Install dependencies
pip install -r requirements.txt

# 4) Launch app
streamlit run streamlit_app.py

```
