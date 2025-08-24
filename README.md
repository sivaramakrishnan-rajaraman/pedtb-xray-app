# 🫁 Explainable Pediatric Frontal Chest X-ray Classifier

**Live app:**
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pedtb-xray-app.streamlit.app/)

This app demonstrates an end-to-end pipeline on pediatric frontal chest X-rays:

1. **Custom YOLO-based lung detector (details will be revealed during publication)** detects the **lung region** and **crops** the lung pixels from the original chest X-ray frontal image.  
2. **Custom fine-tuned DL model (details will be revealed during publication)** classifies the crop as **showing Normal lungs** or **TB-related signs**.  
3. **Explainability (CAM family from Jacobgil Pytorch Grad-CAM repository)** overlays heatmaps **(based on the chosen explainability method)** on the cropped lungs when the model predicts chest X-ray as showing TB-related signs and displays the heatmap-overlaid chest X-ray image.*

All model weights are hosted on **Hugging Face Hub** and **downloaded at runtime**.

---

## 🧭 What is Streamlit? What is Streamlit Cloud?

- **Streamlit** is a Python framework for building interactive web apps in a few lines of code. The code `streamlit_app.py` makes Streamlit render the UI (widgets, images, charts) in the browser.
- **Streamlit Cloud** is a hosted service by Streamlit where we point to our **GitHub repository**, and it *automatically* builds the environment, installs dependencies (**based on the requirements.txt file**), and runs the Streamlit app on their servers.  
  - It runs app’s main file (`streamlit_app.py`).
  - Gives a public URL (here, it is `https://pedtb-xray-app.streamlit.app/`).

## 🧱 Repository Structure
```bash
pedtb-xray-app/
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
