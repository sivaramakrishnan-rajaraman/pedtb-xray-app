# 🫁 Explainable Pediatric Chest X-ray Classifier

**Live app:** *(update this link after deploying)*  
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sivaramakrishhnan.streamlit.app/)

This app demonstrates a full pipeline on CXR images:

1. **YOLO12s** (fine-tuned) to detect the **pediatric lung** region and **crop** the detected lungs.
3. **DPN-68** (fine-tuned) to classify **pediatric** CXRs as showing **Normal** lungs or with **Tuberculosis-related** manifestations.
4. **XAI** visualizations (heatmap, contours, bboxes) superimposed on CXRs to explain the **learned behavior**. 

Weights are hosted on **Hugging Face Hub**:

## 🧱 Repository Structure
```bash
pneumonia-xray-app/
├─ streamlit_app.py # Main Streamlit entry point
├─ requirements.txt
├─ README.md
├─ .gitignore
└─ src/
├─ hf_utils.py # Helper to download from Hugging Face Hub
├─ yolo_utils.py # YOLO detection + crop utilities
├─ model.py # PneumoniaModel (DPN-68 compatible)
└─ cam_utils.py # Grad-CAM utilities
```
## 🚀 Run locally
```bash
# 1) Clone repo
git clone https://github.com/sivaramakrishnan-rajaraman/pedtb-xray-app.git
cd pneumonia-xray-app

# 2) Create env (optional) & install deps
pip install -r requirements.txt

# 3) Launch app
streamlit run streamlit_app.py
```
