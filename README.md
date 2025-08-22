# 🫁 Explainable Pediatric Chest X-ray Classifier

**Live app:** *(update this link after deploying)*  
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sivaramakrishhnan.streamlit.app/)

This app demonstrates a full pipeline on CXR images:

1. **YOLO12s** (fine-tuned) to detect the **lung** region (single class).
2. **Crop** the detected lungs.
3. **DPN-68** (fine-tuned) to classify **Normal** vs **Abnormal**.
4. **Grad-CAM** visualizations (heatmap, contours, bboxes) over the cropped lungs.

Weights are hosted on **Hugging Face Hub**:
- YOLO detector: `sivaramakrishhnan/cxr-yolo12s-lung` → `best.pt`
- Classifier  : `sivaramakrishhnan/cxr-dpn68-tb-cls` → `dpn68_fold2.ckpt`

## 🧱 Repository Structure
```bash
pneumonia-xray-app/
├─ streamlit_app.py # Main Streamlit entry point
├─ requirements.txt
├─ README.md
├─ .gitignore
├─ .streamlit/
│ └─ secrets.toml # Optional (only if HF repos are private)
└─ src/
├─ hf_utils.py # Helper to download from Hugging Face Hub
├─ yolo_utils.py # YOLO detection + crop utilities
├─ model.py # PneumoniaModel (DPN-68 compatible)
└─ cam_utils.py # Grad-CAM utilities
---

## 🚀 Run locally

```bash
# 1) Clone your repo
git clone https://github.com/sivaramakrishnan-rajaraman/pneumonia-xray-app.git
cd pneumonia-xray-app

# 2) Create env (optional) & install deps
pip install -r requirements.txt

# 3) Launch app
streamlit run streamlit_app.py
