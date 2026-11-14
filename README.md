
# Industrial Defect Detection System

An AI-powered industrial anomaly detection system that identifies defective bottle images and highlights suspicious regions using anomaly heatmaps.

## Features

- Normal vs defective bottle classification
- Anomaly score calculation
- Pixel-level anomaly heatmap
- Heatmap overlay on original images
- FastAPI backend
- Streamlit interactive dashboard
- CPU-based local inference

## Model Architecture

This project uses a lightweight PatchCore-like anomaly detection approach.

- Pretrained ResNet18 feature extractor
- Layer2 and Layer3 feature extraction
- Feature concatenation
- Coreset-based memory bank
- Nearest-neighbor anomaly scoring
- Pixel-level anomaly localization

**Note:** This is an educational PatchCore-like implementation, not the official PatchCore implementation.

## Dataset

MVTec AD Bottle category.

- 209 normal training images
- 83 test images
- 63 ground-truth defect masks

## Evaluation Results

| Metric | Score |
|---|---:|
| Image-level ROC-AUC | 1.0000 |
| Pixel-level ROC-AUC | 0.9778 |
| Autoencoder baseline ROC-AUC | 0.4452 |

The evaluation uses a small test set of 83 images. These results should not be interpreted as proof of production-level reliability.

## Technology Stack

- Python
- PyTorch
- Torchvision
- ResNet18
- FastAPI
- Streamlit
- NumPy
- Pillow

## Project Structure

```text
Industrial-Defect-Detection/
├── backend/
│   ├── app/
│   ├── models/
│   │   └── patchcore_final.pth
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   └── dashboard.py
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/hamad-niaz/industrial-defect-detection.git
cd industrial-defect-detection
```

Create virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r backend/requirements.txt
```

## Run Backend

From the project root:

```bash
python -m uvicorn backend.main:app --reload
```

API documentation:

http://127.0.0.1:8000/docs

## Run Dashboard

Open a second terminal:

```bash
streamlit run frontend/dashboard.py
```

Dashboard:

http://localhost:8501

## Important Limitations

- The decision threshold of 2.85 is provisional.
- Anomaly score is not a probability.
- Heatmaps show anomaly intensity, not confirmed defect boundaries.
- More testing on independent industrial images is required before production use.

