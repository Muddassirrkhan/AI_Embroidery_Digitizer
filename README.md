# 🧵 AI Embroidery Digitizer (MVP)

A simple AI-powered embroidery digitizer that converts images into machine-readable `.DST` stitch files.

## ✨ Features
- Upload any logo or image
- Auto digitize to stitches
- Adjustable color (K-Means) segmentation
- Preview stitches before export
- Manual pencil and eraser tools
- Export `.DST` embroidery file

## 🧰 Tech Stack
- Python, FastAPI
- OpenCV, PyEmbroidery, NumPy
- HTML + JS (frontend)
- Matplotlib (for preview)

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/muddassirrkhan/AI_Embroidery_Digitizer.git
2.Install dependencies:
  pip install -r requirements.txt
3.Run the app:
  uvicorn app:app --reload --port 8000
4.Open in browser:
  http://127.0.0.1:8000
