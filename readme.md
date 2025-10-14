# 🧪 Syringe Inspection Using Image Processing Techniques

## 📘 Overview

This project was developed for the Image and Signal Processing (ISP-AD23-FS25) course. It focuses on automated visual inspection of pre-filled syringes to detect defects such as bubbles, cracks, or impurities. The system leverages computer vision and machine learning to improve accuracy and efficiency over manual inspection.

## 👥 Authors

- Hädener Anja
- Heini Sara
- Huber Yeji

Date: 14 May 2025

## ⚙️ Setup

- Requirements: Python ≥ 3.9, Jupyter Notebook (recommended IDE: Visual Studio Code or JupyterLab)
- Install libraries:

```bash
pip install numpy matplotlib seaborn opencv-python scikit-learn tensorflow
```

## 📂 Dataset Structure

The dataset folder is organized as follows:

| Folder | Description |
|---|---|
| cropped ROI | We experimented with cropping images to a smaller region. However, this approach did not improve model performance. |
| dataset_modeltraining | This folder contains the ROIs used for model training. Images were captured with a smartphone, resulting in high-quality, full-syringe ROIs. |
| photo with labtop webcam | Initial attempts to capture images using a laptop webcam. Quality and resolution were too low. Training on these images led to poor performance. |
| Result_data | This folder contains all the ROIs that have been created and used for live testing. These are used in the final evaluation and result analysis. |

## 🧠 Implementation Pipeline

The analysis follows a multi-stage pipeline:

- Preprocessing
  - Grayscale conversion
  - Gaussian blurring and edge enhancement
  - Background normalization
- Feature Extraction
  - Contour detection
  - Texture and shape descriptors
  - Statistical feature computation
- Classification
  - Supervised models (e.g., Random Forest, CNN)
  - Model evaluation (classification report)
- Visualization
  - Data insights and results plotted with Seaborn and Matplotlib

## 🔍 Example Output

- Defective syringe regions highlighted with bounding boxes
- Predicted defect categories (e.g., scratch, bubble, foreign particle)
- Reported metrics after training (precision, recall, F1-score)

## 🧩 Key Libraries

- OpenCV — Image preprocessing and feature extraction
- TensorFlow / Keras — Deep learning model implementation
- scikit-learn — Classical machine learning & evaluation
- Matplotlib / Seaborn — Visualization

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score

## 🚀 Future Improvements

- Add object detection methods (e.g. YOLOv8) to explicitly localize defects instead of relying on binary classification.
- Use larger, more diverse datasets with real-world production variability.
- We attempted post-verification by tuning various parameters to visually confirm defects, but it was unsuccessful; in future work, we aim to improve this step by finding better parameters or applying more advanced detection techniques.

## 🧾 License

This project was developed for academic purposes under the ISP-AD23-FS25 module. Use or modification is permitted for educational and research contexts only.