🧪 Syringe Inspection Using Image Processing Techniques
📘 Overview

This project was developed as part of the Image and Signal Processing (ISP-AD23-FS25) course.
It focuses on automated visual inspection of pre-filled syringes to detect manufacturing defects such as bubbles, cracks, or impurities.

Manual inspection in pharmaceutical manufacturing is time-consuming and prone to human error.
Our system leverages computer vision and machine learning to improve both accuracy and efficiency in quality control.

👥 Authors

Hädener Anja

Heini Sara

Huber Yeji

Date: 14 May 2025

⚙️ Setup
1. Install Required Libraries
pip install numpy matplotlib seaborn opencv-python scikit-learn tensorflow

2. Environment Configuration

Python ≥ 3.9

Jupyter Notebook

Recommended IDE: Visual Studio Code or JupyterLab

📂 Dataset Structure

The dataset folder is organized as follows:

Folder	Description
images/	Raw syringe images collected for analysis
labels/	CSV or JSON annotation files describing defect categories
processed/	Pre-processed and filtered images ready for training
models/	Saved model checkpoints
results/	Visualization outputs and evaluation metrics
🧠 Implementation Pipeline

The project follows a multi-stage image analysis pipeline:

Preprocessing

Grayscale conversion

Gaussian blurring and edge enhancement

Background normalization

Feature Extraction

Contour detection

Texture and shape descriptors

Statistical feature computation

Classification

Supervised models (e.g., Random Forest, CNN)

Model evaluation with confusion matrix & classification report

Visualization

Seaborn and Matplotlib used for data insights and results visualization

🔍 Example Output

Defective syringe images are automatically highlighted with bounding boxes.

The classifier outputs defect categories (e.g., scratch, bubble, foreign particle).

Performance metrics such as precision, recall, F1-score are displayed after training.

🧩 Key Libraries

OpenCV – Image preprocessing and feature extraction

TensorFlow / Keras – Deep learning model implementation

scikit-learn – Classical machine learning & evaluation

Matplotlib / Seaborn – Visualization

📊 Evaluation Metrics

Accuracy

Precision, Recall, F1-Score

Confusion Matrix

ROC Curves (optional)

🚀 Future Improvements

Integrate real-time inspection using camera feed

Deploy trained model into an embedded inspection system

Expand dataset with multiple syringe types and lighting conditions

🧾 License

This project was developed for academic purposes under the ISP-AD23-FS25 module.
Use or modification is permitted for educational and research contexts only.