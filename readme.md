🧪 Syringe Inspection Using Image Processing Techniques

📘 Overview

This project was developed for the Image and Signal Processing (ISP-AD23-FS25) course. It focuses on automated visual inspection of pre-filled syringes to detect defects such as bubbles, cracks, or impurities. The system leverages computer vision and machine learning to improve accuracy and efficiency over manual inspection.

👥 Authors

- Hädener Anja
- Heini Sara
- Huber Yeji

Date: 14 May 2025

⚙️ Setup

- Requirements: Python ≥ 3.9, Jupyter Notebook (recommended IDE: Visual Studio Code or JupyterLab)
- Install libraries:

```bash
pip install numpy matplotlib seaborn opencv-python scikit-learn tensorflow
```

📂 Dataset Structure

The dataset folder is organized as follows:

| Folder     | Description                                                   |
|------------|---------------------------------------------------------------|
| images/    | Raw syringe images collected for analysis                     |
| labels/    | CSV or JSON annotation files describing defect categories     |
| processed/ | Pre-processed and filtered images ready for training          |
| models/    | Saved model checkpoints                                       |
| results/   | Visualization outputs and evaluation metrics                  |

🧠 Implementation Pipeline

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
  - Model evaluation (confusion matrix, classification report)
- Visualization
  - Data insights and results plotted with Seaborn and Matplotlib

🔍 Example Output

- Defective syringe regions highlighted with bounding boxes
- Predicted defect categories (e.g., scratch, bubble, foreign particle)
- Reported metrics after training (precision, recall, F1-score)

🧩 Key Libraries

- OpenCV — Image preprocessing and feature extraction
- TensorFlow / Keras — Deep learning model implementation
- scikit-learn — Classical machine learning & evaluation
- Matplotlib / Seaborn — Visualization

📊 Evaluation Metrics

- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curves (optional)

🚀 Future Improvements

- Integrate real-time inspection using a camera feed
- Deploy the trained model to an embedded inspection system
- Expand the dataset across syringe types and lighting conditions

🧾 License

This project was developed for academic purposes under the ISP-AD23-FS25 module. Use or modification is permitted for educational and research contexts only.