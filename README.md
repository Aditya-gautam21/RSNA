# 🫁 RSNA Pneumonia Detection using CNN from Scratch (PyTorch)

This repository contains a complete end-to-end deep learning pipeline for detecting Pneumonia from chest X-ray images using a **Convolutional Neural Network built from scratch in PyTorch**, trained on the **RSNA Pneumonia Detection Challenge** dataset.

The project is implemented entirely in a Jupyter Notebook and covers data loading from DICOM files, model design, training with early stopping, and generation of a Kaggle-compatible submission file.

---

## 📌 Problem Statement

Pneumonia is a lung infection that appears as opacities in chest radiographs.  
The objective is to classify each chest X-ray as:

- `0` → Normal  
- `1` → Pneumonia  

This is formulated as a **binary image classification** problem.

---

## 📂 Repository Contents

```
RSNA/
│
├── Training.ipynb      # Complete training & evaluation pipeline
├── best_model.pth     # Saved weights of best CNN (early stopping)
├── submission.csv     # Kaggle submission file (patientId, PredictionString)
└── README.md
```

⚠️ The dataset (DICOM images and labels CSV) is **not included** due to size and licensing constraints.

---

## 🗂 Dataset

- Source: RSNA Pneumonia Detection Challenge (Kaggle)
- Format: DICOM chest X-ray images
- Labels: Binary (Pneumonia / Normal)
- Loading: `pydicom` used to read pixel arrays
- Preprocessing:
  - Resizing
  - Normalization
  - Conversion to tensors
  - Train / validation split

---

## 🧠 Model Architecture (From Scratch)

Custom CNN implemented using `torch.nn`:

- Convolutional blocks with:
  - Conv2D
  - ReLU
  - Batch Normalization
  - Max Pooling
- Increasing feature depth: 16 → 32 → 64 → 128
- Adaptive Global Average Pooling (GAP)
- Dropout for regularization
- Fully Connected layer for 2-class output

No pretrained models or transfer learning are used.

---

## ⚙️ Training Details

- Framework: PyTorch
- Loss: CrossEntropyLoss
- Optimizer: Adam
- Device: CUDA (if available)
- Batch Size: GPU-friendly
- Epochs: Up to 51
- Early Stopping:
  - Patience = 5
  - Best model saved based on validation loss

The best performing model is automatically saved as:

```
best_model.pth
```

---

## 📈 Evaluation

During training, the notebook tracks:

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy

Early stopping prevents overfitting and ensures the best generalizing model is stored.

---

## 📤 Kaggle Submission

After training, the model is used to generate predictions for the test set:

- Softmax probability for Pneumonia class
- Formatted as `PredictionString`
- Saved as:

```
submission.csv
```

This file is directly compatible with Kaggle’s RSNA challenge submission format.

---

## 🚀 How to Run

1. Download the RSNA dataset from Kaggle.
2. Place DICOM images in a directory.
3. Update paths inside `Training.ipynb`.
4. Run all cells to:
   - Train the CNN
   - Save best weights
   - Generate `submission.csv`

---

## 🔬 Key Learning Outcomes

- Implemented CNN architecture from first principles
- Worked with real medical DICOM imaging data
- Built custom PyTorch Dataset with caching
- Applied early stopping for robust training
- Understood end-to-end ML competition workflow

---

## 👨‍💻 Author

**Aditya Gautam**  
Deep Learning | PyTorch | Medical Imaging | ML Systems  
Aspiring ML Engineer focused on core model building, not just fine-tuning.
