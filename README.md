# Brain Tumor Segmentation using U-Net

This project implements a deep learning pipeline for semantic segmentation of brain tumors in MRI images using the U-Net architecture. The goal is to accurately detect and segment tumor regions to assist in medical diagnostics.

## 📁 Dataset
The dataset used is the **LGG Segmentation Dataset** from Kaggle, consisting of T1-weighted MRI scans with corresponding ground-truth masks.

- Images: MRI scans of brain slices.
- Masks: Binary images highlighting tumor regions.
- Input shape: 256x256x3 (resized for training)

## 🧰 Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-image

## 🏗️ Model Architecture
The model is based on the **U-Net** architecture, ideal for biomedical image segmentation. It uses:
- Encoder–Decoder structure with skip connections
- Convolutional layers with ReLU activation
- Batch normalization
- MaxPooling and Conv2DTranspose for downsampling & upsampling

Custom loss and metrics:
- **Dice Loss**
- **IoU Coefficient**
- **Accuracy**

## 🧪 Training Details
- Optimizer: Adamax
- Loss Function: Dice Loss
- Epochs: 120
- Batch Size: 40
- Data Augmentation: Rotation, flipping, zoom, etc.

### 🔍 Training Performance (Highlights)
- **Validation Accuracy**: ~99.7%
- **Validation Dice Coefficient**: ~0.89
- **Validation IoU**: ~0.81

Training and validation metrics are visualized using Matplotlib.

## 📊 Visual Results
The project includes:
- Overlay plots of segmented masks on original images
- Training performance curves: Accuracy, Dice, IoU, and Loss

## 📦 File Structure
├── Brain_Tumor_Segmentation.ipynb

├── utils/ # Helper functions and generators

├── unet_model.py # U-Net architecture

├── train.py # Training script

├── data/ # Images and masks



## 🚀 How to Run
1. Clone the repository:
   bash
   git clone https://github.com/yourusername/brain-tumor-segmentation.git
   cd brain-tumor-segmentation

2.Install dependencies:

pip install -r requirements.txt

3.Train the model:

python train.py

🏁 Results & Conclusion
The U-Net model proved to be highly effective in segmenting brain tumors, making it a suitable approach for assisting in clinical diagnostics. This project demonstrates the potential of deep learning in the medical imaging domain.

Acknowledgments
Kaggle LGG Segmentation Dataset
U-Net: Convolutional Networks for Biomedical Image Segmentation 






