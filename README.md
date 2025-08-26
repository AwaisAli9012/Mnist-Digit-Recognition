# MNIST Digit Classification with CNN

A Convolutional Neural Network (CNN) model to classify handwritten digits (0-9) from the MNIST dataset using PyTorch.

## 📊 Dataset
- **Dataset**: MNIST
- **Images**: 28x28 grayscale
- **Classes**: 10 (digits 0–9)
- **Training Samples**: 60,000
- **Testing Samples**: 10,000
- **Normalization**: Mean = 0.1307, Std = 0.3081

## 🛠️ Model Architecture
### CNN Structure
```python
Input → [Conv2d → ReLU → MaxPool2d] × 3 → Flatten → Linear → ReLU → Dropout → Linear → Output
🧠 Training Details
Loss Function: Cross-Entropy Loss
Optimizer: Adam (learning rate = 0.001)
Batch Size: 64
Epochs: 5
Device: GPU (CUDA) if available, else CPU
📈 Results
After 5 epochs:

Training Accuracy: ~99%
Test Accuracy: ~98–99%
The model achieves high accuracy with minimal overfitting, thanks to dropout and proper regularization.

🖼️ Sample Predictions
The show_predictions() function displays:

True vs Predicted labels
Visualized test images with predictions
📦 Requirements
Python 3.x
PyTorch
torchvision
matplotlib
numpy

🔍 Key Features
Data normalization and augmentation
Efficient CNN design with pooling and dropout
GPU acceleration support
Real-time prediction visualization
Clean training and evaluation loop
📄 License
This project is licensed under the MIT License.



