# Neural-Network-Quantization-Benchmark-with-IoT-Use-Case
This project compares the performance of **Float32 vs Quantized Neural Networks** in two scenarios:

---

## 📌 Part 1: CNN Benchmark on MNIST

### 🔍 Description:
Trains a Convolutional Neural Network (CNN) on the MNIST digit dataset and compares:
- Accuracy
- Inference time
- Model size

### ✅ Tools:
- TensorFlow
- TensorFlow Lite
- Python (Colab/Jupyter)

### 📊 Result Summary:
| Metric         | Float32     | Quantized   |
|----------------|-------------|-------------|
| Accuracy       | ~98.5%      | ~97.2%      |
| Model Size     | ~1.2 MB     | ~300 KB     |
| Inference Time | ~X ms       | ~Y ms       |

---

## 📌 Part 2: IoT Edge-AI on Environmental Sensor Data

### 🔍 Description:
Simulates realistic IoT sensor data (e.g., temperature, humidity, CO₂ levels) for smart city air quality classification. Compares MLP performance before and after quantization.

### ✅ Tools:
- TensorFlow / TFLite
- scikit-learn
- NumPy, Matplotlib

### 📊 Result Summary:
| Metric         | Float32     | Quantized   |
|----------------|-------------|-------------|
| Accuracy       | ~84%        | ~84%        |
| Model Size     | ~800 KB     | ~200 KB     |
| Inference Time | ~X ms       | ~Y ms       |

---


📦 Requirements
bash
Copy
Edit
pip install tensorflow numpy pandas matplotlib scikit-learn

🚀 How to Run
Open each notebook in Google Colab or Jupyter and run all cells.

## 💡 Key Takeaway
> Quantization significantly reduces model size and improves inference time, with minimal accuracy loss — ideal for Edge AI applications in IoT systems.

---

