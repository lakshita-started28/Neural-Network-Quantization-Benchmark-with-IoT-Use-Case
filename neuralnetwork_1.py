# Project: Model Quantization Benchmark (MNIST Dataset)
# Goal: Compare Float32 vs Quantized CNN on image data (MNIST)

# ✅ Step 1: Import Libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# ✅ Step 2: Load & Preprocess MNIST Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# ✅ Step 3: Build & Train CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# ✅ Step 4: Evaluate Float32 Model
float_acc = model.evaluate(x_test, y_test, verbose=0)[1]

# ✅ Step 5: Save & Convert to Quantized TFLite Model
model.save("mnist_model_float32.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("mnist_model_quant.tflite", "wb") as f:
    f.write(tflite_model)

# ✅ Step 6: Evaluate Quantized Model
interpreter = tf.lite.Interpreter(model_path='mnist_model_quant.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def evaluate_quant_model():
    correct = 0
    for i in range(1000):
        input_data = np.array([x_test[i]], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        if np.argmax(output) == y_test[i]:
            correct += 1
    return correct / 1000

quant_acc = evaluate_quant_model()

# ✅ Step 7: Compare Size & Inference Time
float_size = os.path.getsize("mnist_model_float32.h5") / 1024
quant_size = os.path.getsize("mnist_model_quant.tflite") / 1024

start = time.time()
for i in range(100):
    input_data = np.array([x_test[i]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    _ = interpreter.get_tensor(output_details[0]['index'])
end = time.time()
quant_time = (end - start) / 100

# ✅ Step 8: Plot Accuracy Comparison
labels = ['Float32 Accuracy', 'Quantized Accuracy']
values = [float_acc, quant_acc]
plt.figure(figsize=(6, 4))
bars = plt.bar(labels, values, color=['purple', 'orange'])
plt.ylim([0.90, 1.0])
plt.ylabel("Accuracy")
plt.title("MNIST Model Accuracy Comparison")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + 0.1, yval + 0.005, f'{yval:.3f}')
plt.tight_layout()
plt.show()

# ✅ Step 9: Print Summary
print(f"Float32 Accuracy: {float_acc:.4f}")
print(f"Quantized Accuracy: {quant_acc:.4f}")
print(f"Float32 Size: {float_size:.2f} KB")
print(f"Quantized Size: {quant_size:.2f} KB")
print(f"Quantized Inference Time: {quant_time*1000:.2f} ms")
