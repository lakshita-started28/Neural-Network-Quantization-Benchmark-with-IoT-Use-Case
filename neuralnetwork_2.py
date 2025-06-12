# Project: Environmental Monitoring with Realistic Sensor Data
# Goal: Compare Float32 vs Quantized NN (MLP) for Smart City IoT Use-case

# ✅ Step 1: Import Libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ✅ Step 2: Simulate Noisy Environmental Dataset (more realistic)
np.random.seed(42)
data_size = 5000
temp = np.random.normal(loc=25, scale=7, size=data_size)
humidity = np.random.normal(loc=55, scale=15, size=data_size)
co2 = np.random.normal(loc=420, scale=80, size=data_size)
noise = np.random.normal(0, 5, size=data_size)

# Label: 0 = Safe, 1 = Polluted (make it more complex)
labels = ((temp + noise > 32) | (humidity < 40) | (co2 > 500)).astype(int)

data = pd.DataFrame({'temp': temp, 'humidity': humidity, 'co2': co2, 'label': labels})

# ✅ Step 3: Preprocess Data
X = data[['temp', 'humidity', 'co2']].values
y = data['label'].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 4: Build & Train Float32 Neural Network (deeper model)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_split=0.1)

# ✅ Step 5: Evaluate Float32 Model
float_acc = model.evaluate(X_test, y_test, verbose=0)[1]

# ✅ Step 6: Save & Convert to Quantized TFLite Model
model.save("float_model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("quantized_model.tflite", "wb") as f:
    f.write(tflite_model)

# ✅ Step 7: Run Inference with TFLite Interpreter
def evaluate_tflite_model():
    interpreter = tf.lite.Interpreter(model_path='quantized_model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    correct = 0
    for i in range(len(X_test)):
        input_data = np.array([X_test[i]], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        pred = 1 if output[0][0] > 0.5 else 0
        if pred == y_test[i]:
            correct += 1
    return correct / len(X_test)

quant_acc = evaluate_tflite_model()

# ✅ Step 8: Compare Size & Inference Time
float_size = os.path.getsize("float_model.h5") / 1024
quant_size = os.path.getsize("quantized_model.tflite") / 1024

interpreter = tf.lite.Interpreter(model_path='quantized_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

start = time.time()
for i in range(100):
    input_data = np.array([X_test[i]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    _ = interpreter.get_tensor(output_details[0]['index'])
end = time.time()
quant_time = (end - start) / 100

# ✅ Step 9: Plot Results
labels = ['Float32 Accuracy', 'Quantized Accuracy']
values = [float_acc, quant_acc]
plt.figure(figsize=(6, 4))
bars = plt.bar(labels, values, color=['blue', 'green'])
plt.ylim([0.70, 1.0])
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + 0.1, yval + 0.01, f'{yval:.3f}')
plt.tight_layout()
plt.show()

# ✅ Step 10: Print Summary
print(f"Float32 Accuracy: {float_acc:.4f}")
print(f"Quantized Accuracy: {quant_acc:.4f}")
print(f"Float32 Size: {float_size:.2f} KB")
print(f"Quantized Size: {quant_size:.2f} KB")
print(f"Quantized Inference Time: {quant_time*1000:.2f} ms")

# Project: Environmental Monitoring with Realistic Sensor Data
# Goal: Compare Float32 vs Quantized NN (MLP) for Smart City IoT Use-case

# ✅ Step 1: Import Libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ✅ Step 2: Simulate Noisy Environmental Dataset (more realistic)
np.random.seed(42)
data_size = 5000
temp = np.random.normal(loc=25, scale=7, size=data_size)
humidity = np.random.normal(loc=55, scale=15, size=data_size)
co2 = np.random.normal(loc=420, scale=80, size=data_size)
noise = np.random.normal(0, 5, size=data_size)

# Label: 0 = Safe, 1 = Polluted (make it more complex)
labels = ((temp + noise > 32) | (humidity < 40) | (co2 > 500)).astype(int)

data = pd.DataFrame({'temp': temp, 'humidity': humidity, 'co2': co2, 'label': labels})

# ✅ Step 3: Preprocess Data
X = data[['temp', 'humidity', 'co2']].values
y = data['label'].values
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 4: Build & Train Float32 Neural Network (deeper model)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_split=0.1)

# ✅ Step 5: Evaluate Float32 Model
float_acc = model.evaluate(X_test, y_test, verbose=0)[1]

# ✅ Step 6: Save & Convert to Quantized TFLite Model
model.save("float_model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("quantized_model.tflite", "wb") as f:
    f.write(tflite_model)

# ✅ Step 7: Run Inference with TFLite Interpreter
def evaluate_tflite_model():
    interpreter = tf.lite.Interpreter(model_path='quantized_model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    correct = 0
    for i in range(len(X_test)):
        input_data = np.array([X_test[i]], dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        pred = 1 if output[0][0] > 0.5 else 0
        if pred == y_test[i]:
            correct += 1
    return correct / len(X_test)

quant_acc = evaluate_tflite_model()

# ✅ Step 8: Compare Size & Inference Time
float_size = os.path.getsize("float_model.h5") / 1024
quant_size = os.path.getsize("quantized_model.tflite") / 1024

interpreter = tf.lite.Interpreter(model_path='quantized_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

start = time.time()
for i in range(100):
    input_data = np.array([X_test[i]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    _ = interpreter.get_tensor(output_details[0]['index'])
end = time.time()
quant_time = (end - start) / 100

# ✅ Step 9: Plot Results
labels = ['Float32 Accuracy', 'Quantized Accuracy']
values = [float_acc, quant_acc]
plt.figure(figsize=(6, 4))
bars = plt.bar(labels, values, color=['blue', 'green'])
plt.ylim([0.70, 1.0])
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + 0.1, yval + 0.01, f'{yval:.3f}')
plt.tight_layout()
plt.show()

# ✅ Step 10: Print Summary
print(f"Float32 Accuracy: {float_acc:.4f}")
print(f"Quantized Accuracy: {quant_acc:.4f}")
print(f"Float32 Size: {float_size:.2f} KB")
print(f"Quantized Size: {quant_size:.2f} KB")
print(f"Quantized Inference Time: {quant_time*1000:.2f} ms")

