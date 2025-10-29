# 🩺 Chest X-Ray Pneumonia Detection using CNN

This project uses a **Convolutional Neural Network (CNN)** to detect **Pneumonia** from **Chest X-ray images**.  
The model is trained in **Google Colab**, deployed with a **Gradio user interface**, and hosted online on **Hugging Face Spaces**.

---

## 👩‍💻 Developed by
**Yasaswi Sri Satya Meenakshi Kathinokkula**

---

## 🧠 Project Overview
- **Goal:** Detect pneumonia automatically from chest X-ray images using deep learning.  
- **Technology:** CNN (Convolutional Neural Networks)  
- **Platform:** Google Colab  
- **Deployment:** Hugging Face (Gradio App Interface)

---

## 🧩 Dataset
**Dataset Name:** Chest X-Ray Images (Pneumonia)  
**Source:** [Kaggle - Paul Timothy Mooney](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

The dataset contains:
- 🫁 **Normal** X-ray images  
- 🫁 **Pneumonia** X-ray images

---

## ⚙️ Steps Followed

### **Step 1 — Load the Dataset**
- Downloaded dataset from Kaggle  
- Extracted the zip file  
- Mounted Google Drive to Colab for access  

```python
from google.colab import drive
drive.mount('/content/drive')

---

Step 2 — Import Libraries

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

---

Step 3 — Data Preprocessing

Used ImageDataGenerator to normalize and augment the data for better performance.

train_datagen = ImageDataGenerator(rescale=1./255)
train_data = train_datagen.flow_from_directory(
    '/content/chest_xray/train',
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)


---

Step 4 — CNN Model Building

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


---

Step 5 — Compile and Train the Model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, validation_data=val_data, epochs=10)


---

Step 6 — Save Model to Google Drive

model.save('/content/pneumonia_cnn_model.h5')

from google.colab import drive
drive.mount('/content/drive')

!cp /content/pneumonia_cnn_model.h5 /content/drive/MyDrive/

🔹 Model file is stored in Google Drive as:

pneumonia_cnn_model.h5
https://drive.google.com/file/d/1unuMAODEuZrVjr_VBYZJFqc1qNme4sAE/view?usp=drive_link
---

Step 7 — Deploy Using Gradio (app.py)

app.py file used to create a simple upload interface for users.

import gradio as gr
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model("pneumonia_cnn_model.h5")

def predict(img):
    img = image.load_img(img, target_size=(150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    prediction = model.predict(img)
    if prediction[0][0] > 0.5:
        return "🩺 Pneumonia Detected"
    else:
        return "✅ Normal Lungs"

iface = gr.Interface(fn=predict, inputs="image", outputs="text", title="Chest X-Ray Pneumonia Detector")
iface.launch()


---

Step 8 — Create requirements.txt

This file helps Hugging Face install necessary libraries automatically.

gradio
tensorflow-cpu
numpy
Pillow


---

Step 9 — Upload to Hugging Face Spaces

1. Go to https://huggingface.co/spaces


2. Create a new Space → Choose Gradio


3. Upload:

app.py

requirements.txt

pneumonia_cnn_model.h5



4. Wait for it to say Running (green) ✅


5. Click “View App” → Your app link will open.




---

🌐 Live Demo Link

👉 Click here to open Pneumonia Detector App
https://huggingface.co/spaces/yasaswik/pneumonia-detector

---

💾 Model File

The trained model is too large for GitHub (over 25MB).
You can download it from Google Drive here:
👉 Google Drive Model Link
https://drive.google.com/file/d/1unuMAODEuZrVjr_VBYZJFqc1qNme4sAE/view?usp=drive_link

---

🖼️ Output Example

Input: Chest X-ray image

Output: “Pneumonia Detected” or “Normal Lungs”



---

📚 Technologies Used

Tool	Purpose

Google Colab	Model training
TensorFlow / Keras	Deep learning
Gradio	Web interface
Hugging Face Spaces	Online hosting
Kaggle	Dataset source
