import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model("pneumonia_cnn_model.h5")

# Prediction function
def predict_pneumonia(img: Image.Image):
    img = img.resize((150,150))
    arr = np.array(img)/255.0
    arr = np.expand_dims(arr, 0)
    pred = model.predict(arr)[0][0]
    return "🩺 Pneumonia Detected" if pred > 0.5 else "✅ Normal Lungs"

# Gradio app interface
demo = gr.Interface(
    fn=predict_pneumonia,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Chest X-Ray Pneumonia Detector",
    description="Upload a chest X-ray image to check if pneumonia is present."
)

if __name__ == "__main__":
    demo.launch()