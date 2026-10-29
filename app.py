import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Aapla saved model load kara
model = tf.keras.models.load_model('models/waste_classifier_model.keras')

# Tumchya dataset madhil folders che naave alphabetical order madhe ithe lihava
# Udaharanarth (tumchya folder nusar change karu sakta):
class_names = ['glass', 'metal', 'paper', 'plastic']

def predict_waste(image):
    # Image la model sathi required size madhe convert kara (150x150)
    image = image.resize((150, 150))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    return f"Prediction: {predicted_class.upper()} (Confidence: {confidence:.2f}%)"

# 2. Gradio Interface Design
demo = gr.Interface(
    fn=predict_waste,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Waste Image Classification System",
    description="Upload an image of waste (Paper, Plastic, Glass, Metal) to classify it using Deep Learning."
)

if __name__ == "__main__":
    demo.launch()