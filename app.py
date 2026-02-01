import gradio as gr
import numpy as np
import joblib
import pandas as pd

# Load trained model
model = joblib.load('random_forest_model.joblib')

def predict(*features):
    features_array = np.array(features).reshape(1, -1)
    
    prediction = model.predict(features_array)[0]
    probability = model.predict_proba(features_array)[0]
    
    result = f"Prediction: Class {prediction}\n"
    result += f"Probabilities: Class 0: {probability[0]:.2%}, Class 1: {probability[1]:.2%}"
    
    return result

inputs = []
for i in range(10):
    inputs.append(
        gr.Slider(
            minimum=-5, 
            maximum=5, 
            value=0, 
            label=f"Feature {i+1}"
        )
    )

interface = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=gr.Textbox(label="Prediction Results"),
    title="Custom Trained Classifier",
    description="Adjust the 10 feature sliders to get predictions from your custom-trained model!",
    theme="grass"
)

if __name__ == "__main__":
    interface.launch()