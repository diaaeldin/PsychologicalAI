import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf  # Import TensorFlow here

# Define a dictionary to map numerical class labels to text labels
label_map = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Pleasant and surprise',
    6: 'Sad',
}

def preprocess_audio(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Streamlit app
def main():
    st.title("Emotion Prediction from Audio")

    # File upload
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        # Load your pre-trained model here
        model = load_model('/home/mostafatarek/Documents/Toronto emotional speech set (TESS)/best_model.h5')
        
        # Preprocess the uploaded audio file
        sample_to_predict = preprocess_audio(uploaded_file)

        # Reshape the data to match the model's input shape
        sample_to_predict = np.expand_dims(sample_to_predict, axis=0)

        # Make predictions
        prediction = model.predict(sample_to_predict,verbose=0)

        # If your predictions are one-hot encoded, you can convert them to class labels
        predicted_label = np.argmax(prediction, axis=1)

        # Look up the text label based on the numerical label
        predicted_text_label = label_map.get(predicted_label[0], 'Unknown')

        st.write(f"Predicted emotion: {predicted_text_label}")

if __name__ == "__main__":
    main()
