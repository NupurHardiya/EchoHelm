# 🌿 EcoHelm - Ecological Conservation and Sustainability

**EcoHelm** is a multi-functional web application developed using **Streamlit** that serves three major purposes:
1. 🔍 Detect forest fires in real-time using webcam or uploaded images with deep learning.
2. 💭 Predict the likelihood of forest fire occurrences using environmental parameters.
3. 📖 Provide chatbot assistance to users about forest-related knowledge using NLP.

---

## 📁 Features

### 🔍 1. Forest Fire Detection
- Upload an image to detect fire using a trained CNN model (`model.h5`).
- Real-time webcam detection with alerts (sound + visual).
- Uses OpenCV for webcam access and TensorFlow/Keras for predictions.

### 💭 2. Forest Fire Prediction
- Input values such as **Month**, **Temperature**, **Humidity**, **Wind Speed**, and **Rainfall**.
- Predicts the chance of a fire using a pre-trained ML model (`model_prediction.pkl`).
- Outputs intuitive results like "🔥 Fire Likely" or "😄 No Fire Likely".

### 📖 3. ForestHelp Chatbot
- NLP-based assistant for forest-related queries.
- Uses an intent classification model (`chatbot_model.h5`) and NLTK for preprocessing.
- Trained with custom intents (`intents.json`).

---

## 🛠️ Requirements

Ensure the following Python packages are installed:

```bash
pip install streamlit tensorflow keras opencv-python-headless numpy pandas playsound nltk streamlit-extras pillow

EcoHelm/
├── model.h5                     # CNN model for image/webcam fire detection
├── model_prediction.pkl         # ML model for fire prediction
├── chatbot_model.h5             # NLP model for chatbot
├── intents.json                 # Intents for chatbot
├── words.pkl                    # Tokenized vocabulary
├── classes.pkl                  # Output classes
├── beep.wav                     # Sound alert for fire
├── logo.png                     # Logo image
├── home.py                       # Main Streamlit application

