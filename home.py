import streamlit as st
st.set_page_config(layout="wide")
from streamlit_extras.stylable_container import stylable_container
import pandas as pd
import numpy as np
import io
import cv2
from PIL import Image
import pickle
import tensorflow as tf
from tensorflow import keras
import time
from playsound import playsound

# Main heading
col1, col2 = st.columns([0.6, 10])

with col1:
    st.image("D:\Major_project\EchoHelm\logo.png", width=80)

with col2:
    with stylable_container(
        key='heading1',
        css_styles="""
        .eco-title {
            color: #4dfa81;
            font-size: 40px;
            margin-bottom: 10px;
        }
        """
    ):
        st.markdown('<h1 class="eco-title">EcoHelm</h1>', unsafe_allow_html=True)

# Different tabs
with stylable_container(
    key='heading2',
    css_styles='''
    div[data-testid="stMarkdownContainer"] > p{
        color: #c7d9cd; 
        font-size: 16px;
    }
    ''',
):
    tab_titles = ['Forest Fire Detection', 'Forest Fire Prediction', 'ForestHelp']
    tabs = st.tabs(tab_titles)    
 
# Fire Detection Tab
with tabs[0]:
    with stylable_container(
    key='heading3',
    css_styles="""
        .tab1 {
            color: #ff5959;
        }
        """
    ):
        st.markdown('<h2 class="tab1">🔍 Forest Fire Detection</h2>', unsafe_allow_html=True)


    col1, col2 = st.columns(2)

    with col1:
        with stylable_container(
        key='video1',
        css_styles='''
        [data-testid="element-container"] > video{
        width: 1500px;
        height: auto;
        }
        ''',
        ):
            # model = keras.models.load_model('model.h5')

            def predictImage(file_name):
                img1 = tf.keras.preprocessing.image.load_img(file_name, target_size=(150,150))
                Y = tf.keras.preprocessing.image.img_to_array(img1)
                X = np.expand_dims(Y, axis=0)
                val = model.predict(X)
                print(val)
                label = ""
                if val > 0.5:
                    label="😄 No Fire detected"
                else:
                    label = "🔥Fire detected"
                st.image(img1,caption = label)

            st.title("Forest Fire Detection by Input Image")
            image = st.file_uploader("Upload an image",['png','jpeg','jpg'])

            if image is not None:
                model = keras.models.load_model('model.h5')
                predictImage(image.name)

    with col2:
        # model = keras.models.load_model("D:\Major_project\EchoHelm\model.h5")

        # Predict frame
        def predict_frame(frame):
            img = cv2.resize(frame, (150, 150))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            prediction = model.predict(img)[0][0]
            return prediction

        # Streamlit UI
        st.title("Real-Time Fire Detection via Webcam")
        st.write("Allow access to your webcam and start real-time fire detection.")

        start = st.button("Start Webcam Detection", key="fire_prediction_start_btn")

        FRAME_WINDOW = st.image([])

        if start:
            cap = cv2.VideoCapture(0)  # Access webcam

            if not cap.isOpened():
                st.error("Could not access webcam. Please check camera permissions.")
            else:
                run = True
                st.write("Press 'Stop' to end detection.")
                stop = st.button("Stop")

                while run and not stop:
                    model = keras.models.load_model("D:\Major_project\EchoHelm\model.h5")

                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read from camera.")
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    prediction = predict_frame(frame_rgb)

                    if prediction < 0.5:
                        playsound(r'D:\Major_project\EchoHelm\beep.wav')

                    label = "🔥 Fire Detected" if prediction < 0.5 else "😄 No Fire Detected"
                    color = (255, 0, 0) if prediction < 0.5 else (0, 255, 0)
                    cv2.putText(frame_rgb, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                    FRAME_WINDOW.image(frame_rgb)
                    time.sleep(0.1)  # Add slight delay to avoid CPU overuse
                if stop:
                    FRAME_WINDOW = st.image([])
    
                cap.release()
                FRAME_WINDOW.empty()
                st.success("Webcam stopped.")
        
# Forest Fire Prediction Tab
with tabs[1]:
    with stylable_container(
    key='heading4',
    css_styles="""
        .tab2 {
            color: #ff5959;
        }
        """
    ):
        st.markdown('<h2 class="tab2">💭 Forest Fire Prediction</h2>', unsafe_allow_html=True)

    st.write('Enter Values of Month, Temperature (°C), Humidity (%), WindSpeed (km/h), and Rain (mm) to predict the likeliness of forest fire!')

    # Load the model
    with open("D:\Major_project\EchoHelm\model_prediction.pkl", 'rb') as file:
        loaded_model = pickle.load(file)

    # Input fields
    mon = st.text_input("Enter Month in Integer (1-12):", key="month_input")
    temp = st.text_input("Enter Temperature in Celsius:", key="temperature_input")
    hum = st.text_input("Enter Humidity Value:", key="humidity_input")
    ws = st.text_input('Enter Wind Speed in km/h:', key="wind_input")
    rn = st.text_input('Enter Rainfall in mm:', key="rain_input")

    b1 = st.button("Submit")

    if b1:
        error_flag = False

        # Validate each input field
        try:
            int_val1 = int(mon)
        except:
            st.error('Please enter a valid **integer** for Month.')
            error_flag = True

        try:
            f_val2 = float(temp)
        except:
            st.error('Please enter a valid **number** for Temperature.')
            error_flag = True

        try:
            f_val3 = float(hum)
        except:
            st.error('Please enter a valid **number** for Humidity.')
            error_flag = True

        try:
            f_val4 = float(ws)
        except:
            st.error('Please enter a valid **number** for Wind Speed.')
            error_flag = True

        try:
            f_val5 = float(rn)
        except:
            st.error('Please enter a valid **number** for Rainfall.')
            error_flag = True

        # Proceed with prediction only if all inputs are valid
        if not error_flag:
            data = np.array([int_val1, f_val2, f_val3, f_val4, f_val5]).reshape(1, -1)
            predictions = loaded_model.predict(data)

            st.success("Prediction of occurrence of forest fire:")
            st.write("🔥 Fire Likely" if predictions[0] == 1 else "😄 No Fire Likely")

 
# ForestHelp Tab
with tabs[2]:
    with stylable_container(
    key='heading5',
    css_styles="""
        .tab3 {
            color: #ff5959;
        }
        """
    ):
        st.markdown('<h2 class="tab3">📖 ForestHelp</h2>', unsafe_allow_html=True)

    st.write('Know more about flora and fauna!')
    import nltk
    import json
    import random
    nltk.download('punkt_tab')
    from nltk.stem import WordNetLemmatizer
    # from keras.models import load_model

    lemmatizer = WordNetLemmatizer()
    intents = json.loads(open('intents.json').read())

    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    # model = keras.models.load_model('chatbot_model.h5')

    def clean_up_sentence(sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words

    def bag_of_words (sentence):
        sentence_words = clean_up_sentence(sentence)
        bag = [0] * len(words)
        for w in sentence_words:
            for i, word in enumerate(words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class (sentence):
        bow = bag_of_words (sentence)
        model = keras.models.load_model('chatbot_model.h5')
        res = model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': classes [r[0]], 'probability': str(r[1])})
        return return_list

    def get_response(intents_list, intents_json):
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice (i['responses'])
                break
        return result

    def main():
        st.title('Get Your Queries Answered')
        message = st.text_input(label="Ask Anything:", key="input1")

        if message:
            message = message.lower()  # Convert to lowercase
            ints = predict_class(message)
            res = get_response(ints, intents)
            
            # Split at full stops and strip extra spaces
            sentences = [s.strip() for s in res.strip().split('.') if s.strip()]
            
            st.markdown("**Response:**")
            for sentence in sentences:
                st.markdown(f"- {sentence}.")
    main()