#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
from PIL import Image
import numpy as np
import cv2
from keras.models import load_model
import os

model = load_model('./model_resnet.h5')

def face_to_bmi(image):
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    cascPath = "./haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)
    faces = face_cascade.detectMultiScale(img_array, 1.2, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_img = img_array[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (224, 224))
        face_img = np.expand_dims(face_img, axis=0)       
        bmi_pred = model.predict(face_img)
        bmi_pred = round(float(bmi_pred[0]),2)
        cv2.putText(img_array, f"BMI: {bmi_pred}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))

st.title("Face to BMI")
st.sidebar.title("Face to BMI")
upload_option = st.sidebar.radio("Select option for predicting:", ["Upload Image For Prediction", "Real Time Webcam Prediction"])
image_placeholder = st.empty()

if upload_option == "Upload Image For Prediction":
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        #st.write("Predicting...")
        pred_img = face_to_bmi(img)
        image_placeholder.image(pred_img, caption='Processed Image with Face Detection and BMI Prediction.', use_column_width=True)

elif upload_option == "Real Time Webcam Prediction":
    image_placeholder.empty()
    st.write("Real Time Webcam Prediction.")
    image_placeholder.empty()
    cascPath = "./haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)
    video_capture = cv2.VideoCapture(0)
    stop_button_pressed = st.button("Stop")
    while True:
        ret, frame = video_capture.read()
        if not ret:
            st.write("The capture has ended.")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(
            frame, 
            scaleFactor=1.2, 
            minNeighbors=5, 
            minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face = frame[y:y+h, x:x+w]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = np.expand_dims(face, axis=0)
            bmi_pred = model.predict(face)
            bmi = 'BMI: '+ str(round(bmi_pred[0][0], 2))
            cv2.putText(frame, bmi, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        image_placeholder.image(frame, channels='RGB')
        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
            break
    video_capture.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

