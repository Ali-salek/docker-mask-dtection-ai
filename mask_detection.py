import cv2
import os
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title='Face Mask Detector' , page_icon='😷' , layout='centered', initial_sidebar_state='expanded')

def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass 

@st.cache_resource
def load_keras_model():
    return load_model('model/trained_model.keras')

def process_mask_image(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    mask_model = load_keras_model()

    image = cv2.imread(image_path)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]

        try:
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (224, 224))
            face_normalized = face_resized / 255.0
            face_ready = np.expand_dims(face_normalized, axis=0)

            prediction = mask_model.predict(face_ready, verbose=0)[0][0]

            if prediction < 0.5:
                label = "Mask"
                color = (0, 255, 0)
                prob = (1 - prediction) * 100                

            else:
                label = "No Mask"
                color = (0, 0, 255)
                prob = prediction * 100

                
            text = f"{label}: {prob:.1f}%"
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
        except Exception as e:
            continue

    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return RGB_img

def mask_detection():
    local_css("css/style.css")
    st.markdown('<h1 align="center">😷 Face Mask Detection</h1>' , unsafe_allow_html=True)
    
    activities = ["Image" , "Webcam"]
    st.sidebar.markdown("### Choose input method")
    choice = st.sidebar.selectbox("Options:" , activities)

    if choice == 'Image':
        st.markdown('<h2 align="center">Detection on Image</h2>' , unsafe_allow_html=True)
        image_file = st.file_uploader("Upload your image here", type=['jpg' , 'png' , 'jpeg'])
        
        if image_file is not None:
            our_image = Image.open(image_file)
            
            os.makedirs('./images', exist_ok=True)
            image_path = './images/out.jpg'
            our_image.save(image_path) 
            
            st.image(image_file, caption='Uploaded Image', use_container_width=True)
            st.markdown('<h3 align="center">Image uploaded successfully!</h3>' , unsafe_allow_html=True)
            
            if st.button('Process'):
                with st.spinner('Detecting mask... please wait. ⏳'):
                    
                    result_img = process_mask_image(image_path)
                    
                    if result_img is not None:
                        st.image(result_img , use_container_width=True)
                        st.balloons() 
                        st.success("Processing completed successfully.")
                    else:
                        st.error("Error reading image!")

    if choice == 'Webcam':   
        st.markdown('<h2 align="center"> Detection on Webcam</h2>', unsafe_allow_html=True)
        st.markdown('<h3 align="center">This feature will be available soon!</h3>', unsafe_allow_html= True)


if __name__ == "__main__":
    mask_detection()