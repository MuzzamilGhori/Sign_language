# live_app_streamlit.py

import streamlit as st
import cv2
from app import sign_utils  # adjust this import to your actual structure

def main():
    st.title("Sign Language Interpreter")

    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.warning("Camera not accessible")
            break

        # Process frame (you might call your classifier here)
        processed_frame = frame  # replace with actual processing

        FRAME_WINDOW.image(processed_frame, channels='BGR')

    camera.release()

if __name__ == "__main__":
    main()
