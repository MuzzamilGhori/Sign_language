# import cv2
# import mediapipe as mp
# import numpy as np
import pickle
import os
import random
import time
import tkinter as tk
from tkinter import simpledialog
from sign_utils import predict_sign_from_image, get_sign_image_paths


# Load trained model
model_dict = pickle.load(open(r'C:\Users\DELL\Desktop\Sign-Language-Interpreter-main\Sign-Language-Interpreter-main\ML_Models\RFC_model.p', 'rb'))
model = model_dict['model']

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

# Labels
labels_dict = {
    0: 'HELLO', 1: 'yes', 2: 'no', 3: 'bye', 4: 'thank you', 5: 'welcome',
    6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
    23: 'X', 24: 'Y', 25: 'Z'
}

# Tkinter setup for input popup
root = tk.Tk()
root.withdraw()  # Hide main window

# Function to display sign images from dataset
def show_sign_images(words):
    for word in words:
        folder_path = f"Dataset/{word.lower()}/"

        if not os.path.exists(folder_path):
            print(f"No folder for '{word}' in dataset")
            continue

        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
        if not image_files:
            print(f"No images found for '{word}'")
            continue

        random_image = random.choice(image_files)
        image_path = os.path.join(folder_path, random_image)

        img = cv2.imread(image_path)
        if img is not None:
            cv2.imshow('Sign Display', img)
            cv2.waitKey(1500)  # Show for 1.5 seconds
        else:
            print(f"Failed to load image for '{word}'")

    # Safely close the window only if it exists
    try:
        if cv2.getWindowProperty('Sign Display', cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow('Sign Display')
    except cv2.error as e:
        print(f"Error closing window: {e}")

# Set initial mode
mode = "sign_to_text"

# Start loop
while True:
    if mode == "sign_to_text":
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        data_aux = []
        x_ = []
        y_ = []
        predicted_character = ""

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Bounding box
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            if len(data_aux) == model.n_features_in_:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
            else:
                predicted_character = "Unknown"

            # Draw box and prediction
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow('Sign Language Interpreter', frame)

    # Key inputs
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    elif key == ord('t'):
        mode = "text_to_sign"
        user_input = simpledialog.askstring("Text to Sign", "Enter text:")
        if user_input:
            words = user_input.upper().split()
            show_sign_images(words)
        mode = "sign_to_text"

# Cleanup
cap.release()
cv2.destroyAllWindows()
