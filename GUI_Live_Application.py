import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import tkinter as tk
from tkinter import ttk

# Load pre-trained model
model_dict = pickle.load(open('./ML_Models/RFC_model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Define labels
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
    10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T',
    19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y', 24: 'FUCK YOU'
}

# Variables for storing characters and sentences
current_sentence = ""
record_sentence = False

# Initialize text-to-speech engine
engine = pyttsx3.init()

def speak_sentence():
    global current_sentence
    engine.say(current_sentence)
    engine.runAndWait()

def clear_sentence():
    global current_sentence
    current_sentence = ""
    text_sentence.delete("1.0", tk.END)

# Create Tkinter window
root = tk.Tk()
root.title("Hand Gesture Recognition")
root.geometry("1200x800")  # Initial size of the window

# Configure grid layout
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=7)
root.grid_columnconfigure(1, weight=3)

# Create frame for video display
video_frame = ttk.Frame(root)
video_frame.grid(row=0, column=0, sticky="nsew")
canvas = tk.Canvas(video_frame)
canvas.pack(fill="both", expand=True)

# Create frame for buttons and sentence display
control_frame = ttk.Frame(root)
control_frame.grid(row=0, column=1, sticky="nsew")
control_frame.grid_rowconfigure(0, weight=1)
control_frame.grid_rowconfigure(1, weight=0)
control_frame.grid_rowconfigure(2, weight=0)

# Create text box for current sentence display
text_sentence = tk.Text(control_frame, wrap="word", font=("Helvetica", 12))
text_sentence.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
button_clear = ttk.Button(control_frame, text="Clear", command=clear_sentence)
button_clear.grid(row=1, column=0, pady=5, padx=10, sticky="ew")
button_speak = ttk.Button(control_frame, text="Speak", command=speak_sentence)
button_speak.grid(row=2, column=0, pady=5, padx=10, sticky="ew")

def update_video():
    global current_sentence, record_sentence

    ret, frame = cap.read()
    if not ret:
        return

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            data_aux = []
            x_ = []
            y_ = []

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            # Get predictions
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Display predicted character
            text = f"{predicted_character}"
            cv2.rectangle(frame, (int(min(x_) * W) - 10, int(min(y_) * H) - 10), (int(max(x_) * W) - 10, int(max(y_) * H) - 10), (0, 0, 0), 4)
            cv2.putText(frame, text, (int(min(x_) * W) - 10, int(min(y_) * H) - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            # Record sentence if Enter key is pressed
            if record_sentence:
                current_sentence += predicted_character
                text_sentence.insert(tk.END, predicted_character)
                record_sentence = False

    # Convert frame to RGB format and display in tkinter window
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (canvas.winfo_width(), canvas.winfo_height()))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Convert the image to Tkinter format
    img_tk = tk.PhotoImage(data=cv2.imencode('.png', img)[1].tobytes())
    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.imgtk = img_tk
    canvas.after(10, update_video)

def on_key_press(event):
    global record_sentence, current_sentence
    if event.keysym == "Return":  # Enter key
        record_sentence = True
    elif event.keysym == "space":  # Space key
        current_sentence += " "
        text_sentence.insert(tk.END, " ")
    elif event.keysym == "Escape":  # Escape key
        root.quit()

# Bind key press events
root.bind("<KeyPress>", on_key_press)

update_video()
root.mainloop()
