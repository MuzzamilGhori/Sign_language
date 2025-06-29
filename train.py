import cv2
import mediapipe as mp
import csv
import os
import numpy as np

# Set the gesture label for this recording session
gesture_label = input("Enter label for this gesture (e.g., 'thank_you_both'): ")

# Create CSV file if it doesn't exist
filename = 'two_hand_gesture_data.csv'
file_exists = os.path.isfile(filename)

# Open CSV file for appending
csvfile = open(filename, 'a', newline='')
csvwriter = csv.writer(csvfile)

# Write header if file is new
if not file_exists:
    header = []
    for hand_num in range(2):
        for lm_num in range(21):
            header.extend([f'x{hand_num}_{lm_num}', f'y{hand_num}_{lm_num}'])
    header.append('label')
    csvwriter.writerow(header)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)
print("Press 's' to save a frame, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    landmarks_all = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            normalized_landmarks = []
            for lm in hand_landmarks.landmark:
                normalized_landmarks.append(lm.x - min(x_))
                normalized_landmarks.append(lm.y - min(y_))

            landmarks_all.append(normalized_landmarks)

            # Draw landmarks on frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Wait for key input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        if len(landmarks_all) == 2:
            row = landmarks_all[0] + landmarks_all[1] + [gesture_label]
            csvwriter.writerow(row)
            print(f"Saved gesture: {gesture_label}")
        else:
            print("Please show TWO hands to save the sample.")
    elif key == ord('q'):
        break

    cv2.putText(frame, f'Label: {gesture_label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow('Two-Hand Data Collector', frame)

# Clean up
cap.release()
csvfile.close()
cv2.destroyAllWindows()
