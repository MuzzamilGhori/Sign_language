import cv2
import mediapipe as mp
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Function to load images and labels
def load_data(data_dir):
    images = []
    labels = []
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            label = int(folder_name)
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img_rgb)
                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    feature_vector = []
                    for lm in hand_landmarks.landmark:
                        feature_vector.extend([lm.x, lm.y, lm.z])
                    images.append(feature_vector)
                    labels.append(label)
    return np.array(images), np.array(labels)

data_dir = 'Data'
images, labels = load_data(data_dir)

# Normalize features
images = np.array(images)
labels = tf.keras.utils.to_categorical(labels, num_classes=24)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

def augment_data(images):
    augmented_images = []
    for img in images:
        noise = np.random.normal(0, 0.01, img.shape)
        img_noisy = img + noise
        augmented_images.append(img_noisy)
    return np.array(augmented_images)

# Augment the training data
X_train_aug = augment_data(X_train)
X_train = np.concatenate((X_train, X_train_aug), axis=0)
y_train = np.concatenate((y_train, y_train), axis=0)

# Reshape data to fit the model input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

# Define the model
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(2),
    Conv1D(128, 3, activation='relu'),
    MaxPooling1D(2),
    TimeDistributed(Flatten()),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(24, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Track highest validation accuracy
class HighestValAccuracy(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        current_val_acc = logs.get('val_accuracy')
        if not hasattr(self, 'highest_val_acc'):
            self.highest_val_acc = 0
        if current_val_acc > self.highest_val_acc:
            self.highest_val_acc = current_val_acc

highest_val_acc_callback = HighestValAccuracy()

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[highest_val_acc_callback])

# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Highest validation accuracy during training: {highest_val_acc_callback.highest_val_acc * 100:.2f}%")

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))

plt.suptitle('MEDIAPIPE on Hybrid CNN & RNN Model', fontsize=20)

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.figtext(0.5, 0.01, f'Validation accuracy: {highest_val_acc_callback.highest_val_acc * 100:.2f}%', ha='center', fontsize=16)

plt.show()

model.save('DL_Models/Mediapipe+CNN+RNN_Model.h5')

import pickle
with open('DL_Models/Mediapipe+CNN+RNN_Hisory.pkl', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)