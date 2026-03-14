import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

DATASET_PATH = "dataset"

mp_hands = mp.solutions.hands.Hands(static_image_mode=True)

X = []
y = []

for letter in os.listdir(DATASET_PATH):

    folder = os.path.join(DATASET_PATH, letter)

    for img_name in os.listdir(folder):

        path = os.path.join(folder, img_name)

        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = mp_hands.process(img_rgb)

        if results.multi_hand_landmarks:

            landmarks = []

            for lm in results.multi_hand_landmarks[0].landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            X.append(landmarks)
            y.append(letter)

print("Training model...")

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model saved as model.pkl")