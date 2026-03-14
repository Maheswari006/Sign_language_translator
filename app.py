from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import base64
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

mp_draw = mp.solutions.drawing_utils

current_text = ""

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    global current_text

    data = request.json["image"]

    # Decode image
    image_data = base64.b64decode(data.split(",")[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand skeleton
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

        # Extract landmarks
        landmarks = []

        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.append(lm.x)
            landmarks.append(lm.y)

        prediction = model.predict([landmarks])[0]

        print("Prediction:", prediction)

        current_text += prediction

        return jsonify({
            "prediction": prediction,
            "text": current_text
        })

    else:
        return jsonify({
            "prediction": "No hand detected",
            "text": current_text
        })


@app.route("/clear", methods=["POST"])
def clear():

    global current_text
    current_text = ""

    return jsonify({"text": current_text})


if __name__ == "__main__":
    app.run(debug=True)