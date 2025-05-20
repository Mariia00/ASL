# # file: sign_language_recognition.py

import os
import pickle
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

from app.asl_model.gemini_api import get_gemini_response

reset_flag = False

base_dir = os.path.dirname(os.path.abspath(__file__))

# Побудувати шлях до моделі
model_path = os.path.join(base_dir, '..', '..', 'models', 'model1.p')

# Завантажити модель
with open(model_path, 'rb') as f:
    model_dict = pickle.load(f)

model = model_dict['model']

# Mediapipe hands module setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'SPACE', 27: 'START', 28: 'SEND'
}

sentence = ""
last_char = ""
last_time = time.time()
recognition_active = False
LETTER_DELAY = 10.0


def get_sentence():
    return sentence


def reset_sentence():
    global sentence, reset_flag
    sentence = ""
    reset_flag = True


def predict_character(data_aux):
    prediction = model.predict([np.asarray(data_aux)])
    return prediction[0]


def speak_text(text):
    def speak():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except RuntimeError as e:
            print(f"[TTS Error] {e}")

    threading.Thread(target=speak, daemon=True).start()


def make_square(image):
    h, w = image.shape[:2]
    size = max(h, w)
    delta_w = size - w
    delta_h = size - h
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2
    color = [0, 0, 0]
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def process_frame(frame):
    global sentence, last_char, last_time, recognition_active, reset_flag

    if reset_flag:
        sentence = ""
        last_char = ""
        recognition_active = False
        reset_flag = False  # Reset it so it doesn't repeat
        cv2.putText(frame, "Sentence Reset", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3, cv2.LINE_AA)
        return frame

    frame = make_square(frame)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            x_, y_, data_aux = [], [], []

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.extend([lm.x - min(x_), lm.y - min(y_)])

            H, W, _ = frame.shape
            x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
            x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10

            pred = predict_character(data_aux)
            try:
                predicted_character = labels_dict[int(pred)] if not isinstance(pred, str) else pred
            except (ValueError, KeyError):
                predicted_character = str(pred)

            current_time = time.time()
            time_since_last = current_time - last_time

            if predicted_character == 'START' and time_since_last > LETTER_DELAY:
                recognition_active = True
                last_char = ''
                last_time = current_time
                cv2.putText(frame, "Recognition Started", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)
                return frame

            if not recognition_active:
                cv2.putText(frame, "Show 'START' gesture to begin", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 255), 2, cv2.LINE_AA)
                return frame

            if predicted_character != last_char and time_since_last > LETTER_DELAY:
                if predicted_character == 'SPACE':
                    sentence += ' '
                elif predicted_character == 'SEND':
                    print("Sentence:", sentence)
                    reply = get_gemini_response(sentence)
                    print("Gemini Response:", reply)
                    speak_text(reply)
                    # speak_text_async(reply)
                    sentence = ""
                elif predicted_character not in ['START']:
                    sentence += predicted_character

                last_char = predicted_character
                last_time = current_time

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 79, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 79, 0), 3, cv2.LINE_AA)

    cv2.putText(frame, f"Sentence: {sentence}", (10, frame.shape[0] - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return frame
