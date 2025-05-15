"""
Hand Sign Recognition Using Mediapipe + Pickle Model
Builds sentences from recognized letters.
"""

import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Webcam capture
cap = cv2.VideoCapture(0)

# Mediapipe hands module setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Optional: If model returns index, use label dict
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

sentence = ""
last_char = ""
last_time = time.time()


def predict_character(data_aux):
    prediction = model.predict([np.asarray(data_aux)])
    return prediction[0]


def process_frame(frame):
    global sentence, last_char, last_time

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
            predicted_character = pred if isinstance(pred, str) else labels_dict.get(int(pred), '?')

            # Update sentence if character has changed and enough time has passed
            current_time = time.time()
            if predicted_character != last_char or (current_time - last_time) > 1.5:
                sentence += predicted_character
                last_char = predicted_character
                last_time = current_time

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 79, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 79, 0), 3, cv2.LINE_AA)

    # Show sentence at bottom of the frame
    cv2.putText(frame, f"Sentence: {sentence}", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def main():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = process_frame(frame)
        cv2.imshow('Sign Language Recognition', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()