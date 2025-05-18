# file: generate_hand_landmark_dataset.py

import os
import pickle
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = '../data'
OUTPUT_FILE = 'data2.pickle'

data = []
labels = []

def make_square(image):
    h, w = image.shape[:2]
    size = max(h, w)
    delta_w = size - w
    delta_h = size - h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    for img_name in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = make_square(img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_list = [lm.x for lm in hand_landmarks.landmark]
                y_list = [lm.y for lm in hand_landmarks.landmark]
                min_x, min_y = min(x_list), min(y_list)

                data_aux = []
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)

                data.append(data_aux)
                labels.append(label)

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Saved {len(data)} samples to {OUTPUT_FILE}")
