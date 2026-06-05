import argparse
import pickle
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf


def extract_landmarks(hand_landmarks):
    landmarks = []
    for point in hand_landmarks.landmark:
        landmarks.extend([point.x, point.y, point.z])
    return np.array(landmarks, dtype=np.float32).reshape(1, -1)


def main():
    parser = argparse.ArgumentParser(
        description="Run live desktop inference with a trained landmark model."
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument(
        "--model",
        default="models/private/asl_model_26.h5",
        help="Path to a trained Keras model.",
    )
    parser.add_argument(
        "--labels",
        default="models/private/label_encoder_26.pkl",
        help="Path to the saved label encoder.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Number of recent predictions used for majority-vote smoothing.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    labels_path = Path(args.labels)
    if not model_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            "Model or label encoder not found. Run train_model.py first, or pass "
            "--model and --labels."
        )

    model = tf.keras.models.load_model(model_path)
    with labels_path.open("rb") as file:
        label_encoder = pickle.load(file)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}.")

    pred_buffer = deque(maxlen=args.smoothing_window)

    print("Running recognizer. Press q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )

                prediction = model.predict(extract_landmarks(hand_landmarks), verbose=0)[0]
                class_id = int(np.argmax(prediction))
                confidence = float(prediction[class_id])

                pred_buffer.append(class_id)
                smoothed = max(set(pred_buffer), key=list(pred_buffer).count)
                label = label_encoder.inverse_transform([smoothed])[0]

                cv2.putText(
                    frame,
                    f"{label} ({confidence:.2%})",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow("Few-Shot Sign Lab Recognizer", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
