import argparse

import cv2
import mediapipe as mp


def main():
    parser = argparse.ArgumentParser(description="Smoke-test MediaPipe hand tracking.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.7,
        help="MediaPipe hand detection confidence threshold.",
    )
    args = parser.parse_args()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}.")

    print('Hand tracker running. Press "q" to quit.')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )

        cv2.imshow('MediaPipe Hands - Press "q" to quit', frame)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Hand tracking test completed successfully.")


if __name__ == "__main__":
    main()
