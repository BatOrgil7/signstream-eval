import argparse
import csv
import string
from pathlib import Path

import cv2
import mediapipe as mp


def parse_labels(value):
    if value.lower() == "alphabet":
        return list(string.ascii_uppercase)
    return [label.strip() for label in value.split(",") if label.strip()]


def extract_landmarks(hand_landmarks):
    landmarks = []
    for point in hand_landmarks.landmark:
        landmarks.extend([point.x, point.y, point.z])
    return landmarks


def main():
    parser = argparse.ArgumentParser(
        description="Collect MediaPipe hand landmarks for isolated sign labels."
    )
    parser.add_argument(
        "--labels",
        default="alphabet",
        help='Comma-separated labels such as "A,B,C" or "alphabet" for A-Z.',
    )
    parser.add_argument(
        "--output",
        default="data/private/landmarks_all.csv",
        help="CSV output path. Private data paths are ignored by git.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=200,
        help="Target samples to collect per label.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.7,
        help="MediaPipe hand detection confidence threshold.",
    )
    args = parser.parse_args()

    labels = parse_labels(args.labels)
    if not labels:
        raise ValueError("At least one label is required.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}.")

    current_index = 0
    collecting = False
    samples_collected = 0

    print("=== Few-Shot Sign Lab: Data Collector ===")
    print("SPACE: start/stop collecting for current label")
    print("n: next label")
    print("q: quit")
    print(f"Labels: {', '.join(labels)}")
    print(f"Writing private samples to: {output_path}")

    with output_path.open("a", newline="") as csv_file:
        writer = csv.writer(csv_file)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            current_label = labels[current_index]

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                    )

                    if collecting:
                        writer.writerow([current_label] + extract_landmarks(hand_landmarks))
                        samples_collected += 1

            info = (
                f"Label: {current_label}  "
                f"Collected: {samples_collected}/{args.samples_per_label}"
            )
            cv2.putText(
                frame,
                info,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            if collecting:
                cv2.putText(
                    frame,
                    "RECORDING",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Few-Shot Sign Lab Collector", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                collecting = not collecting
                status = "Started" if collecting else "Stopped"
                print(f"{status} collecting {current_label}")
            elif key == ord("n"):
                current_index = (current_index + 1) % len(labels)
                samples_collected = 0
                print(f"Switched to {labels[current_index]}")
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Data collection finished.")


if __name__ == "__main__":
    main()
