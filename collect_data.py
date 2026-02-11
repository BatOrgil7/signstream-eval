import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Create dataset directory
DATASET_DIR = "asl_dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

# Define your signs (start with just 2-3)
SIGNS = ["A", "B", "C"]  # Add more as you go

# Create a CSV file to store landmark data
csv_file = open("landmarks.csv", "a", newline="")
csv_writer = csv.writer(csv_file)

# Start webcam
cap = cv2.VideoCapture(0)
current_sign = "A"
collecting = False
counter = 0
TOTAL_SAMPLES = 200  # Per sign

print(f"Press SPACE to start collecting for sign: {current_sign}")
print("Press 'n' for next sign")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip for selfie view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = hands.process(rgb_frame)
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmarks (21 points, each with x, y, z)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            # Save data when collecting
            if collecting and counter < TOTAL_SAMPLES:
                csv_writer.writerow([current_sign] + landmarks)
                counter += 1
                print(f"Collected {counter}/{TOTAL_SAMPLES} for {current_sign}")
    
    # Display info
    cv2.putText(frame, f"Sign: {current_sign}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Collected: {counter}/{TOTAL_SAMPLES}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if collecting:
        cv2.putText(frame, "COLLECTING...", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Data Collection", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' '):  # SPACE to start/stop collecting
        collecting = not collecting
        if collecting:
            print(f"Started collecting for {current_sign}")
        else:
            print(f"Stopped collecting for {current_sign}")
    
    elif key == ord('n'):  # Next sign
        if current_sign in SIGNS:
            next_index = (SIGNS.index(current_sign) + 1) % len(SIGNS)
            current_sign = SIGNS[next_index]
            counter = 0
            print(f"Switched to sign: {current_sign}")
    
    elif key == ord('q'):  # Quit
        break

cap.release()
cv2.destroyAllWindows()
csv_file.close()
print("Data collection complete!")