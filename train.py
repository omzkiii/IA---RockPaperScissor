import cv2
import mediapipe as mp
import csv
import os
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.exceptions import NotFittedError

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Open a video capture stream (you can replace this with your source)
cap = cv2.VideoCapture(2)

# Define the dimensions and position of the bounding box
bbox_x, bbox_y, bbox_width, bbox_height = 100, 100, 200, 200

# Output directory for captured CSV files
output_dir = 'captured_csv'
os.makedirs(output_dir, exist_ok=True)

# CSV file setup
csv_filename = 'Project/hand_positions.csv'
csv_exists = os.path.exists(csv_filename)

# Counter for captured files
capture_counter = 1

# Minimum number of data points required for prediction
min_data_points = 10

# Initialize capture_positions
capture_positions = False

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_filename)
clf = DecisionTreeClassifier()

# Check if the DataFrame has enough data for prediction
if len(df) >= min_data_points:
    # Extract features and labels from the DataFrame
    X = df.drop('Label', axis=1)
    y = df['Label']

    # Train a decision tree classifier
    clf = DecisionTreeClassifier()
    clf.fit(X.values, y)
label = input("Enter label (rock, paper, scissors): ")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw the bounding box on the frame (default is green)
    bbox_color = (0, 255, 0)

    # Check if there is a hand inside the bounding box
    hand_inside_box = False

    # Process the frame to detect hands
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Extract XYZ coordinates for all hand landmarks
            hand_landmarks_data = [landmark.x for landmark in landmarks.landmark] + \
                                   [landmark.y for landmark in landmarks.landmark] + \
                                   [landmark.z for landmark in landmarks.landmark]

            # Check if the hand is inside the bounding box
            hand_inside_box = all(bbox_x < landmark.x * frame.shape[1] < bbox_x + bbox_width
                                  and bbox_y < landmark.y * frame.shape[0] < bbox_y + bbox_height
                                  for landmark in landmarks.landmark)

    # Change the bounding box color to red if no hands are inside
    if not hand_inside_box:
        bbox_color = (0, 0, 255)

    # Draw the bounding box on the frame
    cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_width, bbox_y + bbox_height), bbox_color, 2)

    # Draw landmarks on the frame if a hand is inside the box

    if hand_inside_box:
        for hand_landmarks in results.multi_hand_landmarks:

            try:
                # Use the trained model to make a prediction
                prediction = clf.predict([hand_landmarks_data])

                # Display the prediction on the frame
                cv2.putText(frame, f"Prediction: {prediction[0]}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)

            except NotFittedError:
                print("Not enough data points for prediction. Capture more data.")

            # Capture hand positions and save an image when the spacebar is pressed
            key = cv2.waitKey(1)

            # Append to the CSV file
            with open(csv_filename, 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                if not csv_exists:
                    header = ['X' + str(i) for i in range(21)] + ['Y' + str(i) for i in range(21)] + \
                                ['Z' + str(i) for i in range(21)] + ['Label']
                    csv_writer.writerow(header)  # Header row
                csv_writer.writerow(hand_landmarks_data + [label])
                print(f"Total rows collected: {capture_counter}")
                capture_counter = capture_counter+1
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)


    cv2.imshow('Hand Tracking', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()
