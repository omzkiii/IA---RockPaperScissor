import cv2
import joblib
import mediapipe as mp
import csv
import os
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.exceptions import NotFittedError

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils
# Open a video capture stream (you can replace this with your source)
cap = cv2.VideoCapture(2)

# Define the dimensions and position of the bounding boxes for each hand
bbox1_x, bbox1_y, bbox1_width, bbox1_height = 100, 100, 200, 200
bbox2_x, bbox2_y, bbox2_width, bbox2_height = 300, 100, 200, 200

# Output directory for captured CSV files
output_dir = 'captured_csv'
os.makedirs(output_dir, exist_ok=True)

# CSV file setup
csv_filename = './hand_positions.csv'
csv_exists = os.path.exists(csv_filename)

# Counter for captured files
capture_counter = 1

# Minimum number of data points required for prediction
min_data_points = 10

# Initialize capture_positions
capture_positions = False

# # Read the CSV file into a pandas DataFrame
# df = pd.read_csv(csv_filename)
# clf = DecisionTreeClassifier()
# pipe = joblib.load('./hand_model.sav')
# # Check if the DataFrame has enough data for prediction
# if len(df) >= min_data_points:
#     # Extract features and labels from the DataFrame
#     X = df.drop('Label', axis=1)
#     y = df['Label']
#
#     # Train a decision tree classifier
#     clf = DecisionTreeClassifier()
#     clf.fit(X.values, y)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw bounding boxes on the frame (default is green)
    # bbox1_color = (0, 255, 0)
    # bbox2_color = (0, 255, 0)

    # Process the frame to detect hands
    results = hands.process(frame_rgb)

    # List to store hand landmarks data
    hand_landmarks_data = []

    if results.multi_hand_landmarks:
        for hand_idx, landmarks in enumerate(results.multi_hand_landmarks):
            # Extract XYZ coordinates for all hand landmarks
            landmarks_data = [landmark.x for landmark in landmarks.landmark] + \
                              [landmark.y for landmark in landmarks.landmark] + \
                              [landmark.z for landmark in landmarks.landmark]

            hand_landmarks_data.append(landmarks_data)
            
            # Check if the hands are inside the bounding boxes
            # hand_inside_bbox1 = all(bbox1_x < landmark.x * frame.shape[1] < bbox1_x + bbox1_width
            #                         and bbox1_y < landmark.y * frame.shape[0] < bbox1_y + bbox1_height
            #                         for landmark in landmarks.landmark)
            
            # hand_inside_bbox2 = all(bbox2_x < landmark.x * frame.shape[1] < bbox2_x + bbox2_width
            #                         and bbox2_y < landmark.y * frame.shape[0] < bbox2_y + bbox2_height
            #                         for landmark in landmarks.landmark)

            # Change the bounding box colors if no hands are inside
            # if not hand_inside_bbox1:
            #     bbox1_color = (0, 0, 255)
            # if not hand_inside_bbox2:
            #     bbox2_color = (0, 0, 255)

            # Draw bounding boxes on the frame
            # cv2.rectangle(frame, (bbox1_x, bbox1_y), (bbox1_x + bbox1_width, bbox1_y + bbox1_height),
            #               bbox1_color, 2)
            # cv2.rectangle(frame, (bbox2_x, bbox2_y), (bbox2_x + bbox2_width, bbox2_y + bbox2_height),
            #               bbox2_color, 2)

            # Draw landmarks on the frame if hands are inside the boxes
            # if hand_inside_bbox1 or hand_inside_bbox2:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            # Draw a number on each hand
            hand_number = hand_idx + 1
            hand_label_position = (int(landmarks.landmark[0].x * frame.shape[1]),
                                   int(landmarks.landmark[0].y * frame.shape[0]) - 10)
            cv2.putText(frame, str(hand_number), hand_label_position, cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)


        
        def play(p1,p2):
            if(p1 == p2):
                print("Tie")
                return "Tie"
            elif(p1 == 'rock') & (p2 == 'scissors'):
                print("Player 1 wins")
                return"Player 1 wins"
            elif(p1 == 'scissors') & (p2 == 'paper'):
                print("Player 1 wins")
                return "Player 1 wins"
            elif(p1 == 'paper') & (p2 == 'rock'):
                print("Player 1 wins")
                return "Player 1 wins"
                
            elif(p2 == 'rock') & (p1 == 'scissors'):
                print("Player 2 wins")
                return "Player 2 wins"
            elif(p2 == 'scissors') & (p1 == 'paper'):
                print("Player 2 wins")
                return "Player 2 wins"
            elif(p2 == 'paper') & (p1 == 'rock'):
                print("Player 2 wins")
                return "Player 2 wins"
        try:
            # Use the trained model to make predictions for each hand
            
            predictions = pipe.predict(hand_landmarks_data)
            
            for idx, prediction in enumerate(predictions):
                cv2.putText(frame, f"Prediction {idx + 1}: {prediction}", (10, 20 * (idx + 1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if (len(predictions)>1):
                game_status = play(predictions[0],predictions[1])
                
                cv2.putText(frame, f"{game_status}", (10, 20 * (5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
            # Display predictions on the frame

        except NotFittedError:
            print("Not enough data points for prediction. Capture more data.")

    cv2.imshow('Hand Tracking', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()
