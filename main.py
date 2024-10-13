import cv2
import mediapipe as mp

# Initialize MediaPipe Hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# Function to detect hand and count raised fingers
def detect_fingers(frame, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            #Get finger landmark points
            landmarks = hand_landmarks.landmark

            #Thumb detection logic (if thumb is to the left or right of the hand)
            if landmarks[5].x < landmarks[17].x:
                thumb_is_open = landmarks[4].x < landmarks[3].x
            else:
                thumb_is_open = landmarks[4].x > landmarks[3].x

            #Fingers (check if tip is above the corresponding knuckle)
            fingers = [
                thumb_is_open,  #Thumb
                landmarks[8].y < landmarks[6].y,  #Index finger
                landmarks[12].y < landmarks[10].y,  #Middle finger
                landmarks[16].y < landmarks[14].y,  #Ring finger
                landmarks[20].y < landmarks[18].y  #Little finger
            ]
            return fingers.count(True)
    return 0


#Initialize the webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        # Flip the frame for better orientation and convert to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Detect hands
        results = hands.process(rgb_frame)

        #Detect number of raised fingers
        finger_count = detect_fingers(frame, results)

        cv2.putText(frame, f'Fingers: {finger_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3,
                    cv2.LINE_AA)

        #Show the frame
        cv2.imshow("Hand Number Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

