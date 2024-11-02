import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    # Convert the BGR image to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image to detect hands
    results = hands.process(img_rgb)

    # If hands are detected, draw the landmarks and connections
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the original BGR image
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0,0 , 255), thickness=2, circle_radius=2),
                                   mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2))

    # Display the image
    cv2.imshow("Hand Detection", img)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()