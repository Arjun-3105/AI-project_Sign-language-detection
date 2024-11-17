import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from flask import Flask, render_template, Response, request, jsonify

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Load the model
model_dict = pickle.load(open(r"models\model4.p", 'rb'))
model = model_dict['model']

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7, max_num_hands=1)

# Labels dictionary
labels_dict = {
    0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f',
    6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l',
    12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r',
    18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x',
    24: 'y', 25: 'z', 26: '0', 27: '1', 28: '2',
    29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9',
    36: 'I love You', 37: 'yes', 38: 'No', 39: 'Hello', 40: 'Thanks',
    41: 'Sorry', 43: 'space'
}

# Flask app initialization
app = Flask(__name__)

# Function to update the text displayed on the website
def update_text_field(text):
    return text if text != "space" else " "

# Function to generate video frames for the Flask app
def generate_frames():
    global last_detected_character,fixed_character, delayCounter, start_time
    last_detected_character = None

    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Make prediction using the model
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Draw a rectangle and the predicted character on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

                current_time = time.time()

                # Timer logic: Check if the predicted character is the same for more than 1 second
                if predicted_character == last_detected_character:
                    if (current_time - start_time) >= 1.0:  # Class fixed after 1 second
                        fixed_character = predicted_character
                        if delayCounter == 0:  # Add character once after it stabilizes for 1 second
                            fixed_character = update_text_field(fixed_character)
                            delayCounter = 1
                else:
                    # Reset the timer when a new character is detected
                    start_time = current_time
                    last_detected_character = predicted_character
                    delayCounter = 0  # Reset delay counter for a new character

        # Encode the frame as a JPEG image
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame for streaming in the HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Flask route to serve the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Flask route to serve the camera page with ASL prediction
@app.route('/camera')
def camera():
    return render_template('camera.html')

# Flask route to stream the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# POST route to handle ASL predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Assuming you're sending an image or data in the POST request
    # Here, we're just simulating an ASL prediction based on data you might send
    if request.method == 'POST':
        data = request.json  # You could also use `request.form` for form-data or `request.files` for image files
        
        # Sample processing (e.g., getting data from the request)
        prediction = model.predict([np.asarray(data['landmarks'])])
        predicted_character = labels_dict[int(prediction[0])]

        # Return the prediction as a JSON response
        return jsonify({'prediction': predicted_character})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, threaded=True)
