from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

def gen_frames():  # Generate frame-by-frame from the camera
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    
    while True:
        success, img = cap.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                        mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                        mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2))

        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
