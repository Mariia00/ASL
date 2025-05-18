from flask import Flask, render_template, Response, jsonify
import cv2
from app.asl_model.sign_language_recognition7 import process_frame, get_sentence, reset_sentence, speak_text
from app.asl_model.gemini_api import get_gemini_response

app = Flask(__name__)
cap = cv2.VideoCapture(0)


def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/send_sentence', methods=['POST'])
def send_sentence():
    sentence = get_sentence()
    if not sentence:
        return jsonify({'reply': 'No sentence detected yet.'})

    reply = get_gemini_response(sentence)
    speak_text(reply)
    reset_sentence()
    return jsonify({'reply': reply})


@app.route('/reset_sentence', methods=['POST'])
def reset():
    reset_sentence()
    return jsonify({'status': 'Sentence reset'})


if __name__ == '__main__':
    app.run(debug=True)
