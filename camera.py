from flask import Flask, Response
import cv2
from deepface import DeepFace
import threading
import time
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

import os
from dotenv import load_dotenv
load_dotenv()

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-pro')

app = Flask(__name__)

# Initialize Firebase Admin SDK
cred = credentials.Certificate('credentials.json')  # Path to the downloaded Firebase credentials
firebase_admin.initialize_app(cred)

db = firestore.client()

# Emotion data
emotion_values = { 'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0 }
frame_buffer = None

data_in_minutes = []

def capture_and_analyze_emotions():
    global frame_buffer
    cap = cv2.VideoCapture(1)
    last_saved_time_emotion = time.time()
    last_saved_time_genai = time.time()
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']
            cv2.putText(frame, dominant_emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            emotion_values[dominant_emotion] += 1
        except Exception as e:
            print("Error during analysis:", e)

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame_buffer = buffer.tobytes()

        # Check if 10 seconds have passed
        if time.time() - last_saved_time_emotion >= 10:
            save_highest_emotion()
            last_saved_time_emotion = time.time()
        
        if time.time() - last_saved_time_genai >= 60:
            save_genai_analysis()
            last_saved_time_genai = time.time()

    cap.release()

def save_genai_analysis():
    request = ""
    request += "I am going to give you a list of 6 emotions and their respective time. This emotions are from patients in a mental clinic. Your job is to analyze risk of the patient. Your response will be given to the nurses that work in the clinic. Give the your risk analysis and some suggestions. If the risk is very high, add this text at the start of the response HIGH RISK!!!.\n\nThe Data"
    for emotion in data_in_minutes:
        request += f"{emotion['date']}: {emotion['emotion']}\n"
    response = model.generate_content(request)
    record = {
        'CameraID': 1,
        'date': datetime.now(),
        'analysis': response.text
    }
    db.collection('analysis').add(record)
    print(f"Saved: {response} at {datetime.now()}")

    # Reset emotion counts
    data_in_minutes.clear()

def save_highest_emotion():
    highest_emotion = max(emotion_values, key=emotion_values.get)
    data_in_minutes.append({'emotion': highest_emotion, 'date':datetime.now().strftime("%m/%d/%Y, %H:%M:%S")})
    record = {
        'cameraID': 1,
        'date': datetime.now(),
        'emotion': highest_emotion
    }
    db.collection('emotions').add(record)
    print(f"Saved: {highest_emotion} at {datetime.now()}")

    # Reset emotion counts
    for key in emotion_values.keys():
        emotion_values[key] = 0

def gen_frames():
    global frame_buffer
    while True:
        if frame_buffer is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_buffer + b'\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: text/plain\r\n\r\nWaiting for frame...\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def start_background_task():
    thread = threading.Thread(target=capture_and_analyze_emotions)
    thread.daemon = True
    thread.start()

if __name__ == '__main__':
    start_background_task()
    app.run(debug=True, use_reloader=False, threaded=True)
