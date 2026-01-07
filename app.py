from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import face_recognition
import os
import base64
import cv2
import numpy as np

app = Flask(__name__)

# --- CONFIGURATION ---
DATABASE_FOLDER = 'database'
UPLOAD_FOLDER = 'static/uploads'
TOLERANCE = 0.6

# Ensure folders exist
os.makedirs(DATABASE_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/database/<filename>')
def serve_database_image(filename):
    # This allows the HTML to display images from your protected database folder
    return send_from_directory(DATABASE_FOLDER, filename)

@app.route('/search', methods=['POST'])
def search():
    target_encoding = None

    # 1. CHECK SOURCE: UPLOAD or WEBCAM
    if 'file' in request.files and request.files['file'].filename != '':
        # CASE A: File Upload
        file = request.files['file']
        path = os.path.join(UPLOAD_FOLDER, "target.jpg")
        file.save(path)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            target_encoding = encodings[0]

    elif 'webcam_image' in request.form and request.form['webcam_image'] != '':
        # CASE B: Webcam Snapshot (Base64 string)
        data_url = request.form['webcam_image']
        header, encoded = data_url.split(",", 1)
        data = base64.b64decode(encoded)
        
        # Convert raw bytes to numpy array for OpenCV
        np_arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Convert BGR (OpenCV) to RGB (Face Recognition)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_img)
        if len(encodings) > 0:
            target_encoding = encodings[0]
            
    # 2. PERFORM SEARCH
    if target_encoding is None:
        return "No face found in the input image. <a href='/'>Try again</a>"

    matches = []
    for filename in os.listdir(DATABASE_FOLDER):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(DATABASE_FOLDER, filename)
            try:
                # Load database image
                db_image = face_recognition.load_image_file(image_path)
                db_encodings = face_recognition.face_encodings(db_image)
                
                # Check match
                results = face_recognition.compare_faces(db_encodings, target_encoding, tolerance=TOLERANCE)
                if True in results:
                    matches.append(filename)
            except:
                pass

    return render_template('results.html', matches=matches)

if __name__ == '__main__':
    # '0.0.0.0' allows access from other computers on the network
    app.run(host='0.0.0.0', port=5000, debug=True)