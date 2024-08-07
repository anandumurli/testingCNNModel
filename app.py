import cv2
from keras.api.models import load_model
import numpy as np
from keras.api.preprocessing.image import img_to_array

# Load the trained model
model = load_model('emotion_detection_model.keras')
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral']
# We are scoring the emotions with Anger being scored -3 and Happiness being scored at 4, neutral is 1
# There is no particular logic to these scores.
emotion_scores = [-3, -2, -1, 4, 1]
first_face_found = False

# Function to predict emotion
def predict_emotion(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face = face.astype("float") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    preds = model.predict(face)[0]
    return emotion_labels[np.argmax(preds)], np.max(preds)

# Real-time emotion detection
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
# once it sees a face the meeting sesh should start
# once the face gets out of the camera for good 30 seconds it should stop recording the meeting and be ready for a new meeting
# every meeting should be rated
    
           
    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        first_face_found = True
        emotion, confidence = predict_emotion(face)
        label = f"{emotion}: {confidence*100:.2f}%"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    cv2.imshow('Real-Time Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
