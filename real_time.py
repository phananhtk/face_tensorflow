from PIL import Image
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model

# for face detection
face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# resolution of the webcam
screen_width = 1280       # try 640 if code fails
screen_height = 720

# size of the image to predict
image_width = 224
image_height = 224

# load the trained model
model = load_model('transfer_learning_trained_face_cnn_model.h5')

# the labels for the trained model
with open("face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {key:value for key,value in og_labels.items()}
    print(labels)

# default webcam
stream = cv2.VideoCapture('rtsp://admin:SASEYZ@192.168.5.27:554')
#stream = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    (grabbed, frame) = stream.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # try to detect faces in the webcam
    faces = face_cascade.detectMultiScale(
        rgb, scaleFactor=1.3, minNeighbors=5)

    # for each faces found
    for (x, y, w, h) in faces:
        roi_rgb = rgb[y:y+h, x:x+w]
        # Draw a rectangle around the face
        color = (255, 0, 0) # in BGR
        stroke = 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

        # resize the image
        size = (image_width, image_height)
        resized_image = cv2.resize(roi_rgb, size)
        image_array = np.array(resized_image, "uint8")
        img = image_array.reshape(1,image_width,image_height,3) 
        img = img.astype('float32')
        img /= 255

        # predict the image
        predicted_prob = model.predict(img)
        # Display the label
        font = cv2.FONT_HERSHEY_DUPLEX
        name = 'Unknow'
        accuracy = 0.00
        text_rs = 'Unknow'
        if predicted_prob[0].max() > 0.7:
            name = labels[predicted_prob[0].argmax()]
            accuracy = round(predicted_prob[0].max()*100, 2)
            text_rs = f'[{name}]_[{accuracy}%]'
            print(f'[{name}]_[{accuracy}%]')
        color = (255, 255, 255)
        stroke = 2
        cv2.putText(frame, text_rs, (x,y-8),
            font, 0.6, color, stroke, cv2.LINE_AA)

    # Show the frame
    cv2.imshow("TTKD6", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):    # Press q to break out of the loop
        break

# Cleanup
stream.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)
