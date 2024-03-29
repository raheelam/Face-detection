import base64
from io import BytesIO
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image


# Face Detect
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Face Recognition
recognizer_eigen = cv2.face.EigenFaceRecognizer_create(num_components=15, threshold=4000)
recognizer_fisher = cv2.face.FisherFaceRecognizer_create(num_components=2, threshold=400)
recognizer_lbph = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=1, grid_x=8, grid_y=8, threshold=50)

# Lists to store training and testing data
training_faces = []
training_ids = []
testing_faces = []
testing_ids = []

undetected_faces = []



# Create Tkinter window
window = tk.Tk()
window.title("Face Recognition App")

def reinitialize():
    training_faces.clear()
    training_ids.clear()
    testing_faces.clear()
    testing_ids.clear()
    undetected_faces.clear()
    recognizer_eigen = cv2.face.EigenFaceRecognizer_create(80)
    recognizer_fisher = cv2.face.FisherFaceRecognizer_create()
    recognizer_lbph = cv2.face.LBPHFaceRecognizer_create()



def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def adjust_brightness(image, beta):
    return cv2.convertScaleAbs(image, beta=beta)

def adjust_contrast(image, alpha):
    return cv2.convertScaleAbs(image, alpha=alpha)

def apply_histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return cv2.merge([equalized, equalized, equalized])

def convert_to_images(cv_images):
    images =[]
    for cv_image in cv_images:
        images.append(convert_cvimage_to_image(cv_image))
    return images


def convert_cvimage_to_image(img):
    # Convert NumPy array to PIL image
    pil_img = Image.fromarray(img)

    # Convert PIL image to base64 string
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")  # You can choose the format based on your image type
    encoded_img = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return encoded_img


# Function to detect faces and add them to the list
def face_detect_add(faces, ids, file_name, face_id, is_test=False):
    img = cv2.imread(file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    print(f"filename: {file_name}")

    # gray = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

        # Preprocess the image (e.g., resize, equalize histogram)
    # face_img = cv2.resize(gray, (100, 100))
    
# 1.3,8,30

    brightness_factor = 0.5  # Increase brightness by 50%

    # brighter_image = cv2.addWeighted(gray, 1.0, np.zeros(gray.shape, gray.dtype), 0.0, brightness_factor)
    # gray = brighter_image

    
    directory, filename = os.path.split(file_name)
    
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces_detected) == 1:
        x, y, w, h = faces_detected[0]
        face_img = gray[y:y+h, x:x+w] 
        resized_face = cv2.resize(face_img, (55,55))

        eq_img = cv2.equalizeHist(resized_face)
        brightness_factor = 0.5  # Increase brightness by 50%

        brighter_image = cv2.addWeighted(eq_img, 1.0, np.zeros(eq_img.shape, eq_img.dtype), 0.0, brightness_factor)
        # gray = brighter_image
        faces.append(brighter_image)
        
        # faces.append(resized_face)

        ids.append(face_id)
       
        # output_path = os.path.join('Detected', str(face_id)+filename)
        # cv2.imwrite(output_path, gray)

    
       
# Function to train the face recognition models
def train(recognizer, faces, ids):
    recognizer.train(faces, np.array(ids).flatten())

# Function to perform testing
def test(recognizer, faces, ids):
    correct_predictions = 0
    print(f"ids: {ids}")
    total_confidence = 0
    # print(f"faces: {faces}")
    for idx, face in enumerate(faces):
        label, confidence = recognizer.predict(face)
        print(f"confidence: {confidence} for {label}")
        print(f"idx: {idx}  idsx:{ids[idx]}")
        total_confidence += confidence
        if label == ids[idx]:
            correct_predictions += 1
        
    average_dist = total_confidence / len(ids)
    accuracy = (correct_predictions / len(ids)) * 100
   
    return {'accuracy':accuracy, 'average_dist':average_dist}

def test_button_callback():
  eigen_result = test(recognizer_eigen, testing_faces, testing_ids)
  fisher_result = test(recognizer_fisher, testing_faces, testing_ids)
  lbph_result =  test(recognizer_lbph, testing_faces, testing_ids)
  return {'eigen': eigen_result,'fisher':fisher_result, 'lbph':lbph_result}

# Function to handle the "Load Images" button click
def load_images():
    # Loading faces for training
    reinitialize()
    for face_id in range(1, 6): #6
        for i in range(1, 50):
            file_name = f'Faces/{face_id:02d}/{"00" if i < 10 else "0"}{i}.jpg'
            face_detect_add(training_faces, training_ids, file_name, face_id)

    # Loading faces for testing
    for face_id in range(1, 6):
        for i in range(51, 100):
            file_name = f'Faces/{face_id:02d}/{"00" if i < 10 else "0"}{i}.jpg'
            face_detect_add(testing_faces, testing_ids, file_name, face_id, True)

    return {'testing_faces':convert_to_images(testing_faces) , 'training_faces':convert_to_images(training_faces)}
    

# Function to handle the "Train Models" button click
def train_models():
    train(recognizer_eigen, training_faces, training_ids)
    train(recognizer_fisher, training_faces, training_ids)
    train(recognizer_lbph, training_faces, training_ids)
    print("Models trained successfully.")

