import cv2
import os
import numpy as np
import time

data_path = 'Photos'
person_name = 'Messi'
person_path = os.path.join(data_path, person_name)
desired_size = (300, 300)
# Upload the images
images = []
labels = []
id = 0
for image_name in os.listdir(person_path):
    image_path = os.path.join(person_path, image_name)
    image = cv2.imread(image_path)
    image = cv2.resize(image, desired_size)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    images.append(image_gray)
    labels.append(id)
    print(image_gray.shape)
    print(image_gray)
    
# Training model
face_recognizer = cv2.face.EigenFaceRecognizer_create()
print("Training...")
inicio = time.time()
face_recognizer.train(images, np.array(labels))
tiempoEntrenamiento = time.time()-inicio
print("Training Time: ", tiempoEntrenamiento)

face_recognizer.write('modeloEigenFace.xml')