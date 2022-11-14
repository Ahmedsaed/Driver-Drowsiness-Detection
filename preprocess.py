import os
import mediapipe as mp
import matplotlib.pyplot as plt
import cv2
import logging
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from setup_data import setup_dirs
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


mp_facemesh = mp.solutions.face_mesh
mp_drawing  = mp.solutions.drawing_utils
denormalize_coordinates = mp_drawing._normalized_to_pixel_coordinates

# Landmark points corresponding to left eye
all_left_eye_idxs = list(mp_facemesh.FACEMESH_LEFT_EYE)
# flatten and remove duplicates
all_left_eye_idxs = set(np.ravel(all_left_eye_idxs)) 
 
# Landmark points corresponding to right eye
all_right_eye_idxs = list(mp_facemesh.FACEMESH_RIGHT_EYE)
all_right_eye_idxs = set(np.ravel(all_right_eye_idxs))
 
# Combined for plotting - Landmark points for both eye
all_idxs = all_left_eye_idxs.union(all_right_eye_idxs)
 
# The chosen 12 points:   P1,  P2,  P3,  P4,  P5,  P6
chosen_left_eye_idxs  = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33,  160, 158, 133, 153, 144]
all_chosen_idxs = chosen_left_eye_idxs + chosen_right_eye_idxs

def add_landmarks(name, img_dt, cat, face_landmarks, ts_thickness=1, ts_circle_radius=2, lmk_circle_radius=3, save_img=False, black_background=False):
    logging.debug(f'Adding landmarks to {name} image. {"" if save_img else "not "} saving {name} to {os.path.join(".", "Data", "landmarks", cat, str(name))}')

    img_height, img_width = img_dt.shape[0], img_dt.shape[1]

    # For plotting Face Tessellation
    if black_background:
        img_background = np.zeros((img_dt.shape[0], img_dt.shape[1], img_dt.shape[2]), dtype='uint8')
    else:
        img_background = img_dt.copy()

    img_eye_lmks_chosen = image_eye_lmks = image_drawing_tool = img_background
 
    # Initializing drawing utilities for plotting face mesh tessellation
    connections_drawing_spec = mp_drawing.DrawingSpec(
        thickness=ts_thickness, 
        circle_radius=ts_circle_radius, 
        color=(255, 255, 255))

 
    # Draw landmarks on face using the drawing utilities.
    mp_drawing.draw_landmarks(
        image=image_drawing_tool,
        landmark_list=face_landmarks,
        connections=mp_facemesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=connections_drawing_spec)

    landmarks = face_landmarks.landmark
 
    for landmark_idx, landmark in enumerate(landmarks):
        if landmark_idx in all_idxs:
            pred_cord = denormalize_coordinates(landmark.x, 
                                                landmark.y, 
                                                img_height, img_width)
            cv2.circle(image_eye_lmks, 
                       pred_cord, 
                       lmk_circle_radius, 
                       (255, 255, 255), 
                       -1)
 
        if landmark_idx in all_chosen_idxs:
            pred_cord = denormalize_coordinates(landmark.x, 
                                                landmark.y, 
                                                img_height, img_width)
            cv2.circle(img_eye_lmks_chosen, 
                       pred_cord, 
                       lmk_circle_radius, 
                       (255, 255, 255), 
                       -1)

    if save_img:
        cv2.imwrite(os.path.join('.', 'Data', 'landmarks', cat, str(name)), image_drawing_tool)

    return image_drawing_tool


def process_image(image, category, name, face_cas_path="./Data/prediction-images/haarcascade_frontalface_default.xml", save_img=True):
    logging.info(f'Processing {name} in {category}')
    resized_img=None
    IMG_SIZE = 145
    
    face_cascade = cv2.CascadeClassifier(face_cas_path)
    faces = face_cascade.detectMultiScale(image, 1.3, 5)

    roi_color = None
    for (x, y, w, h) in faces:
        img = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_color = img[y:y+h, x:x+w]

    image = np.ascontiguousarray(roi_color)

    with mp_facemesh.FaceMesh(
        static_image_mode=True,       
        max_num_faces=1,              
        refine_landmarks=False,       
        min_detection_confidence=0.5, 
        min_tracking_confidence= 0.5,) as face_mesh:

        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
              resized_img = add_landmarks(name=name, img_dt=image.copy(), cat=category, face_landmarks=face_landmarks, save_img=save_img)

        resized_img = cv2.resize(resized_img, (IMG_SIZE, IMG_SIZE))

    return resized_img

def process_dataset(dir_faces="./Data/drowsiness-prediction-dataset", categories=None):
    
    logging.info('Setting up directories for preprocessing')
    setup_dirs(categories)

    imgs_with_landmarks=[]
    for category in categories:
        logging.info(f'Processing {category}')
        path_link = os.path.join(dir_faces, category)
        class_num = categories.index(category)

        for image in os.listdir(path_link):
            try:
                image_array = cv2.imread(os.path.join(path_link, image))
                land_face_array = process_image(image_array, category, image)
                imgs_with_landmarks.append([land_face_array, class_num])
            except:
                logging.info(f"Couldn't process {image} in {category}")

    return imgs_with_landmarks

def setup_training_data(data, test_size=0.2):
    logging.info('Extracting features and labels for training')

    x, y = [], []
    for feature, label in data:
        x.append(feature)
        y.append(label)

    x = np.array(x)
    
    label_bin = LabelBinarizer()
    y = label_bin.fit_transform(y)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=test_size)

    train_generator = ImageDataGenerator(rescale=1/255, zoom_range=0.2, horizontal_flip=True, rotation_range=30)
    test_generator = ImageDataGenerator(rescale=1/255)

    train_generator = train_generator.flow(np.array(X_train), y_train, shuffle=False)
    test_generator = test_generator.flow(np.array(X_test), y_test, shuffle=False)

    return train_generator, test_generator

def load_landmarks(categories):
    logging.info(f'Loading landmarks from {os.path.join(".", "Data", "landmarks")}')
    
    IMG_SIZE=145
    imgs_with_landmarks=[]
    for category in categories:
        category_path = os.path.join('.', 'Data', 'landmarks', category)
        class_num = categories.index(category)

        for image in os.listdir(category_path):
            image_array = cv2.imread(os.path.join(category_path, image), cv2.IMREAD_COLOR)
            image_array = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
            imgs_with_landmarks.append([image_array, class_num])
    
    return imgs_with_landmarks