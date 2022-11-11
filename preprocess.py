import os
import mediapipe as mp
import matplotlib.pyplot as plt
import cv2
import logging
import numpy as np
from setup_data import setup_dirs

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
 
    # Initialize a matplotlib figure.
    fig = plt.figure(figsize=(20, 15))
    fig.set_facecolor("white")
 
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


def process_image(image, category, name):
    logging.info(f'Processing {name} in {category}')
    resized_img=None
    image = np.ascontiguousarray(image)
                            
    with mp_facemesh.FaceMesh(
        static_image_mode=True,       
        max_num_faces=1,              
        refine_landmarks=False,       
        min_detection_confidence=0.5, 
        min_tracking_confidence= 0.5,) as face_mesh:

        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
              resized_img = add_landmarks(name=name, img_dt=image.copy(), cat=category, face_landmarks=face_landmarks, save_img=True)
                          
    return resized_img

def process_dataset(dir_faces="./Data/drowsiness-prediction-dataset", face_cas_path="./Data/prediction-images/haarcascade_frontalface_default.xml", categories=None):
    
    logging.info('Setting up directories for preprocessing')
    setup_dirs(categories)

    imgs_with_landmarks=[]
    i=1
    for category in categories:
        logging.info(f'Processing {category}')
        path_link = os.path.join(dir_faces, category)
        class_num = categories.index(category)

        for image in os.listdir(path_link):
            image_array = cv2.imread(os.path.join(path_link, image), cv2.IMREAD_COLOR)
            face_cascade = cv2.CascadeClassifier(face_cas_path)
            faces = face_cascade.detectMultiScale(image_array, 1.3, 5)

            for (x, y, w, h) in faces:
                img = cv2.rectangle(image_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
                roi_color = img[y:y+h, x:x+w]
                land_face_array = process_image(roi_color, category, image)
                imgs_with_landmarks.append([land_face_array, class_num])
                i=i+1

    return imgs_with_landmarks