import os
import mediapipe as mp
import matplotlib.pyplot as plt
import cv2
import logging
import numpy as np

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
    logging.debug(f'Adding landmarks to {name} image, save_img is set to {save_img}')

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


def preprocess_images(dir="./drowsiness-dataset/train"):
    IMG_SIZE = 145
    categories = ["drowse", "not_drowse"]
    for category in categories:
        path_link = os.path.join(dir, category)
        class_num1 = categories.index(category)
        for image_file in os.listdir(path_link):
            image = cv2.imread(os.path.join(path_link, image_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert to RGB
            image = np.ascontiguousarray(image)
            imgH, imgW, _ = image.shape
                             
            # Running inference using static_image_mode 
            with mp_facemesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,               
                refine_landmarks=False,        
                min_detection_confidence=0.5,  
                min_tracking_confidence= 0.5,) as face_mesh:

                results = face_mesh.process(image)

                # If detections are available.
                if results.multi_face_landmarks:  
                    # Iterate over detections of each face. Here, we have max_num_faces=1, 
                    # So there will be at most 1 element in 
                    # the 'results.multi_face_landmarks' list            
                    # Only one iteration is performed.
                    for face_id, face_landmarks in enumerate(results.multi_face_landmarks):    
                        add_landmarks(image_file, image.copy(), category, face_landmarks)

    # TODO: should return the status of preprocessing and image  

    return 