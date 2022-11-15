from model import load_saved_model, predict
from PIL import  Image
import numpy as np
import cv2
import tempfile

model = load_saved_model(load_last=True)

def predict_rt(img):
    return predict(model, img)

def predict_video(video_file):
    video_name = video_file.name 
    tempVidFile = tempfile.NamedTemporaryFile(delete=False)
    tempVidFile.write(video_file.read())

    vidCap = cv2.VideoCapture(tempVidFile.name)
    frame_width = int(vidCap.get(3)) 
    frame_height = int(vidCap.get(4)) 
    size = (frame_width, frame_height)

    red_img  = np.full((frame_height,frame_width,3), (0,0,255), np.uint8)
    out = cv2.VideoWriter(f'{video_name}_pred.mp4', cv2.VideoWriter_fourcc(*'DIVX'), vidCap.get(cv2.CAP_PROP_FPS), size)
    cur_frame = 0
    success = True

    while success:
        success, frame = vidCap.read() # get next frame from video
        # frame_img = Image.fromarray(frame) # convert opencv frame (with type()==numpy) into PIL Image
        frame_img = frame
        prediction = predict(model, frame_img)
        # print(type(frame_img), frame_width, frame_height)
        if prediction:
            frame_img = cv2.add(frame_img.shape, red_img)
        # try:
        # except:
        #     pass
        out.write(frame_img)
        cur_frame += 1

    return f'{video_name}_pred.avi'

