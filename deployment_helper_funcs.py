from model import load_saved_model, predict
from PIL import  Image
import cv2

model = load_saved_model(load_last=True)

def predict_rt(img):
    return predict(model, img)

def predict_video(video_name):
    vidCap = cv2.VideoCapture(video_name)
    frame_width = int(vidCap.get(3)) 
    frame_height = int(vidCap.get(4)) 
    size = (frame_width, frame_height)

    out = cv2.VideoWriter(f'{video_name}_pred.avi', cv2.VideoWriter_fourcc(*'DIVX'), vidCap.get(cv2.CAP_PROP_FPS), size)
    cur_frame = 0
    success = True

    while success:
        success, frame = vidCap.read() # get next frame from video
        pil_img = Image.fromarray(frame) # convert opencv frame (with type()==numpy) into PIL Image

        

        cur_frame += 1

