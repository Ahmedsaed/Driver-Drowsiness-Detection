import streamlit as st
import av
import threading
import cv2
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer
from deployment_helper_funcs import predict_rt

st.set_page_config(
    page_title="Driver Drowsiness Detection | LearnOpenCV",
    page_icon="https://learnopencv.com/wp-content/uploads/2017/12/favicon.png",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "### View the code at the GitHub repo",
    },
)

lock = threading.Lock()
img_container = {"img": None}

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    print(img.shape)
    # with lock:
    #     img_container["img"] = img
    if predict_rt(img):
        img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


st.title("Driver Drowsiness Detection!")

ctx = webrtc_streamer(
    key="driver-drowsiness-detection",
    video_frame_callback=video_frame_callback,
    video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=False),
)

# while ctx.state.playing:
#     with lock:
#         img = img_container["img"]
#     if img is None:
#         continue
    
    # prediction = predict(model, img)

    
