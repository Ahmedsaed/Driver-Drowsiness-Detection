import streamlit as st
import av
import threading
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
    # with lock:
    #     img_container["img"] = img
    if predict_rt(img):
        pass

    return av.VideoFrame.from_ndarray(img, format="bgr24")

with st.sidebar:
    st.image("./Assets/Logo.png")
    st.markdown("# Driver Drowsiness Detection", )
    st.markdown('## Menu')
    choice = st.radio("Menu", ["Train","Upload","Real Time", "Logs"], label_visibility='collapsed')
    st.info("Description")

if choice == 'Train':
    st.title("Train the model")
    st.code('''
        model = Sequential([
                Conv2D(16, 3, activation='relu', input_shape=(145, 145, 3)),
                BatchNormalization(),
                MaxPooling2D(),
                Dropout(0.1),

                Conv2D(32, 5, activation='relu'),
                BatchNormalization(),
                MaxPooling2D(),
                Dropout(0.1),

                Conv2D(64, 10, activation='relu'),
                BatchNormalization(),
                MaxPooling2D(),
                Dropout(0.1),


                Conv2D(128, 12, activation='relu'),
                BatchNormalization(),

                Flatten(),

                Dense(512, activation='relu'),
                Dropout(0.1),
                Dense(1, activation='sigmoid')
            ])
    ''')
    # st.image('./Assets/')


if choice == 'Real Time':
    st.title("Real Time Drowsiness Detection!")
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

    
