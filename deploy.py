import streamlit as st
import av
import os
import cv2
import numpy as np
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer
from deployment_helper_funcs import predict_rt, predict_video
from run import datasets, train_model

st.set_page_config(
    page_title="Driver Drowsiness Detection | LearnOpenCV",
    page_icon="https://learnopencv.com/wp-content/uploads/2017/12/favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "### View the code at the GitHub repo",
    },
)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    red_img  = np.full((img.shape[0],img.shape[1],img.shape[2]), (0,0,255), np.uint8)

    # if not predict_rt(img):
    #     pass

    if predict_rt(img):
        img = cv2.add(img, red_img)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

with st.sidebar:
    st.image("./Assets/Logo.png")
    st.markdown("# Driver Drowsiness Detection", )
    st.subheader("Prevent accidents by alerting the driver")
    st.markdown('## Menu')
    choice = st.radio("Menu", ["Train","Upload","Real Time", "Logs"], label_visibility='collapsed')

if choice == 'Train':
    st.title("Train the model")
    st.markdown("### Model Architecture")

    col1, col2= st.columns([3,2])

    with col1:
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
    with col2:
        st.image('./Assets/cnn_arch.png')

    col1, col2= st.columns([1,1])

    with col1:
        training_dataset = st.selectbox('Training dataset name', datasets)
    with col2:
        epochs = st.number_input('Number of Epochs', step=1)
    
    col1, col2= st.columns([1,9])
    train_btn_state = False

    with col1:
        if st.button('Train', disabled=train_btn_state):
            with col2:
                with st.spinner('Training'):
                    train_btn_state = True
                    train_model(os.path.join('.', 'Data', training_dataset.split('/')[-1]), int(epochs))
                st.success('Done!')

if choice == 'Upload':
    st.title('Upload Video For Prediction')
    video_file = st.file_uploader('Video', type=['mp4'])
    col1, col2, col3= st.columns([2,6,2])

    with col2:
        if video_file is not None:
            with open(predict_video(video_file=video_file), 'rb') as out_file:
                st.video(out_file.read())

    

if choice == 'Real Time':
    st.title("Real Time Drowsiness Detection!")
    col1, col2, col3= st.columns([2,4,2])

    with col2:
        ctx = webrtc_streamer(
            key="driver-drowsiness-detection",
            video_frame_callback=video_frame_callback,
            # video_html_attrs=VideoHTMLAttributes(autoPlay=False, controls=False, muted=False),
            # rtc_configuration={
            # "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            # }
            video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=True)
        )
    

if choice == 'Logs':
    st.title('Logs')

    log_data = ''
    with st.spinner('Loading Logs'):
        with open(os.path.join('.', 'Logs', 'debug.log')) as log_file:
            lines = log_file.readlines()[-100:]
            lines.reverse()
            for line in lines:
                log_data += line
    st.markdown(
    f'''
    ```log
        {log_data}
    '''
    )