import cv2
import tempfile
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("YOLO11 Object Detection")


task = st.sidebar.radio("Task", ["Detection", "Segmentation", "Pose Estimation"])
confidence = st.sidebar.slider("Confidence", 25, 100, 40) / 100


if task == "Detection":
    model = YOLO("weights/yolo11n.pt")
elif task == "Segmentation":
    model = YOLO("weights/yolo11n-seg.pt")
else:
    model = YOLO("weights/yolo11n-pose.pt")


source = st.sidebar.radio("Source", ["Image", "Video"])

if source == "Image":
    file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if file:
        image = Image.open(file)
        st.image(image, caption="Uploaded Image")

        if st.button("Detect Objects"):
            results = model(np.array(image), conf=confidence)
            result_plotted = results[0].plot()
            st.image(result_plotted, caption="Detected Image")

else:
    file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

    if file:
        st.video(file)

        if st.button("Detect Objects"):
            temp_video = tempfile.NamedTemporaryFile(delete=False)
            temp_video.write(file.read())

            video_cap = cv2.VideoCapture(temp_video.name)
            st_frame = st.empty()

            while video_cap.isOpened():
                success, frame = video_cap.read()
                if not success:
                    break

                frame = cv2.resize(frame, (720, 405))
                results = model(frame, conf=confidence)
                result_plotted = results[0].plot()
                st_frame.image(result_plotted, channels="BGR")

            video_cap.release()