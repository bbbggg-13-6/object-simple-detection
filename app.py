import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

model = YOLO("yolov8n.pt")

st.title("YOLO Detection")

file = st.file_uploader(
    "Upload Image", type=["jpg", "jpeg", "png", "webp"]
)

if file:

    image = Image.open(file)

    results = model(
        np.array(image)
    )

    result = results[0].plot()

    st.image(result)