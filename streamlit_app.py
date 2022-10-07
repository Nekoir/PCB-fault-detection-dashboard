import streamlit as st
import pandas as pd
import numpy as np
import plost
import cv2 as cv
from PIL import Image
from deep_sort.tracker import Tracker
from object_detection import ObjectDetection

st.title("Test Object Detection (Yolov4, Streamlit, OpenCV)")
FRAME_WINDOW = st.image([])
detect = st.checkbox('Detect')

od = ObjectDetection()
deep = Tracker(10)
tracker = deep.predict()

cam = cv.VideoCapture(0)

cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

allowed_objects = ["keyboard", "laptop", "mouse", "remote", "scissors", "car", "person", "cell phone"]

while True:
    ret, frame = cam.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    (class_ids, scores, boxes) = od.detect(frame)
    for (class_id, score, box) in zip(class_ids, scores, boxes):
        x, y, w, h = box
        class_name = od.classes[class_id]
        
        if class_name in allowed_objects:
            color = od.colors[class_id]
            print(color)
            
            if detect is True:
                cv.putText(frame, "{}".format(class_name), (x, y - 15), cv.FONT_HERSHEY_PLAIN, 2, color, 2)
                cv.rectangle(frame, (x, y), (x + w, y + h), color, 3)
    
    FRAME_WINDOW.image(frame)
        
    if cv.waitKey(5) & 0xFF == 27:
        break
    
cam.release()
        
