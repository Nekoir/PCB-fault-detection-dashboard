from tabnanny import check
import torch
import numpy as np
import cv2
import streamlit as st
from time import time
from deep_sort.tracker import Tracker
import streamlit as st

class PCBFaultDetection:
    """
    Class implements Yolo5 model to make inferences on a custom yolov5 model using Opencv2.
    """ 
    
    def __init__(self, capture_index, model_name):
        """
        Initializes the class with device and custom model.
        :capture_index: choose device.
        :model_name: file location of the custom weight.
        """
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
        deep = Tracker(10)
        deep.predict()
      
        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:   
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """     
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        cap = self.get_video_capture()
        assert cap.isOpened()
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        FRAME_WINDOW = st.image([])
            
        detect = st.button(label='Detect')
        
        if 'check' not in st.session_state:
            st.session_state.check = False

        while True:
          
            if detect:
                st.session_state.check = True
                
            ret, frame = cap.read()
            assert ret
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            start_time = time()
            results = self.score_frame(frame)
            
            labels, cord = results
            n = len(labels)
            x_shape, y_shape = frame.shape[1], frame.shape[0]
            for i in range(n):
                row = cord[i]
                if row[4] >= 0.3:
                    x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                    bgr = (0, 255, 0)
                    
                    if st.session_state.check is True:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                        cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
            
            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
             
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
            FRAME_WINDOW.image(frame)
 
        
        
st.title("Test Object Detection (Yolov4, Streamlit, OpenCV)")

# Create a new object and execute.
detector = PCBFaultDetection(capture_index=0, model_name='last.pt')
detector()