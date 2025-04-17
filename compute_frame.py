import os, time, sys, json
import threading
from datetime import datetime
import loop
from ultralytics import YOLO
import torch

class Human_Vision():
    def __init__(self):
        self.bar_model, self.bone_model = self.model_init()
        self.skeleton_connections = [
            (0, 1), (0, 2), (2, 4), (1, 3),  # Right arm
            (5, 7), (5, 6), (7, 9), (6, 8),  # Left arm
            (6, 12), (12, 14), (14, 16),  # Right leg
            (5, 11), (11, 13), (13, 15)   # Left leg
        ]
        self.threads =[]
        self.barrier = threading.Barrier(3)
        self.frame_count = 0

    def model_init(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        bar_model = YOLO("/path_to_best.pt")
        bone_model = YOLO("/path_to_yolov8n-pose.pt")
        bar_model.to(device)
        bone_model.to(device)
        return bar_model, bone_model
        
    def creat_thread(self, i, frame):
        thread = threading.Thread(target=self.process_vision,
                                    args = (i, frame, self.barrier) , daemon=True)
        self.threads.append(thread)
        thread.start()
        thread.join()
        return self.frame

    def process_vision(self, i, frame, barrier):
        start_time = time.time()
        fps = 0
        if i == 0:
            self.frame, fps, self.frame_count = loop.bar_frame(
                frame, self.bar_model, barrier, fps, start_time, self.frame_count)
        elif i == 1:
            self.frame, fps, self.frame_count = loop.bone_frame(
                frame, self.bar_model, barrier, fps, start_time, self.frame_count, self.skeleton_connections)
        else:
            self.frame, fps, self.frame_count = loop.general_frame(
                frame, barrier, fps, start_time, self.frame_count)
        