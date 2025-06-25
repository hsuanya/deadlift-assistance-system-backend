import threading
import queue
from ultralytics import YOLO
import torch
import loop
import os
import cv2
import numpy as np
from datetime import datetime
import json
from tools.Deadlift_tool.interpolate import run_interpolation
from tools.Deadlift_tool.interpolate import run_interpolation
from tools.Deadlift_tool.bar_data_produce import run_bar_data_produce
from tools.Deadlift_tool.data_produce import run_data_produce
from tools.Deadlift_tool.data_split import run_data_split
from tools.Deadlift_tool.predict import run_predict
from tools.trajectory import plot_trajectory

class Human_Vision:
    def __init__(self):
        self.bar_model, self.bone_model = self.model_init()
        dir = './'
        self.save_path = {
            'Deadlift': os.path.join(dir, 'recordings', 'deadlift'),
            'Benchpress': os.path.join(dir, 'recordings', 'benchpress'),
            'Squat': os.path.join(dir, 'recordings', 'squat')
        }
        self.visions = ['bar', 'left-front', 'left-back']

        self.skeleton_connections = [
            (0, 1),
            (0, 2),
            (2, 4),
            (1, 3),  # Right arm
            (5, 7),
            (5, 6),
            (7, 9),
            (6, 8),  # Left arm
            (6, 12),
            (12, 14),
            (14, 16),  # Right leg
            (5, 11),
            (11, 13),
            (13, 15)  # Left leg
        ]
        self.frame_count = 0
        self.lock = threading.Lock()  # Protect frame_count
        self.barrier = threading.Barrier(3)
        self.clear_runtime_data()
        self.recording_sig = False
        self.frame_count_for_detect = 0

    def clear_runtime_data(self):
        self.threads = []
        self.frames = queue.Queue()
        self.idxs = queue.Queue()

    def model_init(self):
        device_0 = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        bar_model = YOLO("./model/deadlift/bar/best.pt").to(device_0)
        bone_model = YOLO("./model/deadlift/skeleton/yolo11s-pose.pt").to(device_0)

        # 預先跑一次，避免 multi-thread 時出現 fuse 的錯誤
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        with torch.no_grad():
            bar_model.predict(dummy)
            bone_model.predict(dummy)

        return bar_model, bone_model

    def create_thread(self, i, frame, rc_sig, id):
        # rc_sig 為新訊號
        # self.recording_sig 為舊訊號
        if rc_sig and rc_sig != self.recording_sig:
            if i == 0:
                now = datetime.now()
                timestamp = now.strftime("%Y%m%d_%H%M%S")
                folder = os.path.join(self.save_path['Deadlift'],
                                        f"recording_{timestamp}")
                os.makedirs(folder, exist_ok=True)
                with open("recordings.json", mode='r', encoding='utf-8') as f:
                    data = json.load(f)
                if str(id) not in data:
                    data[str(id)] = {}
                data[str(id)]["recording_folder"] = folder
                with open("recordings.json", mode='w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)

                self.outs = [None] * 3
                self.bar_file = None
                self.skeleton_files = [None] * 3
                self.outs, self.bar_file, self.skeleton_files = self.rc_prep(frame, folder)
                print('Prepare for writing image')
                self.frame_count_for_detect = 0
        elif rc_sig == self.recording_sig:
            if i == 0:
                self.frame_count_for_detect += 1
        else:
            if i == 0:
                print('close file')
                for out in self.outs:
                    if out:
                        out.release()
                for file in self.bar_files:
                    if file:
                        file.close()
                for file in self.skeleton_files:
                    if file:
                        file.close()
                self.outs = [None] * 3
                self.bar_file = None
                self.skeleton_files = [None] * 3
                self.frame_count_for_detect = 0

        self.recording_sig = rc_sig
        thread = threading.Thread(target=self.process_vision,
                                    args=(i, frame),
                                    daemon=True)
        self.threads.append(thread)

    def process_vision(self, i, frame):
        out = self.outs[i]
        bar_file = self.bar_file
        skeleton_file = self.skeleton_files[i]
        if i == 0:
            frame = loop.bar_frame(frame, self.bar_model, self.bone_model,
                                    self.skeleton_connections, skeleton_file,
                                    bar_file, self.frame_count_for_detect)
        elif i == 1:
            frame = loop.bone_frame(frame, self.bone_model,
                                    self.skeleton_connections, skeleton_file,
                                    self.frame_count_for_detect)
        else:
            frame = loop.bone_frame(frame, self.bone_model,
                                    self.skeleton_connections, skeleton_file,
                                    self.frame_count_for_detect)
        cond = skeleton_file is not None and out is not None and bar_file is not None
        if cond:
            out.write(frame)

        self.frames.put(frame)
        self.idxs.put(i)

    def get_frame(self):
        for thread in self.threads:
            thread.start()
        for thread in self.threads:
            thread.join()

        frames, idxs = [], []
        while not self.frames.empty():
            frames.append(self.frames.get())
            idxs.append(self.idxs.get())

        self.clear_runtime_data()
        return frames, idxs

    def rc_prep(self, frame, folder):
        bar_file = open(os.path.join(folder, 'coordinates.txt'), "w")
        outs = [None] * 3
        skeleton_files = [None] * 3
        for idx, vision in enumerate(self.visions):
            file = os.path.join(folder, f'vision_{vision}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 編碼
            frame_size = (frame.shape[0], frame.shape[1])
            out = cv2.VideoWriter(file, fourcc, 30, frame_size)
            skeleton_file = open(os.path.join(folder, f'skeleton_{vision}.txt'), "w")
            outs[idx] = out
            skeleton_files[idx] = skeleton_file
        return outs, bar_file, skeleton_files

def predict(folder):
    os.makedirs(f'{folder}/config', exist_ok=True)
    # 對槓端及骨架做內插
    run_interpolation(folder)
    # bar
    run_bar_data_produce(folder, sport='deadlift')
    # angle
    run_data_produce(folder)
    # split data
    run_data_split(folder)
    # modle predict
    run_predict(folder)
    
    plot_trajectory(folder)