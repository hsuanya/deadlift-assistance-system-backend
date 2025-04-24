import threading
import queue
from ultralytics import YOLO
import torch
import loop
import os, cv2
import numpy as np
from datetime import datetime
import json


class Human_Vision:

    def __init__(self):
        self.bar_model, self.bone_model = self.model_init()
        dir = './'
        self.save_path = {
            'Deadlift': os.path.join(dir, 'recordings', 'deadlift'),
            'Benchpress': os.path.join(dir, 'recordings', 'benchpress'),
            'Squat': os.path.join(dir, 'recordings', 'squat')
        }

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
        self.outs = [None] * 3
        self.bar_files = [None] * 3
        self.skeleton_files = [None] * 3
        self.frame_count_for_detect = 0

    def clear_runtime_data(self):
        self.threads = []
        self.frames = queue.Queue()
        self.idxs = queue.Queue()

    def model_init(self):
        device_0 = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        device_1 = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        bar_model = YOLO("./model/deadlift/bar/best.pt").to(device_0)
        bone_model = YOLO("./model/deadlift/skeleton/yolov8n-pose.pt").to(
            device_1)

        # 預先跑一次，避免 multi-thread 時出現 fuse 的錯誤
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        with torch.no_grad():
            bar_model.predict(dummy)
            bone_model.predict(dummy)

        return bar_model, bone_model

    def create_thread(self, i, frame, rc_sig, id):
        print(rc_sig)
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
                self.bar_files = [None] * 3
                self.skeleton_files = [None] * 3
                for idx in range(3):
                    self.outs[idx], self.bar_files[idx], self.skeleton_files[
                        idx] = self.rc_prep(idx, frame, folder)
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
                self.bar_files = [None] * 3
                self.skeleton_files = [None] * 3
                self.frame_count_for_detect = 0

        self.recording_sig = rc_sig
        thread = threading.Thread(target=self.process_vision,
                                  args=(i, frame),
                                  daemon=True)
        self.threads.append(thread)

    def process_vision(self, i, frame):
        out = self.outs[i]
        bar_file = self.bar_files[i]
        skeleton_file = self.skeleton_files[i]
        if i == 0:
            frame = loop.bar_frame(frame, self.bar_model, self.barrier, out,
                                   bar_file, self.frame_count_for_detect)
        elif i == 1:
            frame = loop.bone_frame(frame, self.bone_model,
                                    self.skeleton_connections, self.barrier,
                                    out, skeleton_file,
                                    self.frame_count_for_detect)
        else:
            frame = loop.general_frame(frame, self.barrier, out)

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

    def rc_prep(self, idx, frame, folder):
        file = os.path.join(folder, f'vision{idx+1}.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frame_size = (frame.shape[0], frame.shape[1])  # 幀大小 (width, height)
        out = cv2.VideoWriter(file, fourcc, 5, frame_size)
        bar_file = open(os.path.join(folder, 'yolo_coordinates.txt'), "w")
        skeleton_file = open(os.path.join(folder, 'yolo_skeleton.txt'), "w")
        return out, bar_file, skeleton_file


def predict(folder):
    # 對槓端及骨架做內插
    os.system(f'python ./tools/Deadlift_tool/interpolate.py {folder}')
    # bar
    os.system(
        f'python ./tools/Deadlift_tool/bar_data_produce.py {folder} --out ./recordings/{folder}/config --sport deadlift'
    )
    # angle
    os.system(
        f'python ./tools/Deadlift_tool/data_produce.py {folder} --out ./recordings/{folder}/config'
    )
    # split data
    os.system(f'python ./tools/Deadlift_tool/data_split.py {folder}')
    # modle predict
    os.system(
        f'python ./tools/Deadlift_tool/predict.py {folder} --out ./recordings/{folder}/config'
    )
    os.system(f'python ./tools/trajectory.py {folder}')
