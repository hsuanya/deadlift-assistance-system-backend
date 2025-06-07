import cv2
import time
import numpy as np
from loop import *
from ultralytics import YOLO
import torch
import os

def model_init():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    bar_model = YOLO("./model/best.pt").to(device)
    bone_model = YOLO("./model/yolo11s-pose.pt").to(device)

    # 預先跑一次，避免 multi-thread 時出現 fuse 的錯誤
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    with torch.no_grad():
        bar_model.predict([dummy])
        bone_model.predict([dummy])

    return bar_model, bone_model

def rc_prep(folder):
    bar_file = open(os.path.join(folder, 'yolo_coordinates.txt'), "w")
    outs = [None] * 3
    skeleton_files = [None] * 3
    for idx  in range (3):
        file = os.path.join(folder, f'vision{idx+1}_skeleton.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 編碼
        frame_size = (480, 640)  # 幀大小 (width, height)
        out = cv2.VideoWriter(file, fourcc, 30, frame_size)
        skeleton_file = open(os.path.join(folder, f'yolo_skeleton_vision{idx+1}.txt'), "w")
        outs[idx] = out
        skeleton_files[idx] = skeleton_file
    return outs, bar_file, skeleton_files

def main():
    video_path = './recordings/recording_20250328_140019'
    bar_model, bone_model = model_init()
    skeleton_connections = [
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
    frame_count_for_detect = 0
    caps = []
    outs = [None] * 3
    skeleton_files = [None] * 3
    outs, bar_file, skeleton_files = rc_prep(video_path)
    print('Prepare for writing image')
    for i in range(3):
        cap = cv2.VideoCapture(f'{video_path}/vision{i+1}.avi')
        if not cap.isOpened():
            print(f"Failed to open camera {i+1}")
            continue
        caps.append(cap)

    while True:
        all_done = True  # 假設全部都讀完了
        start_time = time.time()
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                print(f"Camera {i} has no more frames.")
                continue  # 這支相機讀不到就跳過

            all_done = False  # 有任何一支還在跑，就不跳出
            
            if i == 0:
                processed_frame = bar_frame(frame, bar_model,
                                            bone_model, skeleton_connections,
                                            outs[i], skeleton_files[i],
                                            bar_file, frame_count_for_detect)
            else:
                processed_frame = bone_frame(frame, bone_model,
                                            skeleton_connections,
                                            outs[i], skeleton_files[i],
                                            frame_count_for_detect)
            outs[i].write(processed_frame)

        frame_count_for_detect += 1
        print('processing time per frame : ', time.time() - start_time)
        if all_done:
            print("All videos have been processed.")
            break
        
    for cap in caps:
        cap.release()
    for out in outs:
        if out:
            out.release()
    if bar_file:
        bar_file.close()
    for file in skeleton_files:
        if file:
            file.close()
            
if __name__ == "__main__":
    main()
