import cv2
import time
import numpy as np
from loop import *
from ultralytics import YOLO
import torch
import os
from tools.Deadlift_tool.interpolate import run_interpolation
from tools.Deadlift_tool.bar_data_produce import run_bar_data_produce
from tools.Deadlift_tool.data_produce import run_data_produce
from tools.Deadlift_tool.data_split import run_data_split
from tools.Deadlift_tool.predict import run_predict
from tools.trajectory import plot_trajectory

def model_init():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    bar_model = YOLO("./model/deadlift/bar/best.pt").to(device)
    bone_model = YOLO("./model/deadlift/skeleton/yolo11s-pose.pt").to(device)

    # 預先跑一次，避免 multi-thread 時出現 fuse 的錯誤
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    with torch.no_grad():
        bar_model.predict([dummy])
        bone_model.predict([dummy])

    return bar_model, bone_model

def rc_prep(folder):
    visions = ['bar', 'left-front', 'left-back']
    bar_file = open(os.path.join(folder, 'coordinates.txt'), "w")
    outs = [None] * 3
    skeleton_files = [None] * 3
    for idx, vision  in enumerate(visions):
        if idx == 0:
            file = os.path.join(folder, f'vision{idx+1}_drawed.avi')
        else:
            file = os.path.join(folder, f'vision{idx+1}_skeleton.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用 mp4v 編碼
        frame_size = (480, 640)  # 幀大小 (width, height)
        out = cv2.VideoWriter(file, fourcc, 29, frame_size)
        skeleton_file = open(os.path.join(folder, f'skeleton_{vision}.txt'), "w")
        outs[idx] = out
        skeleton_files[idx] = skeleton_file
    return outs, bar_file, skeleton_files

def predict(folder):
    os.makedirs(f'{folder}/config', exist_ok=True)
    # 對槓端及骨架做內插
    first = time.time()
    run_interpolation(folder)
    print("run_interpolation time :", time.time()-first)
    
    # bar
    first = time.time()
    run_bar_data_produce(folder, sport='deadlift')
    print("run_bar_data_produce time :", time.time()-first)
    
    # angle
    first = time.time()
    run_data_produce(folder)
    print("run_data_produce time :", time.time()-first)
    
    # split data
    first = time.time()
    run_data_split(folder)
    print("run_data_split time :", time.time()-first)
    
    first = time.time()
    plot_trajectory(folder)
    print("plot_trajectory time :", time.time()-first)
    
    first = time.time()
    run_predict(folder)
    print("run_predict time :", time.time()-first)

def pre_run(video_path):
    first_time = time.time()
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
    skeleton_data = {0:{},1:{},2:{}}
    bar_data = {}
    outs, bar_file, skeleton_files = rc_prep(video_path)
    print('Prepare for writing image')
    for i in range(3):
        if os.path.exists(f'{video_path}/vision{i+1}.mp4'):
            cap = cv2.VideoCapture(f'{video_path}/vision{i+1}.mp4')
        else:
            cap = cv2.VideoCapture(f'{video_path}/vision{i+1}.avi')
        if not cap.isOpened():
            print(f"Failed to open camera {i+1}")
            continue
        caps.append(cap)

    start_time = time.time()
    while True:
        all_done = True  # 假設全部都讀完了
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # resize to 640x480
            if not ret:
                print(f"Camera {i} has no more frames.")
                continue  # 這支相機讀不到就跳過

            all_done = False  # 有任何一支還在跑，就不跳出
            
            if i == 0:
                processed_frame, skeleton_data[i][frame_count_for_detect], bar_data[frame_count_for_detect] = bar_frame(
                    frame, bar_model,bone_model, skeleton_connections,bar_file, frame_count_for_detect)
            else:
                processed_frame, skeleton_data[i][frame_count_for_detect] = bone_frame(frame, bone_model,
                                            skeleton_connections,
                                            frame_count_for_detect)
                outs[i].write(processed_frame)

        frame_count_for_detect += 1
        # print('processing time per frame : ', time.time() - start_time)
        if all_done:
            print("All videos have been processed.")
            print("predict time :", time.time()-start_time)
            break
    
    # ✅ 儲存 skeleton_data（合併記憶體資料再寫入）
    for vision_index, frames in skeleton_data.items():
        buffer = []
        for f, data in sorted(frames.items()):
            buffer.extend(data)  # 每一幀的 list[str]
        if skeleton_files[vision_index]:  # 確保檔案存在
            skeleton_files[vision_index].writelines(buffer)

    # ✅ 儲存 bar_data
    bar_buffer = []
    for f, data in sorted(bar_data.items()):
        bar_buffer.extend(data)  # 一幀可能多筆 bar box 資料
    if bar_file:
        bar_file.writelines(bar_buffer)
        
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
    # 執行預測
    predict(video_path)
    print('processing time per video : ', time.time() - first_time)
if __name__ == "__main__":
    video_path = './recordings/recording_20250328_140412'
    pre_run(video_path)
