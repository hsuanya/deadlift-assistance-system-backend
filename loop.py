import time, os, cv2

def bar_frame(frame, model, barrier, fps, start_time):
    # fps 計算
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # 錄影開始
    # if recording_sig:
    #     if out is None:  # 初始化 VideoWriter
    #         file = os.path.join(folder, f'vision{i + 1}.avi')
    #         fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #         frame_size = (frame.shape[1], frame.shape[0])  # 幀大小 (width, height)
    #         out = cv2.VideoWriter(file, fourcc, 29, frame_size)
    #         print(f"Initialized VideoWriter for camera {i + 1}")
    #     out.write(frame)

    #     if txt_file is None:
    #         txt_file_path = os.path.join(folder, 'yolo_coordinates.txt')
    #         txt_file = open(txt_file_path, "w")  # ✅ 錄影開始時開啟檔案
    #         frame_count_for_detect = 0  # ✅ 只在錄影開始時歸零
    #         print(f"Started writing data to {txt_file_path}")

    # frame 處理
    results = model(source=frame, imgsz=320, conf=0.5, verbose=False)
    boxes = results[0].boxes
    detected = False
    for result in results:
        frame = result.plot()
    
    # write result
    # if recording_sig or txt_file is not None:
    #     for box in boxes.xywh:
    #         detected = True
    #         x_center, y_center, width, height = box
    #         frame_count_for_detect += 1
    #         txt_file.write(f"{frame_count_for_detect},{x_center},{y_center},{width},{height}\n")
            
    #     if not detected:
    #         frame_count_for_detect += 1
    #         txt_file.write(f"{frame_count_for_detect},no detection\n")

    barrier.wait()
    # if not recording_sig:
    #     frame_count_for_detect = 0
    #     # 錄影結束
    #     if save_sig and out is not None:
    #         out.release()
    #         print(f"Released VideoWriter for camera {i + 1}")
    #         save_sig = False
    #     out = None
    #     if txt_file is not None:
    #         txt_file.close()
    #         txt_file = None  # ✅ 確保 `txt_file` 被正確關閉
    #         print(f"Closed txt_file for camera {i + 1}")

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb, fps, frame_count

def bone_frame(frame, model, barrier, fps, start_time, frame_count, skeleton_connections):
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # ✅ 錄影開始
    # if recording_sig:
    #     if out is None:
    #         file = os.path.join(folder, f'vision{i + 1}.avi')
    #         fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #         frame_size = (frame.shape[1], frame.shape[0])
    #         out = cv2.VideoWriter(file, fourcc, 29, frame_size)
    #         print(f"Initialized VideoWriter for camera {i + 1}")
    #     out.write(frame)

    #     if txt_file is None:
    #         txt_file_path = os.path.join(folder, 'mediapipe_landmarks.txt')
    #         txt_file = open(txt_file_path, "w")
    #         frame_count_for_detect = 0
    #         print(f"Started writing data to {txt_file_path}")

    # ✅ YOLO 偵測骨架
    results = list(model(source=frame, stream=True, verbose=False))
    frame_count_for_detect += 1

    if results and results[0].keypoints:  # ✅ 確保有偵測到人
        r2 = results[0]  # ✅ 只取第一個偵測結果
        keypoints = r2.keypoints
        kpts = keypoints[0]  # ✅ 只取第一個人的骨架點
        keypoints_xy = kpts.xy  # shape: (1, 17, 2) -> 17 個關鍵點

        # ✅ 過濾無效骨架點 (0,0)
        kp_coords = []
        frame_data = []  # 存放該幀的骨架點
        for idx, kp in enumerate(keypoints_xy[0]):
            x_kp, y_kp = int(kp[0].item()), int(kp[1].item())

            # ✅ 若骨架點為 (0,0)，則標記為 None（不畫）
            if x_kp == 0 and y_kp == 0:
                kp_coords.append(None)
            else:
                kp_coords.append((x_kp, y_kp))
                cv2.circle(frame, (x_kp, y_kp), 5, (0, 255, 0), cv2.FILLED)
            
            frame_data.append(f"{frame_count_for_detect},{idx},{x_kp},{y_kp}")

        # ✅ 繪製骨架連線，若其中一個點為 None，則不畫線
        for start_idx, end_idx in skeleton_connections:
            if start_idx < len(kp_coords) and end_idx < len(kp_coords):
                if kp_coords[start_idx] is None or kp_coords[end_idx] is None:
                    continue
                cv2.line(frame, kp_coords[start_idx], kp_coords[end_idx], (0, 255, 255), 2)

        # ✅ **將骨架點資訊寫入 `txt_file`**
        # if txt_file is not None:
        #     txt_file.write("\n".join(frame_data) + "\n")

    # else:
    #     # ❌ **沒有偵測到人，寫入 "no detection"**
    #     if txt_file is not None:
    #         txt_file.write(f"{frame_count_for_detect},no detection\n")

    # ✅ **錄影完全結束後才關閉 `txt_file`**
    barrier.wait()
    # if not recording_sig:
    #     if txt_file is not None:
    #         txt_file.close()
    #         txt_file = None
    #         print(f"Closed txt_file for camera {i + 1}")

    #     if out is not None:
    #         out.release()
    #         out = None
    #         print(f"Released VideoWriter for camera {i + 1}")
    #     save_sig = False

    # ✅ 繪製 FPS
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb, fps, frame_count


def general_frame(frame, barrier, fps, start_time, frame_count):
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
        
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # 錄影開始
    # if recording_sig:
    #     if out is None:  # 初始化 VideoWriter
    #         file = os.path.join(folder, f'vision{i + 1}.avi')
    #         fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #         frame_size = (frame.shape[1], frame.shape[0])  # 幀大小 (width, height)
    #         out = cv2.VideoWriter(file, fourcc, 29, frame_size)
    #         print(f"Initialized VideoWriter for camera {i + 1}")
    #     out.write(frame)
    
    # 錄影結束    
    barrier.wait()
    # if not recording_sig:
    #     if out is not None:
    #         out.release()
    #         print(f"Released VideoWriter for camera {i + 1}")
    #     out = None
    
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb, fps, frame_count
 