import time, os, cv2
import torch


def bar_frame(frame,
              model,
              barrier,
              out,
              txt_file,
              frame_count_for_detect,
              target_fps=10):
    # fps 計算
    start_time = time.time()

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # 錄影開始
    cond = txt_file is not None and out is not None
    if cond:
        # waste_time(start_time, target_fps)
        out.write(frame)
        print(f"Started writing video(bar)")

    # frame 處理
    results = model(source=frame, imgsz=320, conf=0.5, verbose=False)
    boxes = results[0].boxes
    detected = False
    for result in results:
        frame = result.plot()

    # write result
    if txt_file is not None:
        for box in boxes.xywh:
            detected = True
            x_center, y_center, width, height = box
            txt_file.write(
                f"{frame_count_for_detect},{x_center},{y_center},{width},{height}\n"
            )

        if not detected:
            frame_count_for_detect += 1
            txt_file.write(f"{frame_count_for_detect},no detection\n")

    # barrier.wait()
    return frame


def bone_frame(frame,
               model,
               skeleton_connections,
               barrier,
               out,
               txt_file,
               frame_count_for_detect,
               target_fps=10):
    # fps 計算
    start_time = time.time()
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # 錄影開始
    cond = txt_file is not None and out is not None
    if cond:
        # waste_time(start_time, target_fps)
        out.write(frame)
        print(f"Started writing video(bone)")

    # ✅ YOLO 偵測骨架
    results = list(model(source=frame, stream=True, verbose=False))
    # frame_count_for_detect += 1
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
                cv2.line(frame, kp_coords[start_idx], kp_coords[end_idx],
                         (0, 255, 255), 2)

        # ✅ **將骨架點資訊寫入 `txt_file`**
        if txt_file is not None:
            txt_file.write("\n".join(frame_data) + "\n")

    else:
        # ❌ **沒有偵測到人，寫入 "no detection"**
        if txt_file is not None:
            txt_file.write(f"{frame_count_for_detect},no detection\n")

    # barrier.wait()
    return frame


def general_frame(frame, barrier, out, target_fps=10):
    # fps 計算
    start_time = time.time()
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    # 錄影開始
    cond = out is not None
    if cond:
        # waste_time(start_time, target_fps)
        out.write(frame)
        print(f"Started writing video")

    # 錄影結束
    # barrier.wait()
    return frame


def waste_time(start_time, target_fps):
    # 等待剩餘時間以達成 target_fps
    elapsed = time.time() - start_time
    frame_duration = 1 / target_fps
    wait_time = frame_duration - elapsed
    if wait_time > 0:
        end_time = time.time()
        while time.time() - end_time < wait_time:
            pass  # 忙等待(busy wait)，也可用 time.sleep(wait_time) 替代
