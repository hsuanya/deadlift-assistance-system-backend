import time
import cv2
import numpy as np

def bar_frame(frame,
                bar_model,
                bone_model,
                skeleton_connections,
                bar_file,
                frame_count_for_detect):
    # frame 處理
    start_time = time.time()
    results = bar_model(source=frame, imgsz=320, conf=0.5, verbose=False, device="cuda:0")
    # print('bar predict :', time.time() - start_time)
    boxes = results[0].boxes
    detected = False
    for result in results:
        frame = result.plot()
    
    # write result
    bar_data = []
    # if bar_file is not None:
    for box in boxes.xywh:
        detected = True
        x_center, y_center, width, height = box
        bar_data.append(
            f"{frame_count_for_detect},{x_center},{y_center},{width},{height}\n"
        )

    if not detected:
        frame_count_for_detect += 1
        bar_data.append(f"{frame_count_for_detect},no detection\n")
        
    results = list(bone_model(source=frame, verbose=False, device="cuda:0"))
    skeleton_data = []  # 存放該幀的骨架點
    if results and results[0].keypoints:
        keypoints = results[0].keypoints
        frame_h, frame_w = frame.shape[:2]
        center_frame = np.array([frame_w / 2, frame_h / 2])

        min_dist = float('inf')
        target_kpts = None

        for kp in keypoints:
            coords = kp.xy[0].cpu().numpy()
            valid_coords = coords[(coords != 0).all(axis=1)]
            if len(valid_coords) == 0:
                continue
            person_center = np.mean(valid_coords, axis=0)
            dist = np.linalg.norm(person_center - center_frame)
            if dist < min_dist:
                min_dist = dist
                target_kpts = coords

        if target_kpts is not None:
            keypoints_xy = [target_kpts]
        else:
            keypoints_xy = []

        # ✅ 過濾無效骨架點 (0,0)
        kp_coords = []

        if keypoints_xy:
            for idx, kp in enumerate(keypoints_xy[0]):
                x_kp, y_kp = int(kp[0].item()), int(kp[1].item())

                # ✅ 若骨架點為 (0,0)，則標記為 None（不畫）
                if x_kp == 0 and y_kp == 0:
                    kp_coords.append(None)
                else:
                    kp_coords.append((x_kp, y_kp))

                skeleton_data.append(f"{frame_count_for_detect},{idx},{x_kp},{y_kp}\n")

            # ✅ 繪製骨架連線，若其中一個點為 None，則不畫線
            # for start_idx, end_idx in skeleton_connections:
            #     if start_idx < len(kp_coords) and end_idx < len(kp_coords):
            #         if kp_coords[start_idx] is None or kp_coords[end_idx] is None:
            #             continue
            #         cv2.line(frame, kp_coords[start_idx], kp_coords[end_idx],
            #                 (0, 255, 255), 2)

    else:
        skeleton_data.append(f"{frame_count_for_detect},no detection\n")
    return frame, skeleton_data, bar_data

def bone_frame(frame,
                model,
                skeleton_connections,
                frame_count_for_detect):
    # ✅ YOLO 偵測骨架
    start_count = time.time()
    results = list(model(source=frame, verbose=False, device="cuda:0"))
    skeleton_data = []  # 存放該幀的骨架點
    # print('bone predict :', time.time() - start_time)
    if results and results[0].keypoints:
        keypoints = results[0].keypoints
        frame_h, frame_w = frame.shape[:2]
        center_frame = np.array([frame_w / 2, frame_h / 2])

        min_dist = float('inf')
        target_kpts = None

        for kp in keypoints:
            coords = kp.xy[0].cpu().numpy()
            valid_coords = coords[(coords != 0).all(axis=1)]
            if len(valid_coords) == 0:
                continue
            person_center = np.mean(valid_coords, axis=0)
            dist = np.linalg.norm(person_center - center_frame)
            if dist < min_dist:
                min_dist = dist
                target_kpts = coords

        if target_kpts is not None:
            keypoints_xy = [target_kpts]
        else:
            keypoints_xy = []

        # ✅ 過濾無效骨架點 (0,0)
        kp_coords = []
        skeleton_data = []  # 存放該幀的骨架點
        if keypoints_xy:
            for idx, kp in enumerate(keypoints_xy[0]):
                x_kp, y_kp = int(kp[0].item()), int(kp[1].item())

                # ✅ 若骨架點為 (0,0)，則標記為 None（不畫）
                if x_kp == 0 and y_kp == 0:
                    kp_coords.append(None)
                else:
                    kp_coords.append((x_kp, y_kp))
                    cv2.circle(frame, (x_kp, y_kp), 5, (0, 255, 0), cv2.FILLED)

                skeleton_data.append(f"{frame_count_for_detect},{idx},{x_kp},{y_kp}\n")

            # ✅ 繪製骨架連線，若其中一個點為 None，則不畫線
            for start_idx, end_idx in skeleton_connections:
                if start_idx < len(kp_coords) and end_idx < len(kp_coords):
                    if kp_coords[start_idx] is None or kp_coords[end_idx] is None:
                        continue
                    cv2.line(frame, kp_coords[start_idx], kp_coords[end_idx],
                            (0, 255, 255), 2)

    else:
        skeleton_data.append(f"{frame_count_for_detect},no detection\n")
    return frame, skeleton_data
