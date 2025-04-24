import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dir',type=str)
args = parser.parse_args()
dir = args.dir

# 載入影片和座標
video_path = os.path.join(dir, 'vision1.avi')
coordinates_path = os.path.join(dir, 'yolo_coordinates_interpolated.txt')
output_path = os.path.join(dir, 'vision1_drawed.avi')

def trajectory(video_path, coordinates_path, output_path):
    # 解析座標檔案
    coordinates = {}
    with open(coordinates_path, 'r') as file:
        for line in file:
            if line.strip():  # 確保不處理空行
                data = line.strip().split(',')  # 用逗號分隔
                frame_number = int(data[0])    # 第一欄是 frame_number
                x = float(data[1])             # 第二欄是 x 座標
                y = float(data[2])             # 第三欄是 y 座標
                coordinates[frame_number] = (int(x), int(y))  # 存入座標字典

    # 打開影片
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 設定軌跡視窗範圍
    trajectory_window = 40  # 顯示最近 N 幀點

    # 開始處理影片
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 取得視窗範圍內的軌跡點
        trajectory = []
        for i in range(frame_count - trajectory_window + 1, frame_count + 1):
            if i in coordinates:
                trajectory.append(coordinates[i])

        # 繪製每幀叉叉（漸層效果）
        for i in range(len(trajectory)):
            # 計算顏色深度（漸層：越早的幀越淺，越晚的幀越深）
            alpha = int(255 * (i + 1) / len(trajectory))  # 計算透明度比例
            color = (255, 255 - alpha, 255)  # 淺藍色到深藍色，沒有紅色分量

            # 繪製叉叉（兩條斜線）
            size = 6  # 叉叉的大小
            center = trajectory[i]  # 中心點位置
            cv2.line(frame, (center[0] - size, center[1] - size), (center[0] + size, center[1] + size), color, thickness=2)  # 左上到右下
            cv2.line(frame, (center[0] - size, center[1] + size), (center[0] + size, center[1] - size), color, thickness=2)  # 左下到右上

        # 寫入輸出影片
        out.write(frame)


    cap.release()
    out.release()
    print("output_path：", output_path)

trajectory(video_path, coordinates_path, output_path)