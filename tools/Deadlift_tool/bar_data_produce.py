import json, os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
parser.add_argument('--out', type=str)
parser.add_argument('--sport', type=str)
args = parser.parse_args()
dir = args.dir
out = args.out
sport = args.sport
# 讀取 yolo 檔案
yolo_txt_path = os.path.join(
    dir, "yolo_coordinates_interpolated.txt")  # 你的 txt 檔案路徑
if sport == 'deadlift':
    output_json_path = os.path.join(out, "Bar_Position.json")  # 輸出的 JSON 檔案
elif sport == 'benchpress':
    output_json_path = os.path.join(out, 'Benchpress_data',
                                    "Bar_Position.json")  # 輸出的 JSON 檔案

# 初始化數據存儲
frames = []
values = []

# 讀取 YOLO 偵測數據
with open(yolo_txt_path, "r") as file:
    for line in file:
        parts = line.strip().split(",")
        if len(parts) < 3:  # 確保資料完整
            continue

        frame_count = int(parts[0])  # 幀數
        x_center = float(parts[1])  # X 中心

        frames.append(frame_count)
        values.append(x_center)

if values:
    x_min = min(values) * 0.9  # X 軸最小值，留 10% 緩衝
    x_max = max(values) * 1.1  # X 軸最大值，留 10% 緩衝
else:
    x_min = x_max = 0

# 轉換成 JSON 格式
data = {
    "title": "Barbell Center Positions",
    "y_label": "Position (pixels)",
    "y_min": x_min,
    "y_max": x_max,
    "frames": frames,
    "values": values
}

# 存成 JSON 檔案
with open(output_json_path, "w") as json_file:
    json.dump(data, json_file, indent=4)

print(f"✅ JSON 檔案已儲存: {output_json_path}")
