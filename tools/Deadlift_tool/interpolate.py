import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
###寫一個把三次內插整合在一起的
#第一次:把mediapipe裡面沒偵測到的先內插補植
#第二次:把槓端的沒偵測到的內插補植
#第三次:把mediapipe資料的長度對齊yolo的長度

###處理mediapipe資料的no detection
def interpolate_landmarks(input_file, output_file):
    # 讀取TXT文件
    df = pd.read_csv(input_file, header=None, names=["frame", "landmark", "x", "y"])

    # 將 'no detection' 替換為 NaN
    df.replace('no detection', np.nan, inplace=True)

    # 將需要進行插值的列轉換為數值型
    df[['landmark', 'x', 'y']] = df[['landmark', 'x', 'y']].apply(pd.to_numeric, errors='coerce')

    # 取得所有 frame 的範圍
    all_frames = df['frame'].unique()

    # 準備一個空的列表來儲存最終結果
    interpolated_data = []

    # 對每個 frame 進行處理
    for frame in all_frames:
        current_frame_data = df[df['frame'] == frame]

        # 儲存該 frame 的所有資料
        if current_frame_data.empty:
            continue  # 如果沒有資料則跳過

        if 'no detection' in current_frame_data.values:
            # 如果有 'no detection' 的標記，儲存該行
            for landmark in range(0, 17):
                interpolated_data.append([frame, landmark, np.nan, np.nan])
            continue

        # 對 landmarks 11 到 24 進行插值
        for landmark in range(0, 17):
            landmark_data = current_frame_data[current_frame_data['landmark'] == landmark]

            if landmark_data.empty:
                # 如果當前 landmark 沒有資料，則進行內插
                prev_frame_data = df[(df['frame'] < frame) & (df['landmark'] == landmark)].iloc[-1] if not df[(df['frame'] < frame) & (df['landmark'] == landmark)].empty else None
                next_frame_data = df[(df['frame'] > frame) & (df['landmark'] == landmark)].iloc[0] if not df[(df['frame'] > frame) & (df['landmark'] == landmark)].empty else None
                
                if prev_frame_data is not None and next_frame_data is not None:
                    interp_x = (prev_frame_data['x'] + next_frame_data['x']) / 2
                    interp_y = (prev_frame_data['y'] + next_frame_data['y']) / 2
#                     interp_z = (prev_frame_data['z'] + next_frame_data['z']) / 2
                elif prev_frame_data is not None:
                    interp_x = prev_frame_data['x']
                    interp_y = prev_frame_data['y']
#                     interp_z = prev_frame_data['z']
                elif next_frame_data is not None:
                    interp_x = next_frame_data['x']
                    interp_y = next_frame_data['y']
#                     interp_z = next_frame_data['z']
                else:
                    interp_x = interp_y = interp_z = np.nan
                
                interpolated_data.append([frame, landmark, interp_x, interp_y])
            else:
                # 如果有資料，則直接取用
                landmark_row = landmark_data.iloc[0]
                interpolated_data.append([frame, landmark_row['landmark'], landmark_row['x'], landmark_row['y']])

    # 將結果轉換為 DataFrame
    output_df = pd.DataFrame(interpolated_data, columns=["frame", "landmark", "x", "y"])

    # 將結果保存到新的TXT文件，不輸出索引與標題行
    output_df.to_csv(output_file, index=False, header=False)

    print(f"已完成內插並將結果儲存至 {output_file}")

parser = argparse.ArgumentParser()
parser.add_argument('dir',type=str)
args = parser.parse_args()
dir = args.dir

# 設定要處理的文件
input_files = [os.path.join(dir, 'mediapipe_landmarks.txt')]  # 請替換為你的輸入檔名
output_files = [os.path.join(dir, 'mediapipe_landmarks_1st_interp.txt')]  # 對應的輸出檔名

# 對每個文件進行內插處理
for input_file, output_file in zip(input_files, output_files):
    interpolate_landmarks(input_file, output_file)

####處理yolo的no detection並且內插


def load_yolo_data(filename):
    # 讀取 YOLO 資料
    data = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                values = [float(x) for x in line.strip().split(',')]
                if len(values) == 5:  # YOLO 的資料應有 5 個欄位
                    data.append(values)
            except ValueError:
                print(f"Skipping line: {line}")
    return np.array(data)


def interpolate_missing_detections(yolo_data):
    frames = yolo_data[:, 0]
    x_center = yolo_data[:, 1]
    y_center = yolo_data[:, 2]
    widths = yolo_data[:, 3]
    heights = yolo_data[:, 4]

    # 找到有效檢測的幀索引
    valid_indices = np.where((x_center != -1) & (y_center != -1) & (widths != -1) & (heights != -1))[0]

    # 確保所有幀都有值
    all_frames = np.arange(int(frames.min()), int(frames.max()) + 1)

    # 使用有效檢測的幀進行插值
    interp_x_center = interp1d(frames[valid_indices], x_center[valid_indices], kind='linear', fill_value="extrapolate")
    interp_y_center = interp1d(frames[valid_indices], y_center[valid_indices], kind='linear', fill_value="extrapolate")
    interp_widths = interp1d(frames[valid_indices], widths[valid_indices], kind='linear', fill_value="extrapolate")
    interp_heights = interp1d(frames[valid_indices], heights[valid_indices], kind='linear', fill_value="extrapolate")

    # 對於所有幀進行插值
    interpolated_x_center = interp_x_center(all_frames)
    interpolated_y_center = interp_y_center(all_frames)
    interpolated_widths = interp_widths(all_frames)
    interpolated_heights = interp_heights(all_frames)

    # 組合內插後的資料
    interpolated_data = np.column_stack((all_frames, interpolated_x_center, interpolated_y_center, interpolated_widths, interpolated_heights))
    return interpolated_data

# 載入 YOLO 資料
yolo_data = load_yolo_data(os.path.join(dir, 'yolo_coordinates.txt'))

# 內插 YOLO 中的 "no detection" 資料
interpolated_yolo_data = interpolate_missing_detections(yolo_data)
yolo_interpolated_path = os.path.join(dir, 'yolo_coordinates_interpolated.txt')
# 將結果保存到新的檔案，第一列為整數，其他為浮點數
np.savetxt(yolo_interpolated_path, interpolated_yolo_data, delimiter=',', fmt='%d,%.8f,%.8f,%.8f,%.8f')

print("YOLO 'no detection' frames have been interpolated and saved to 'yolo_coordinates_interpolated.txt'.")

####這邊先把mediapipe資料內插成yolo的數量，因yolo資料量通常較長

# 讀取 YOLO 的資料
yolo_data = np.loadtxt(yolo_interpolated_path, delimiter=',')
yolo_frames = yolo_data[:, 0]  # frame numbers

# 讀取第一份 MediaPipe 的資料
mediapipe_data_1 = np.loadtxt(os.path.join(dir, 'mediapipe_landmarks_1st_interp.txt'), delimiter=',')
landmarks_1 = np.unique(mediapipe_data_1[:, 1])  # unique landmark numbers

def interpolate_mediapipe(yolo_frames, mediapipe_data, landmarks):
    interpolated_data = []
    # original_data = []
    
    for landmark in landmarks:
        # 獲取該 landmark 的座標
        group = mediapipe_data[mediapipe_data[:, 1] == landmark]
        frames = group[:, 0]
        x_coords = group[:, 2]
        y_coords = group[:, 3]
#         z_coords = group[:, 4]

        # original_data.append((frames, x_coords, y_coords, z_coords))


        # 將 MediaPipe 的第一幀對應到 YOLO 的第一幀，最後一幀對應到 YOLO 的最後一幀
        mediapipe_first_frame = frames[0]
        mediapipe_last_frame = frames[-1]

        # 將 YOLO 的幀範圍內的第一幀和最後一幀對應到 MediaPipe 的幀範圍
        aligned_frames = np.linspace(yolo_frames[0], yolo_frames[-1], len(frames))

        # 使用 interp1d 進行內插

        interp_x = interp1d(aligned_frames, x_coords, kind='linear')
        interp_y = interp1d(aligned_frames, y_coords, kind='linear')
#         interp_z = interp1d(aligned_frames, z_coords, kind='linear')

        
        interp_x = interp1d(aligned_frames, x_coords, kind='linear', fill_value="extrapolate")
        interp_y = interp1d(aligned_frames, y_coords, kind='linear', fill_value="extrapolate")
#         interp_z = interp1d(aligned_frames, z_coords, kind='linear', fill_value="extrapolate")
        
        # 使用 YOLO 的幀數範圍內進行內插
        interpolated_x = interp_x(yolo_frames)
        interpolated_y = interp_y(yolo_frames)
#         interpolated_z = interp_z(yolo_frames)
        
        # 將內插的資料添加到列表中
        for i, frame in enumerate(yolo_frames):
            interpolated_data.append([frame, landmark, interpolated_x[i], interpolated_y[i]])
    
    return interpolated_data

# 處理第一份 MediaPipe 資料
interpolated_data_1 = interpolate_mediapipe(yolo_frames, mediapipe_data_1, landmarks_1)

output_mediapipe = os.path.join(dir, 'interpolated_mediapipe_landmarks_1.txt')
# 將內插後的資料寫入 TXT 檔案
with open(output_mediapipe, 'w') as f:
    for entry in interpolated_data_1:
        f.write(f"{int(entry[0])},{int(entry[1])},{int(entry[2])},{int(entry[3])}\n")
