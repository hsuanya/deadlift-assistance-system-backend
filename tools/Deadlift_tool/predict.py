import torch.nn as nn
import torch
import argparse
import os, glob, json
import numpy as np


class ResidualBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 downsample=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels,
                               out_channels,
                               kernel_size,
                               stride,
                               padding=kernel_size // 2,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels,
                               out_channels,
                               kernel_size,
                               stride=1,
                               padding=kernel_size // 2,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm1d(out_channels))

    def forward(self, x):
        identity = self.shortcut(x)  # 短路連接
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # 殘差連接
        out = torch.relu(out)
        return out


class ResNet32(nn.Module):

    def __init__(self, input_dim, num_classes=2):
        super(ResNet32, self).__init__()
        self.initial = nn.Conv1d(input_dim,
                                 64,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 bias=False)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

        # ResNet-32 主要包含 5 個 Block（共 32 層）
        self.layer1 = self.make_layer(64, 64, 5)
        self.layer2 = self.make_layer(64, 128, 5, downsample=True)
        self.layer3 = self.make_layer(128, 256, 5, downsample=True)
        self.layer4 = self.make_layer(256, 512, 5, downsample=True)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # 壓縮時間維度
        self.fc = nn.Linear(512, num_classes)  # 最終分類

    def make_layer(self,
                   in_channels,
                   out_channels,
                   num_blocks,
                   downsample=False):
        layers = [
            ResidualBlock(in_channels, out_channels, downsample=downsample)
        ]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.initial(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_avg_pool(x)  # [batch, channels, 1]
        x = x.squeeze(-1)  # 去掉最後的 1 維
        x = self.fc(x)  # 最終分類
        return x


def merge_data(folder):
    features = []
    delta_path = os.path.join(folder, 'filtered_delta_norm')
    delta2_path = os.path.join(folder, 'filtered_delta2_norm')
    square_path = os.path.join(folder, 'filtered_delta_square_norm')
    zscore_path = os.path.join(folder, 'filtered_zscore_norm')
    orin_path = os.path.join(folder, 'filtered_norm')

    if not all(
            map(os.path.exists,
                [delta_path, delta2_path, zscore_path, square_path, orin_path
                 ])):
        print(f"Missing data in {folder}")
        return

    deltas = glob.glob(os.path.join(delta_path, '*.txt'))
    delta2s = glob.glob(os.path.join(delta2_path, '*.txt'))
    squares = glob.glob(os.path.join(square_path, '*.txt'))
    zscores = glob.glob(os.path.join(zscore_path, '*.txt'))
    orins = glob.glob(os.path.join(orin_path, '*.txt'))

    data_per_ind = list(fetch(zip(deltas, delta2s, zscores, squares,
                                  orins)))  # Ensure list output
    features.extend(data_per_ind)
    return features


def fetch(uds):
    data_per_ind = []
    for ud in uds:
        parsed_data = []
        for file in ud:
            with open(file, 'r') as f:
                lines = f.read().strip().split('\n')
                parsed_data.append(
                    [list(map(float, line.split(','))) for line in lines])

        for num in zip(*parsed_data):
            data_per_ind.append([item for sublist in num for item in sublist])
            if len(data_per_ind) == 110:
                yield torch.tensor(data_per_ind,
                                   dtype=torch.float32)  # 確保是 Tensor
                data_per_ind = []


def predict(model, feature):
    if not isinstance(feature, torch.Tensor):
        feature = torch.as_tensor(feature, dtype=torch.float32)
    feature = feature.clone().detach().to(device)  # ✅ 修正方式
    feature = feature.unsqueeze(0)  # 增加 batch 維度
    with torch.no_grad():
        output = model(feature)  # 獲取模型輸出
        predicted_class = torch.argmax(output, dim=1)  # 取得最大信心值的類別
        confidence_scores = torch.softmax(output, dim=1)  # 計算分類信心值
    return predicted_class.item(), confidence_scores.cpu().numpy()


def save_to_config(y_data, output_file):
    # 遍歷數據並將 float32 轉為 Python 原生 float
    def convert(o):
        if isinstance(o, (torch.Tensor, np.ndarray)):
            return o.tolist()  # 轉換為 Python list
        elif isinstance(o, np.float32):
            return float(o)  # 轉換為 Python float
        elif isinstance(o, torch.float32):
            return float(o)  # 轉換為 Python float
        return o

    config_data = {"results": y_data}  # 儲存 key

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4, default=convert)


parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
parser.add_argument('--out', type=str)
args = parser.parse_args()
dir = args.dir
out = args.out
output_file = os.path.join(out, 'Score.json')

data = {}
category = {
    '2': 'Wrong feet position',
    '3': 'Butt fly',
    '4': 'Skip Knee',
    '5': 'Hunchback'
}
results = {}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_path = os.path.join(dir, 'data_norm')
features = merge_data(data_path)
for num, name in category.items():
    model = ResNet32(input_dim=25)
    state_dict = torch.load(
        f"./model/deadlift/Pscore/{str(num)}/ResNet32_Model.pth",
        map_location=device,
        weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    # 把每一下的結果丟入模型
    for i, feature in enumerate(features):
        if f"{i}" not in results:
            results[f"{i}"] = []  # 初始化 key
        pred, conf = predict(model, feature)
        result = (pred, conf[0])
        results[f'{i}'].append(result)

rounded_results = {}
score = {}

for key, values in results.items():
    rounded_values = []
    decs = []
    for pred, conf in values:
        rounded_conf = [round(c, 4) for c in conf]  # 保留到小數第 4 位
        decs.append(rounded_conf[1])
        rounded_values.append(rounded_conf)
    score = 1 - sum(decs) * 0.25
    rounded_results[key] = (score, rounded_values)

save_to_config(rounded_results, output_file)
