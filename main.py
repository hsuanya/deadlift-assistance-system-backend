import asyncio
from datetime import datetime
import shutil
from typing import List, Union
from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from filelock import FileLock
from sse_starlette.sse import EventSourceResponse
import random
import requests
import json
import ast
import re
import cv2
import numpy as np
import base64
import os, time
import subprocess
import copy

from compute_frame import Human_Vision, predict
from pre_run import pre_run
import api_func

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 或指定 frontend 網域，如 "http://localhost:3000"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

is_recording = False
id = 2000


@app.get("/detection_result/{item_id}")
def get_detection_result(item_id: int):
    with open("recordings.json", mode='r', encoding='utf-8') as json_file:
        recordings = json.load(json_file)
        folder = recordings[str(item_id)]["recording_folder"]

    if ('error' not in recordings[str(item_id)]):
        pre_run(folder)

    with open(f"{folder}/config/Score.json", mode='r', encoding='utf-8') as json_file:
        score_data = json.load(json_file)['results']
    with open(f"{folder}/config/Split_info.json", mode='r', encoding='utf-8') as json_file:
        split_data = json.load(json_file)
    
    if os.path.exists(os.path.join(folder, "vision2.avi")):
        cap = cv2.VideoCapture(os.path.join(folder, "vision2.avi"))
    else: 
        cap = cv2.VideoCapture(os.path.join(folder, "vision2.mp4"))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    result = {'id': str(item_id), 'total_frames': total_frames, 'score': {}}
    for id, (key, item) in enumerate(score_data.items()):
        result['score'][key] = {}
        result['score'][key]['start_frame'] = split_data[key]['start']
        result['score'][key]['end_frame'] = split_data[key]['end']
        result['score'][key]['total_score'] = item[0]

        values = [item[1][0][1], item[1][1][1], item[1][2][1], item[1][3][1]]
        result['score'][key]['away_from_the_shins'] = values[0]
        result['score'][key]['hips_rise_before_barbell'] = values[1]
        result['score'][key]['colliding_with_the_knees'] = values[2]
        result['score'][key]['lower_back_rounding'] = values[2]

        index_max = max(range(len(values)), key=values.__getitem__)
        if values[index_max] < 0.5:
            result['score'][key]['error'] = '動作正確'
        else:
            error = ["槓鈴距離小腿過遠", "臀部先上升", "槓鈴繞過膝蓋", "駝背"]
            result['score'][key]['error'] = error[index_max]

    error_config = copy.deepcopy(result['score'])
    for key, item in error_config.items():
        del error_config[key]['start_frame']
        del error_config[key]['end_frame']
        del error_config[key]['total_score']

    with open("recordings.json", mode='w', encoding='utf-8') as json_file:
        recordings[str(item_id)]['total_frames'] = total_frames
        recordings[str(item_id)]['error'] = error_config
        json.dump(recordings, json_file, indent=4)

    # error = ["動作正確", "槓鈴遠離小腿", "臀部先上升", "槓鈴繞膝蓋", "駝背"]
    # error_result = random.randint(0, 4)
    # confidence_level = {
    #     "away_from_the_shins":
    #     random.randint(0 if error_result != 1 else 50,
    #                    50 if error_result != 1 else 100) / 100.0,
    #     "Hips_rise_before_barbell":
    #     random.randint(0 if error_result != 2 else 50,
    #                    50 if error_result != 2 else 100) / 100.0,
    #     "colliding_with_the_knees":
    #     random.randint(0 if error_result != 3 else 50,
    #                    50 if error_result != 3 else 100) / 100.0,
    #     "lower_back_rounding":
    #     random.randint(0 if error_result != 4 else 50,
    #                    50 if error_result != 4 else 100) / 100.0,
    # }
    # result = {
    #     "id": str(item_id),
    #     "error": error[error_result],
    #     "confidence_level": confidence_level
    # }
    # try:
    #     with open("recordings.json", mode='r', encoding='utf-8') as json_file:
    #         data = json.load(json_file)
    # except:
    #     data = {}
    # data[item_id] = result
    # with open("recordings.json", mode='w', encoding='utf-8') as json_file:
    #     json.dump(data, json_file, indent=4)

    return result


@app.get("/graph/{item_id}")
def get_graph(item_id: int):
    with open("recordings.json", mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        folder = data[str(item_id)]["recording_folder"]
    file_names = ['Bar_Position', 'Hip_Angle', 'Knee_Angle', 'Knee_to_Hip']
    result = []
    for file_name in file_names:
        with open(f"{folder}/config/{file_name}.json",
                  mode='r',
                  encoding='utf-8') as json_file:
            data = json.load(json_file)
        result.append(data)
    return {'result': result}


# @app.get("/video/{item_id}/{vision_index}")
# async def get_video(item_id: int, vision_index: int):
#     with open("recordings.json", mode='r', encoding='utf-8') as json_file:
#         data = json.load(json_file)
#         folder = data[str(item_id)]["recording_folder"]

#     video_name = f"vision{vision_index}_drawed" if vision_index == 1 else f"vision{vision_index}"
#     mp4_path = os.path.join(folder, f"{video_name}.mp4")

#     count = 0
#     while (not os.path.exists(os.path.join(folder, f"{video_name}.avi"))):
#         if (count == 20): break
#         count += 1
#         await asyncio.sleep(1.5)

#     # 轉檔為 MP4
#     if not os.path.exists(mp4_path):
#         ffmpeg_cmd = [
#             "ffmpeg", "-i",
#             os.path.join(folder, f"{video_name}.avi"), "-c:v", "libx264",
#             "-c:a", "aac", "-strict", "experimental", mp4_path
#         ]
#         subprocess.run(ffmpeg_cmd, check=True)

#     print(mp4_path)
#     return FileResponse(mp4_path, media_type="video/mp4")

@app.get("/video/{item_id}/{vision_index}")
async def get_video(item_id: int, vision_index: int):
    # 讀取錄影資料夾
    with open("recordings.json", mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        folder = data[str(item_id)]["recording_folder"]

    video_name = f"vision{vision_index}_drawed" if vision_index == 1 else f"vision{vision_index}_skeleton"
    avi_path = os.path.join(folder, f"{video_name}.avi")
    mp4_path = os.path.join(folder, f"{video_name}.mp4")

    # 等待 AVI 出現
    count = 0
    while not os.path.exists(avi_path):
        if count == 100:
            break
        count += 1
        await asyncio.sleep(1.5)

    # 如果 mp4 已存在，就直接傳送
    if os.path.exists(mp4_path):
        return FileResponse(mp4_path, media_type="video/mp4")

    avi_to_mp4(folder, video_name)
    # 加鎖轉檔，避免多個請求同時轉同一個檔案
    # with FileLock(lock_path):
    #     # 二次確認轉檔後是否已存在，避免重複轉檔
    #     if not os.path.exists(mp4_path):
    #         # 轉成 temp 檔，轉完再 rename 為正式 mp4
    #         ffmpeg_cmd = [
    #             "ffmpeg", "-i", avi_path,
    #             "-c:v", "libx264", "-c:a", "aac",
    #             "-strict", "experimental", temp_path
    #         ]
    #         subprocess.run(ffmpeg_cmd, check=True)

    #         os.rename(temp_path, mp4_path)

    return FileResponse(mp4_path, media_type="video/mp4")

def avi_to_mp4(folder, video_name):
    avi_path = os.path.join(folder, f"{video_name}.avi")
    mp4_path = os.path.join(folder, f"{video_name}.mp4")
    lock_path = os.path.join(folder, f"{video_name}.lock")
    temp_path = os.path.join(folder, f"{video_name}.temp.mp4")

    # 加鎖轉檔，避免多個請求同時轉同一個檔案
    with FileLock(lock_path):
        # 二次確認轉檔後是否已存在，避免重複轉檔
        if not os.path.exists(mp4_path):
            # 轉成 temp 檔，轉完再 rename 為正式 mp4
            ffmpeg_cmd = [
                "ffmpeg", "-i", avi_path,
                "-c:v", "libx264", "-c:a", "aac",
                "-strict", "experimental", temp_path
            ]
            subprocess.run(ffmpeg_cmd, check=True)

            os.rename(temp_path, mp4_path)
    return mp4_path

@app.get("/feedback/{item_id}")
def read_feedback(item_id: int):
    # return {'result': 'result'}
    with open("recordings.json", mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        error = data[str(item_id)]["error"]

    feedback_prompt = f"""你是一個健身教練，這是我在做一組硬舉訓練時發生的錯誤--{error}，key為第幾組，value包含了4個錯誤動作的信心值可以輔助判斷(回答以信心值作為嚴重程度的標準，例如:可能、有一點、明顯等等，不要直接出現信心值)，越大表示該錯誤錯得越明顯，請問要怎麼修正他的動作，指出哪一下有問題。請用 markdown 語法回答，給我中文回答"""
    feedback_result = api_func.get_openai_response(feedback_prompt)

    return {'result': extract_markdown(feedback_result)}

def extract_markdown(llm_output: str) -> str:
    return re.sub(r'^```(?:markdown)?\n([\s\S]+?)\n```$', r'\1', llm_output.strip())

@app.get("/workout_plan/{item_id}")
def read_workout_plan(item_id: int):
    # return {'result': 'result'}
    with open("recordings.json", mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        error = data[str(item_id)]["error"]

    plan_prompt = f"""你是一個健身教練，這是我在做一組硬舉訓練時發生的錯誤--{error}，key為第幾組，value包含了4個錯誤動作分類模型的信心值(回答以信心值作為嚴重程度的標準，例如:可能、有一點、明顯等等，不要直接出現信心值)，你會建議他怎麼「安排一週詳細的訓練菜單」，越清楚越好。請用 markdown 語法回答"""    
    plan_result = api_func.get_openai_response(plan_prompt)
    return {'result': extract_markdown(plan_result)}


@app.get("/video_list/{item_id}")
def read_video_list(item_id: int):
    # return {'result': []}
    with open("recordings.json", mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        error = data[str(item_id)]["error"]
    with open("video_list.json", mode='r', encoding='utf-8') as json_file:
        movement = json.load(json_file)

    movement_list = ['平板支撐', '臀推', '深蹲', '傳統硬舉', '羅馬尼亞硬舉', '臥推', '引體向上', '肩推']
    recommend_video_prompt = """你是一個健身教練，這是我在做一組硬舉訓練時發生的錯誤--%s，你會建議我這些動作裡的哪些%s？請回我一個格式一樣的 list of string""" % (error, ' '.join(movement_list))
    times = 0
    while True:
        result = []
        try:
            videos = api_func.get_openai_response(recommend_video_prompt)
            print(videos)
            # 用正則抓中括號內部的資料
            match = re.search(r'\[.*\]', videos, re.DOTALL)
            list_text = match.group(0)
            # 用 ast.literal_eval 將字串轉為真正的 list of dict
            video_list = ast.literal_eval(list_text)
            for text in video_list:
                if text in movement_list:
                    video = movement[text][0]
                    video["video_id"] = video['url'].split("v=")[-1]
                    result.append(video)
            break
        except:
            times += 1
            if times >= 5:
                break
            continue

    # result = []
    # for video in video_list:
    #     oembed_url = f"https://www.youtube.com/oembed?url={video['url']}&format=json"
    #     response = requests.get(oembed_url)
    #     if response.status_code == 200:
    #         video["video_id"] = video['url'].split("v=")[-1]
    #         result.append(video)
    return {'result': result}

@app.post("/upload_record")
def upload_files(files: List[UploadFile] = File(...)):
    # 建立以 timestamp 命名的資料夾
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"recording_{timestamp}"
    folder_path = "./recordings/deadlift/" + folder_name
    os.makedirs(folder_path, exist_ok=True)

    saved_files = []
    for file in files:
        file_location = os.path.join(folder_path, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file_location)

        # # 轉檔為 MP4
        # if os.path.splitext(file_location)[1] == ".avi":
        #     print("############filename:"+os.path.splitext(file.filename)[0])
        #     avi_to_mp4(folder_path, os.path.splitext(file.filename)[0])

    global id
    with open("recordings.json", mode='r', encoding='utf-8') as json_file:
        recordings = json.load(json_file)
    recordings[id] = {}
    recordings[id]['recording_folder'] = folder_path
    with open("recordings.json", mode='w', encoding='utf-8') as json_file:
        json.dump(recordings, json_file,  indent=4)
    record_id = id
    id += 1
    return {"id": str(record_id)}

@app.get("/thumbnail/{item_id}")
def get_thumbnail(item_id: int):
    with open("recordings.json", mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        folder = data[str(item_id)]["recording_folder"]
    thumbnail_path = os.path.join(folder, "thumbnail.jpg")
    if not os.path.exists(thumbnail_path):
        success = extract_frame(os.path.join(folder, "vision1_drawed.mp4"), 0)
        if not success:
            raise HTTPException(status_code=404, detail="Thumbnail not found")
    
    return FileResponse(thumbnail_path, media_type="image/jpeg")

@app.get("/example_list")
async def example_list():
    with open("example_list.json", mode='r', encoding='utf-8') as json_file:
        ids = json.load(json_file)
    with open("recordings.json", mode='r', encoding='utf-8') as json_file:
        recordings = json.load(json_file)
    
    result = []
    for item in ids:
        if str(item) in recordings:
            if ('error' not in recordings[str(item)]):
                get_detection_result(item)
                for i in range(1, 4):
                    await get_video(item, i)
                with open("recordings.json", mode='r', encoding='utf-8') as json_file:
                    recordings = json.load(json_file)

            thumbnail_path = os.path.join(recordings[str(item)]['recording_folder'], "thumbnail.jpg")
            if not os.path.exists(thumbnail_path):
                success = extract_frame(os.path.join(recordings[str(item)]['recording_folder'], "vision1_drawed.mp4"), 0)
                if not success:
                    continue
                
            result.append({
                'id': str(item),
                'title': os.path.basename(recordings[str(item)]['recording_folder']),
                'total_set': len(recordings[str(item)].get('error', {})),
                'error': [v["error"] for v in recordings[str(item)].get('error', {}).values()]
            })

    return {'result': result}

def extract_frame(video_path, frame_number=0):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return False

    # 跳到指定的幀（預設第0幀）
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    if success:
        base_dir = os.path.dirname(video_path)
        save_path = os.path.join(base_dir, "thumbnail.jpg")
        cv2.imwrite(save_path, frame)
        cap.release()
    else:
        return False

    return True

@app.post("/record")
async def record(files: List[UploadFile] = File(...)):
    # 建立以 timestamp 命名的資料夾
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"recording_{timestamp}"
    folder_path = "./recordings/deadlift/" + folder_name
    os.makedirs(folder_path, exist_ok=True)

    mp4_files = []
    for i, file in enumerate(files):
        # 讀取 webm 的 bytes
        webm_bytes = await file.read()

        # 設定 mp4 輸出檔案名稱
        mp4_filename = f"vision{i + 1}.mp4"
        mp4_path = os.path.join(folder_path, mp4_filename)

        # 轉檔
        webm_bytes_to_mp4_file(webm_bytes, mp4_path)

        mp4_files.append(mp4_path)
    
    global id
    with open("recordings.json", mode='r', encoding='utf-8') as json_file:
        recordings = json.load(json_file)
    recordings[id] = {}
    recordings[id]['recording_folder'] = folder_path
    with open("recordings.json", mode='w', encoding='utf-8') as json_file:
        json.dump(recordings, json_file,  indent=4)
    record_id = id
    id += 1
    return {"id": str(record_id)}


def webm_bytes_to_mp4_file(webm_bytes: bytes, output_path: str):
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",                      # 覆蓋舊檔案
        "-i", "pipe:0",            # 從 stdin 讀入影片 bytes
        "-vf", "transpose=1,fps=29",  # 旋轉影片, Downsample to 30 fps
        "-c:v", "libx264",         # 使用 H.264 壓縮
        "-preset", "fast",         # 壓縮速度
        "-pix_fmt", "yuv420p",     # 確保瀏覽器相容性
        "-c:a", "aac",             # 音訊壓縮格式（若有音訊）
        "-strict", "experimental", # aac 所需
        output_path
    ]

    process = subprocess.run(
        ffmpeg_cmd,
        input=webm_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {process.stderr.decode()}")

    return output_path

@app.post("/start_record")
def start_record():
    global is_recording
    is_recording = True
    return {'result': True}


@app.post("/stop_record")
def start_record():
    global is_recording, id
    is_recording = False
    record_id = id
    id += 1
    return {'id': str(record_id)}


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    human_vision = Human_Vision()

    # count = 0
    while True:
        data = await websocket.receive_text()
        # print('receive_text', count)
        # count += 1
        result = []
        result_ts = []
        messages = json.loads(data)
        server_recv_ts = time.time() * 1000

        for message in messages:
            if "timestamp" in message:
                client_ts = message["timestamp"]
                continue
            img_data = base64.b64decode(message['image'])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cam_index = message['index']

            # cv2.imwrite(f'./images/{cam_index}.jpg', frame)
            human_vision.create_thread(cam_index, frame, is_recording, id)
        print(f'create_thread: ${time.time()* 1000 - server_recv_ts}')

        frames, idxs = human_vision.get_frame()
        print(f'get_frame: ${time.time()* 1000 - server_recv_ts}')

        for frame, idx in zip(frames, idxs):
            _, buffer = cv2.imencode('.jpg', frame)
            processed_b64 = base64.b64encode(buffer).decode('utf-8')
            result.append({
                "index": idx,
                "image": processed_b64,
            })
            result_ts = {
                'result':result,
                "timestamp": {
                    "timestamp": client_ts,
                    "server_recv_ts": server_recv_ts,
                    "server_process_end": time.time() * 1000
                }
            }
        print(f'total: ${time.time()* 1000 - server_recv_ts}')
        await websocket.send_text(json.dumps(result_ts))

# @app.websocket("/ws/stream")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     while True:
#         t0 = time.time()
#         data = await websocket.receive_bytes()
#         t1 = time.time()
#         print("[TIMER] Received data in:", round((t1 - t0) * 1000), "ms")

#         pointer = 0
#         results = []
#         t2 = time.time()
#         while pointer + 8 <= len(data):
#             idx = int.from_bytes(data[pointer:pointer + 4], byteorder='big')
#             pointer += 4

#             img_len = int.from_bytes(data[pointer:pointer + 4], byteorder='big')
#             pointer += 4

#             img_data = data[pointer:pointer + img_len]
#             pointer += img_len

#             # decode frame
#             nparr = np.frombuffer(img_data, np.uint8)
#             frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#             # process frame (dummy: return original image)
#             _, buffer = cv2.imencode('.jpg', frame)
#             encoded = buffer.tobytes()

#             # prepare response with index
#             result = idx.to_bytes(4, byteorder='big') + len(encoded).to_bytes(4, byteorder='big') + encoded
#             results.append(result)
#         t3 = time.time()
#         print("[TIMER] Decoded and encoded in:", round((t3 - t2) * 1000), "ms")

#         # 組合所有 processed frame 成為一包
#         await websocket.send_bytes(b''.join(results))
#         t4 = time.time()
#         print("[TIMER] Sent result in:", round((t4 - t3) * 1000), "ms")
#         print("[TOTAL] Full round took:", round((t4 - t0) * 1000), "ms")