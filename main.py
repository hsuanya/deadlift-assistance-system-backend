import asyncio
from datetime import datetime
from typing import Union
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse
import random
import requests
import json
import ast
import re
import cv2
import numpy as np
import base64
import os
import subprocess

from compute_frame import Human_Vision, predict
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
id = 1001


@app.get("/detection_result/{item_id}")
def get_detection_result(item_id: int):
    with open("recordings.json", mode='r', encoding='utf-8') as json_file:
        recordings = json.load(json_file)
        folder = recordings[str(item_id)]["recording_folder"]

    predict(folder)

    with open(f"{folder}/config/Score.json", mode='r',
              encoding='utf-8') as json_file:
        data = json.load(json_file)['results']
    result = {'id': str(item_id), 'score': {}}
    for key, item in data.items():
        result['score'][key] = {}
        result['score'][key]['total_score'] = item[0]

        values = [item[1][0][1], item[1][1][1], item[1][2][1], item[1][3][1]]
        result['score'][key]['away_from_the_shins'] = values[0]
        result['score'][key]['hips_rise_before_barbell'] = values[1]
        result['score'][key]['colliding_with_the_knees'] = values[2]
        result['score'][key]['lower_back_rounding'] = values[2]

        index_max = max(range(len(values)), key=values.__getitem__)
        if values[index_max] < 70:
            result['score'][key]['error'] = '動作正確'
        else:
            error = ["槓鈴遠離小腿", "臀部先上升", "槓鈴繞膝蓋", "駝背"]
            result['score'][key]['error'] = error[index_max]

    with open("recordings.json", mode='w', encoding='utf-8') as json_file:
        recordings[str(item_id)]['error'] = result['score'][key]['error']
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


@app.get("/video/{item_id}/{vision_index}")
async def get_video(item_id: int, vision_index: int):
    with open("recordings.json", mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        folder = data[str(item_id)]["recording_folder"]

    video_name = f"vision{vision_index}_drawed" if vision_index == 1 else f"vision{vision_index}"
    mp4_path = os.path.join(folder, f"{video_name}.mp4")

    count = 0
    while (not os.path.exists(os.path.join(folder, f"{video_name}.avi"))):
        if (count == 10): break
        count += 1
        await asyncio.sleep(1)

    # 轉檔為 MP4
    if not os.path.exists(mp4_path):
        ffmpeg_cmd = [
            "ffmpeg", "-i",
            os.path.join(folder, f"{video_name}.avi"), "-c:v", "libx264",
            "-c:a", "aac", "-strict", "experimental", mp4_path
        ]
        subprocess.run(ffmpeg_cmd, check=True)

    print(mp4_path)
    return FileResponse(mp4_path, media_type="video/mp4")


@app.get("/feedback/{item_id}")
def read_feedback(item_id: int):
    # return {'result': 'result'}
    with open("recordings.json", mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        error = data[str(item_id)]["error"]

    if error == "動作正確":
        prompt = f"""你是一個健身教練，如果有個學員在做硬舉時動作非常正確，你會怎麼鼓勵他？ 請用 markdown 語法回答，不要加任何多餘的內容"""
    else:
        prompt = f"""你是一個健身教練，如果有個學員在做硬舉時發生了錯誤--{error}，請問要怎麼修正他的動作？ 請用 markdown 語法回答，不要加任何多餘的內容"""
    result = api_func.get_openai_response(prompt)
    return {'result': result}


@app.get("/workout_plan/{item_id}")
def read_workout_plan(item_id: int):
    # return {'result': 'result'}
    with open("recordings.json", mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        error = data[str(item_id)]["error"]

    if error == "動作正確":
        prompt = f"""你是一個健身教練，如果有個學員在做硬舉時動作非常正確，你會建議他怎麼安排訓練菜單？ 請用 markdown 語法回答"""
    else:
        prompt = f"""你是一個健身教練，如果有個學員在做硬舉時發生了錯誤--{error}，你會建議他怎麼安排訓練菜單？ 請用 markdown 語法回答"""
    result = api_func.get_openai_response(prompt)
    return {'result': result}


@app.get("/video_list/{item_id}")
def read_video_list(item_id: int):
    # return {'result': []}
    with open("recordings.json", mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        error = data[str(item_id)]["error"]
    with open("video_list.json", mode='r', encoding='utf-8') as json_file:
        movement = json.load(json_file)

    movement_list = ['平板支撐', '臀推', '深蹲', '傳統硬舉', '羅馬尼亞硬舉', '臥推', '引體向上', '肩推']

    if error == "動作正確":
        prompt = """你是一個健身教練，如果有個學員在做硬舉時動作非常正確，你會建議他這些動作裡的哪些%s？請回我一個格式一樣的 list of string""" % ' '.join(
            movement_list)
    else:
        prompt = """你是一個健身教練，如果有個學員在做硬舉時發生了錯誤--%s，你會建議他這些動作裡的哪些%s？請回我一個格式一樣的 list of string""" % (
            error, ' '.join(movement_list))
    videos = api_func.get_openai_response(prompt)
    # 用正則抓中括號內部的資料
    match = re.search(r'\[.*\]', videos, re.DOTALL)
    list_text = match.group(0)
    # 用 ast.literal_eval 將字串轉為真正的 list of dict
    video_list = ast.literal_eval(list_text)
    result = []
    for text in video_list:
        if text in movement_list:
            video = movement[text][0]
            video["video_id"] = video['url'].split("v=")[-1]
            result.append(video)

    # result = []
    # for video in video_list:
    #     oembed_url = f"https://www.youtube.com/oembed?url={video['url']}&format=json"
    #     response = requests.get(oembed_url)
    #     if response.status_code == 200:
    #         video["video_id"] = video['url'].split("v=")[-1]
    #         result.append(video)
    return {'result': result}


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
        messages = json.loads(data)

        for message in messages:
            img_data = base64.b64decode(message['image'])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cam_index = message['index']

            # cv2.imwrite(f'./images/{cam_index}.jpg', frame)
            human_vision.create_thread(cam_index, frame, is_recording, id)

        frames, idxs = human_vision.get_frame()

        for frame, idx in zip(frames, idxs):
            _, buffer = cv2.imencode('.jpg', frame)
            processed_b64 = base64.b64encode(buffer).decode('utf-8')
            result.append({
                "index": idx,
                "image": processed_b64,
            })
        await websocket.send_text(json.dumps(result))
