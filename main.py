from datetime import datetime
from typing import Union
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import random
import requests
import json
import ast
import re
import cv2
import numpy as np
import base64

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
id = 1000


@app.get("/detection_result/{item_id}")
def get_detection_result(item_id: int):
    error = ["動作正確", "槓鈴遠離小腿", "臀部先上升", "槓鈴繞膝蓋", "駝背"]
    error_result = random.randint(0, 4)
    confidence_level = {
        "away_from_the_shins":
        random.randint(0 if error_result != 1 else 50,
                       50 if error_result != 1 else 100) / 100.0,
        "Hips_rise_before_barbell":
        random.randint(0 if error_result != 2 else 50,
                       50 if error_result != 2 else 100) / 100.0,
        "colliding_with_the_knees":
        random.randint(0 if error_result != 3 else 50,
                       50 if error_result != 3 else 100) / 100.0,
        "lower_back_rounding":
        random.randint(0 if error_result != 4 else 50,
                       50 if error_result != 4 else 100) / 100.0,
    }
    result = {
        "id": str(item_id),
        "error": error[error_result],
        "confidence_level": confidence_level
    }
    try:
        with open("recordings.json", mode='r', encoding='utf-8') as json_file:
            data = json.load(json_file)
    except:
        data = {}
    data[item_id] = result
    with open("recordings.json", mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)

    return result


@app.get("/graph/{item_id}")
def get_graph(item_id: int):
    pass


@app.get("/feedback/{item_id}")
def read_feedback(item_id: int):
    return {'result': 'result'}
    with open("recordings.json", mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        error = data[str(item_id)]["error"]

    if error == "動作正確":
        prompt = f"""你是一個健身教練，如果有個學員在做硬舉時動作非常正確，你會怎麼鼓勵他？ 請用 markdown 語法回答"""
    else:
        prompt = f"""你是一個健身教練，如果有個學員在做硬舉時發生了錯誤--{error}，請問要怎麼修正他的動作？ 請用 markdown 語法回答"""
    result = api_func.get_openai_response(prompt)
    return {'result': result}


@app.get("/workout_plan/{item_id}")
def read_workout_plan(item_id: int):
    return {'result': 'result'}
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
    return {'result': []}
    with open("recordings.json", mode='r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        error = data[str(item_id)]["error"]

    if error == "動作正確":
        prompt = """你是一個健身教練，如果有個學員在做硬舉時動作非常正確，你會建議他哪些影片？ 請至少推薦 10 個，並將結果輸出成一個 list of dict，dict 的格式為 {'url': '...', 'title': '...'}"""
    else:
        prompt = """你是一個健身教練，如果有個學員在做硬舉時發生了錯誤--%s，你會建議他哪些影片？ 請至少推薦 10 個，並將結果輸出成一個 list of dict，dict 的格式為 {'url': '...', 'title': '...'}""" % error
    videos = api_func.get_openai_response(prompt)
    # 用正則抓中括號內部的資料
    match = re.search(r'\[.*\]', videos, re.DOTALL)
    list_text = match.group(0)
    # 用 ast.literal_eval 將字串轉為真正的 list of dict
    video_list = ast.literal_eval(list_text)

    result = []
    for video in video_list:
        oembed_url = f"https://www.youtube.com/oembed?url={video['url']}&format=json"
        response = requests.get(oembed_url)
        if response.status_code == 200:
            video["video_id"] = video['url'].split("v=")[-1]
            result.append(video)
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

    while True:
        data = await websocket.receive_text()
        result = []
        messages = json.loads(data)

        for message in messages:
            img_data = base64.b64decode(message['image'])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cam_index = message['index']
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
