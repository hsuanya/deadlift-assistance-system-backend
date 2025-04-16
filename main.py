from datetime import datetime
from typing import Union
from fastapi import FastAPI, Request, WebSocket
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

import api_func

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 或指定 frontend 網域，如 "http://localhost:3000"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.get("/detection_result/{item_id}")
def read_detection_result(item_id: int):
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
        with open("feedback.json", mode='r', encoding='utf-8') as json_file:
            data = json.load(json_file)
    except:
        data = {}
    data[item_id] = result
    with open("feedback.json", mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)

    return result


@app.get("/feedback/{item_id}")
def read_feedback(item_id: int):
    # return {'result': 'result'}
    with open("feedback.json", mode='r', encoding='utf-8') as json_file:
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
    # return {'result': 'result'}
    with open("feedback.json", mode='r', encoding='utf-8') as json_file:
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
    with open("feedback.json", mode='r', encoding='utf-8') as json_file:
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


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # while True:
    #     try:
    #         data = await websocket.receive_bytes()
    #         # 你可以用 OpenCV 處理這個 frame，例如：
    #         # image_np = np.frombuffer(data, np.uint8)
    #         # frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    #         print(f"Received frame size: {len(data)}")

    #         # 處理完後把回傳資料送回（這邊只是 echo）
    #         await websocket.send_text("Frame received")
    #     except Exception as e:
    #         print("WebSocket error:", e)
    #         break
    while True:
        data = await websocket.receive_text()
        message = json.loads(data)
        img_data = base64.b64decode(message['image'])
        cam_index = message['index']

        # 解碼成影像
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 儲存影像（可根據時間與 camera index 命名）
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # filename = f"tmp/camera_{timestamp}.jpg"
        # cv2.imwrite(filename, frame)

        # 做一些處理（範例：轉灰階）
        processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        # 回傳處理後的影像
        _, buffer = cv2.imencode('.jpg', processed)
        processed_b64 = base64.b64encode(buffer).decode('utf-8')

        await websocket.send_text(
            json.dumps({
                "index": cam_index,
                "processed_image": processed_b64,
            }))
