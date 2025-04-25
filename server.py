from fastapi import FastAPI, Response, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import uvicorn
import asyncio
import pyaudio
import wave
import io
import time
import threading
from typing import List, Dict

from fastapi.staticfiles import StaticFiles

app = FastAPI()

# ルーティングの前に追加
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORSミドルウェアを追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境ではより制限的に設定する
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# カメラ設定
camera = cv2.VideoCapture(0)  # 0はデフォルトのカメラ
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)

# 音声設定
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
audio = pyaudio.PyAudio()
audio_stream = None
audio_frames = []
is_recording = False
recording_lock = threading.Lock()

def record_audio():
    global audio_frames, is_recording
    
    audio_stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    
    while is_recording:
        data = audio_stream.read(CHUNK)
        with recording_lock:
            audio_frames.append(data)
    
    audio_stream.stop_stream()
    audio_stream.close()

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/")
async def index():
    return {"message": "カメラとマイクAPIサーバー"}

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), 
                            media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/start_recording")
async def start_recording():
    global is_recording, audio_frames
    
    if not is_recording:
        audio_frames = []
        is_recording = True
        threading.Thread(target=record_audio).start()
        return {"status": "recording_started"}
    return {"status": "already_recording"}

@app.post("/stop_recording")
async def stop_recording():
    global is_recording, audio_frames
    
    if is_recording:
        is_recording = False
        time.sleep(0.5)  # 録音スレッドが終了するのを待つ
        
        # WAVファイルを作成
        with recording_lock:
            if audio_frames:
                output = io.BytesIO()
                wf = wave.open(output, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(audio_frames))
                wf.close()
                
                # ファイルポインタを先頭に戻す
                output.seek(0)
                return Response(content=output.getvalue(), media_type="audio/wav")
    
    return {"status": "not_recording"}

@app.get("/audio_status")
async def audio_status():
    return {"is_recording": is_recording, "frames_count": len(audio_frames)}

# WebSocketを使用したオーディオストリーミング
@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    
    # WebSocketを通じてオーディオをストリーミング
    audio_ws_stream = audio.open(format=FORMAT,
                                 channels=CHANNELS,
                                 rate=RATE, input=True,
                                 frames_per_buffer=CHUNK)
    
    try:
        while True:
            data = audio_ws_stream.read(CHUNK)
            await websocket.send_bytes(data)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        audio_ws_stream.stop_stream()
        audio_ws_stream.close()

@app.on_event("shutdown")
def shutdown_event():
    camera.release()
    audio.terminate()

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
