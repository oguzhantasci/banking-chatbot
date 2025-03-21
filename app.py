import uvicorn
from pydantic import BaseModel
import openai
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from graph import build_app
from tools import transcribe_audio, synthesize_text
from fastapi import FastAPI, WebSocket, Request, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
import uuid
import json
from main import run_chatbot, build_app
from tools import transcribe_audio, text_to_speech

# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS (gerekirse localhost ve frontend için açılır)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

config, chatbot_app = build_app()

@app.post("/chat")
async def chatbot_endpoint(request: Request):
    data = await request.json()
    query = data.get("query", "")
    customer_id = data.get("customer_id", "")

    if not query or not customer_id:
        return JSONResponse(content={"error": "Eksik parametre"}, status_code=400)

    response = await run_chatbot(chatbot_app, query, customer_id, config)
    output_audio_path = f"static/response_audio.wav"
    text_to_speech(response, output_audio_path)

    return {
        "response": response,
        "audio_url": "/static/response_audio.wav"
    }

@app.websocket("/ws")
async def websocket_voice_endpoint(websocket: WebSocket):
    await websocket.accept()
    customer_id = websocket.query_params.get("customer_id")

    if not customer_id:
        await websocket.send_text(json.dumps({"error": "Müşteri ID alınamadı"}))
        await websocket.close()
        return

    audio_filename = f"audio_{uuid.uuid4().hex}.wav"
    with open(audio_filename, "wb") as audio_file:
        try:
            audio_data = await websocket.receive_bytes()
            audio_file.write(audio_data)
        except Exception as e:
            await websocket.send_text(json.dumps({"error": str(e)}))
            await websocket.close()
            return

    try:
        query = transcribe_audio(audio_filename)
        await websocket.send_text(json.dumps({"text": query}))

        config, chatbot_app = build_app()
        response = await run_chatbot(chatbot_app, query, customer_id, config)

        audio_path = f"static/response_audio.wav"
        text_to_speech(response, audio_path)

        with open(audio_path, "rb") as audio_file:
            await websocket.send_bytes(audio_file.read())

    except Exception as e:
        await websocket.send_text(json.dumps({"error": f"İşlem hatası: {str(e)}"}))

    await websocket.close()
    os.remove(audio_filename)

@app.get("/")
async def root():
    return {"message": "AI Banking Assistant is running 🎧💬"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)