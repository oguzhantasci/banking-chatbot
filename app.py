import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import asyncio
from graph import build_app
from fastapi.middleware.cors import CORSMiddleware
from main import run_chatbot
from tools import is_valid_customer, transcribe_audio, text_to_speech
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://oguzhantasci.github.io"],  # ✅ Allow GitHub Pages
    allow_credentials=True,
    allow_methods=["*"],  # ✅ Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # ✅ Allow all headers
)

# ✅ Ensure this matches the expected input
class ChatRequest(BaseModel):
    customer_id: str
    message: str

banking_app = build_app()  # ✅ Build the chatbot app at startup


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # ✅ Ensure required keys exist in config
        config = {
            "configurable": {
                "thread_id": request.customer_id,  # ✅ Uses customer_id as session ID
                "checkpoint_ns": "default",  # ✅ Assign a default value
                "checkpoint_id": "default",  # ✅ Assign a default value
            }
        }

        if not is_valid_customer(request.customer_id):
            return {"error": f"⚠️ Geçersiz Müşteri ID: {request.customer_id}. Lütfen kontrol ediniz."}

        response = await run_chatbot(banking_app, request.message, request.customer_id, config)

        if not response:
            raise HTTPException(status_code=400, detail="Yanıt alınamadı. Lütfen tekrar deneyiniz.")

        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hata oluştu: {str(e)}")

@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    """Speech-to-Text (STT) API: Converts speech to text using OpenAI Whisper."""
    try:
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as buffer:
            buffer.write(await file.read())

        text = transcribe_audio(file_location)

        # Remove the temporary audio file
        os.remove(file_location)

        return {"transcription": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT Hata: {str(e)}")

@app.post("/tts")
async def text_to_speech_api(text: str = Form(...)):
    """Text-to-Speech (TTS) API: Converts text to speech using OpenAI API."""
    try:
        output_audio = "output_speech.wav"
        text_to_speech(text, output_audio)

        return {"audio_file": output_audio}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS Hata: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
