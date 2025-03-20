import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import openai
import os
import tempfile
from main import build_app, run_chatbot
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set OpenAI API Key securely
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize AI App
ai_app = build_app()

class ChatRequest(BaseModel):
    customer_id: str
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        config = {
            "configurable": {
                "thread_id": request.customer_id,
                "checkpoint_ns": "banking_session",
                "checkpoint_id": f"session_{request.customer_id}"
            }
        }
        response = await run_chatbot(ai_app, request.message, request.customer_id, config)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot Error: {str(e)}")

@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(await file.read())
            temp_audio_path = temp_audio.name

        transcript = openai.Audio.transcribe("whisper-1", open(temp_audio_path, "rb"))
        os.remove(temp_audio_path)
        return {"transcription": transcript["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI STT Error: {str(e)}")

@app.post("/tts")
async def text_to_speech(text: str = Form(...)):
    try:
        response = openai.Audio.synthesize("tts-1", text)
        return {"audio_url": response["url"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI TTS Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
