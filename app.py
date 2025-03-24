import uvicorn
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from main import run_chatbot, build_app
from tools import transcribe_audio, generate_speech_base64, is_valid_customer
import base64
import tempfile

app = FastAPI()
chatbot_app = build_app()

# CORS ayarları (frontend erişimi için gerekli)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Gerekirse sadece frontend domain ile sınırla
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chatbot_endpoint(request: Request):
    data = await request.json()
    query = data.get("query", "")
    customer_id = data.get("customer_id", "")

    if not is_valid_customer(customer_id):
        return JSONResponse(content={"response": "❌ Geçersiz müşteri ID."}, status_code=400)

    config = {
        "configurable": {
            "thread_id": customer_id,
            "checkpoint_ns": "banking_session",
            "checkpoint_id": f"webchat_{customer_id}"
        }
    }

    # AI cevabı
    response = await run_chatbot(chatbot_app, query, customer_id, config)

    # TTS ile base64 ses
    audio_base64 = await generate_speech_base64(response)

    return JSONResponse(content={
        "response": response,
        "audio": audio_base64
    })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    customer_id = websocket.query_params.get("customer_id", "anon")

    config = {
        "configurable": {
            "thread_id": customer_id,
            "checkpoint_ns": "banking_session",
            "checkpoint_id": f"voicebot_{customer_id}"
        }
    }

    try:
        audio_data = await websocket.receive_bytes()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_data)
            audio_file_path = temp_audio.name

        # STT: Türkçe ses → metin
        query = await transcribe_audio(audio_file_path)

        if not query or query.startswith("⚠️"):
            await websocket.send_json({"text": "⚠️ Ses çözümlenemedi."})
            return

        # AI yanıtı al
        response = await run_chatbot(chatbot_app, query, customer_id, config)

        # TTS: metin → ses (base64)
        audio_base64 = await generate_speech_base64(response)

        await websocket.send_json({
            "query": query,  # ✅ Bu satırı ekle
            "text": response
        })
        await websocket.send_bytes(base64.b64decode(audio_base64))

    except Exception as e:
        print(f"❌ WebSocket Error: {e}")
        try:
            await websocket.send_json({"text": f"⚠️ Hata: {e}"})
        except:
            pass
    finally:
        await websocket.close()

@app.get("/")
async def root():
    return {"message": "AI Banking Assistant is running 🎧💬"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
