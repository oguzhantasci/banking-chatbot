import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket
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
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("❌ ERROR: OpenAI API Key is missing! Set it in Render Environment Variables.")

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Set in env
openai.api_key = os.getenv("OPENAI_API_KEY")  # Set for OpenAI API

# Initialize AI App
ai_app = build_app()

class ChatRequest(BaseModel):
    customer_id: str
    message: str

# Store conversation history
conversation_history = {}

@app.post("/chat")
async def chatbot_endpoint(customer_id: str = Form(...), message: str = Form(...)):
    """Handles text-based chatbot interactions."""
    session_id = customer_id

    # Initialize conversation history
    if session_id not in conversation_history:
        conversation_history[session_id] = []

    # Add user message to chat history
    conversation_history[session_id].append({"role": "user", "content": message})

    # Generate AI response using GPT-4o
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=conversation_history[session_id]
    )["choices"][0]["message"]["content"]

    # Add AI response to chat history
    conversation_history[session_id].append({"role": "assistant", "content": response})

    return {"response": response}

@app.websocket("/ws")
async def websocket_voice_endpoint(websocket: WebSocket):
    """Handles real-time voice conversation over WebSocket."""
    await websocket.accept()
    session_id = websocket.client.host

    # Initialize conversation history
    if session_id not in conversation_history:
        conversation_history[session_id] = []

    try:
        while True:
            # Receive voice data
            audio_data = await websocket.receive_bytes()

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name

            # Convert speech to text
            transcript = openai.Audio.transcribe(
                model="whisper-1",
                file=open(temp_audio_path, "rb")
            )["text"]

            os.remove(temp_audio_path)

            # Add user message to chat history
            conversation_history[session_id].append({"role": "user", "content": transcript})

            # Generate AI response using GPT-4o
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=conversation_history[session_id]
            )["choices"][0]["message"]["content"]

            # Add AI response to chat history
            conversation_history[session_id].append({"role": "assistant", "content": response})

            # Convert AI text response to speech
            tts_response = openai.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=response
            )

            # Send voice response back to the user
            await websocket.send_bytes(tts_response.content)

    except Exception as e:
        print(f"Error: {str(e)}")
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)