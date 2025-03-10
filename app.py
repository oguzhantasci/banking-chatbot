import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from graph import build_app  # Ensure this is correctly imported
from main import run_chatbot

app = FastAPI()

# ✅ Ensure this matches the expected input
class ChatRequest(BaseModel):
    customer_id: str
    message: str

banking_app = build_app()  # ✅ Build the chatbot app at startup

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        config = {}
        # ✅ Pass the banking_app to run_chatbot()
        response = await run_chatbot(banking_app, request.message, request.customer_id, config)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
