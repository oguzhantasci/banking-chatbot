from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from main import run_chatbot  # Import your chatbot logic

app = FastAPI()

class ChatRequest(BaseModel):
    customer_id: str
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = await run_chatbot(request.customer_id, request.message)
        return {"response": response}  # ✅ Always return JSON
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # ✅ Handle errors properly

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)