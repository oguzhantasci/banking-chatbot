from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from main import run_chatbot  # ✅ Ensure this is correctly imported

app = FastAPI()

class ChatRequest(BaseModel):
    customer_id: str
    message: str

@app.get("/")
def home():
    return {"message": "FastAPI is running!"}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        config = {}  # Ensure config is not None
        response = await run_chatbot(app, request.message, request.customer_id, config)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
