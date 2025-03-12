import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from graph import build_app  # Ensure this is correctly imported
from main import run_chatbot
from tools import is_valid_customer
from fastapi.middleware.cors import CORSMiddleware

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
                "thread_id": request.customer_id,  # ✅ Use customer_id as session ID
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


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)