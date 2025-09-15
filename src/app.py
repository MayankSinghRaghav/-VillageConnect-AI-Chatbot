from fastapi import FastAPI
from pydantic import BaseModel
import os
from inference import VillageBot

app = FastAPI(title="VillageConnect Chatbot API")
MODEL_DIR = os.environ.get("MODEL_DIR", "./models/villageconnect-dialo")
bot = VillageBot(MODEL_DIR)

class ChatRequest(BaseModel):
    user_text: str

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    reply = bot.chat(req.user_text)
    return ChatResponse(reply=reply)

@app.get("/")
async def root():
    return {"message": "VillageConnect Chatbot API. POST /chat with {user_text}."}
