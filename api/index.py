import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fetch the API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

class ChatRequest(BaseModel):
    message: str

class TopicRequest(BaseModel):
    topic: str

@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    try:
        # Upgraded to the new Gemini 3.5 Flash model
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=req.message,
            config=types.GenerateContentConfig(
                # Gemini 3 feature: Sets the depth of internal reasoning
                thinking_level="low", 
                system_instruction="You are an expert personal academic tutor specializing in high school curriculums."
            )
        )
        return {"reply": response.text}
    except Exception as e:
        print(f"Chat Error: {str(e)}") 
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/research")
def research_endpoint(req: ChatRequest):
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=f"Find helpful study notes, textbook resources, and direct download PDF links for: {req.message}",
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}], 
                thinking_level="low",
                system_instruction="You are an educational research assistant. Scan live search indexes to discover academic notes, guides, and direct PDF links. Present sources explicitly with titles and clickable text URLs."
            )
        )
        return {"reply": response.text}
    except Exception as e:
        print(f"Research Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-cards")
def generate_cards_endpoint(req: TopicRequest):
    try:
        prompt = f"Generate exactly 4 distinct high-yield study flashcards for the topic: {req.topic}. You must output valid raw JSON matching this structure: [{{\"f\": \"Question or core term\", \"b\": \"Clear, concise explanation or definition\"}}, ...]"
        
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3
            )
        )
        return {"flashcards": json.loads(response.text)}
    except Exception as e:
        print(f"Flashcard Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
