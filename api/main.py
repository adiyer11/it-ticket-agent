"""
api/main.py
REST API endpoint for the IT ticket automation agent.
"""

import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from agent.router import route

app = FastAPI(title="IT Ticket Agent", version="1.0")

# Load classifier pipeline at startup
pipeline = joblib.load("classifier/model.pkl")


class TicketRequest(BaseModel):
    text: str


class TicketResponse(BaseModel):
    category: str
    confidence: float
    resolved: bool
    escalated: bool
    resolution: str


@app.post("/ticket", response_model=TicketResponse)
def submit_ticket(req: TicketRequest):
    # Classify
    category = pipeline.predict([req.text])[0]
    proba = pipeline.predict_proba([req.text])[0]
    confidence = float(max(proba))

    # Route + resolve
    result = route(req.text, category, confidence)

    return TicketResponse(
        category=category,
        confidence=round(confidence, 3),
        resolved=result["resolved"],
        escalated=result["escalated"],
        resolution=result["resolution"],
    )


@app.get("/health")
def health():
    return {"status": "ok"}
