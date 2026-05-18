# IT Ticket Automation Agent

An agentic system that automatically classifies, routes, and resolves internal IT support tickets using LLM orchestration and ML classification — built to eliminate manual triage for IT teams.

Inspired by production work at Cisco, where this architecture achieved **80% ticket automation accuracy** across a live service-request workflow.

---

## What it does

1. **Classifies** incoming tickets into categories (password reset, access request, hardware issue, software install, onboarding, etc.) using a trained ML classifier
2. **Routes** each ticket to the appropriate resolution handler via an LLM-orchestrated agent loop
3. **Resolves** high-confidence tickets automatically; flags low-confidence ones for human review via a validation layer
4. **Learns** from unresolved tickets to suggest new automation rules over time

---

## Architecture

```
Incoming Ticket (Slack / API)
        │
        ▼
┌─────────────────────┐
│   Classifier Agent  │  scikit-learn model (TF-IDF + LogisticRegression)
│   + LLM fallback    │  GPT-4o for ambiguous / novel categories
└────────┬────────────┘
         │  category + confidence score
         ▼
┌─────────────────────┐
│   Router Agent      │  LLM-orchestrated dispatch (LangChain)
│                     │  routes to resolution handler based on category
└────────┬────────────┘
         │
    ┌────┴─────────────────────────┐
    │                              │
    ▼                              ▼
┌──────────────┐          ┌──────────────────┐
│ Auto-Resolve │          │  Human Escalation │
│ Handler      │          │  Queue (low conf) │
└──────────────┘          └──────────────────┘
         │
         ▼
  Resolution logged + feedback loop for retraining
```

---

## Stack

| Layer | Tool |
|---|---|
| Classification | scikit-learn (TF-IDF + Logistic Regression) |
| LLM orchestration | LangChain + OpenAI GPT-4o |
| Synthetic training data | LLM-generated via prompt engineering |
| API layer | FastAPI (REST) |
| Deployment | Docker + GitHub Actions CI/CD |
| Validation | Confidence threshold layer + human escalation queue |

---

## Project structure

```
it-ticket-agent/
├── classifier/
│   ├── train.py          # Train and evaluate the ML classifier
│   ├── predict.py        # Run inference on new tickets
│   └── generate_data.py  # LLM-powered synthetic training data generation
├── agent/
│   ├── router.py         # LangChain agent loop for ticket routing
│   ├── handlers.py       # Resolution handlers per ticket category
│   └── validator.py      # Confidence threshold + escalation logic
├── api/
│   └── main.py           # FastAPI REST endpoint
├── data/
│   └── sample_tickets.csv
├── tests/
│   └── test_classifier.py
├── Dockerfile
├── .github/workflows/ci.yml
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/adiyer11/it-ticket-agent
cd it-ticket-agent
pip install -r requirements.txt

# Generate synthetic training data via LLM
python classifier/generate_data.py --categories "password_reset,access_request,hardware,software,onboarding" --n 200

# Train the classifier
python classifier/train.py --data data/tickets.csv

# Start the API
uvicorn api.main:app --reload
```

```bash
# Submit a ticket
curl -X POST http://localhost:8000/ticket \
  -H "Content-Type: application/json" \
  -d '{"text": "I cannot log into my Okta account, getting MFA errors"}'

# Response
{
  "category": "access_request",
  "confidence": 0.94,
  "resolved": true,
  "resolution": "Okta MFA reset link sent to user email.",
  "escalated": false
}
```

---

## Key design decisions

**Why a hybrid classifier + LLM approach?**
The ML classifier is fast and cheap for common ticket types (80%+ of volume). The LLM handles novel or ambiguous tickets where the classifier confidence falls below threshold — giving the best of both worlds on cost and accuracy.

**Synthetic training data via prompt engineering**
Real IT ticket data is sensitive and hard to obtain. Instead, GPT-4o generates realistic synthetic tickets per category with controlled variation in phrasing, urgency, and technical detail — producing a robust training set without privacy concerns.

**Confidence-gated escalation**
Every prediction includes a confidence score. Tickets below 0.75 confidence are routed to a human review queue rather than auto-resolved. This keeps automation accuracy high while ensuring edge cases get human attention.

---

## Results (Cisco production deployment)

- **80% of tickets** resolved without human intervention
- Average resolution time reduced from ~4 hours to ~3 minutes for automated categories
- Classifier retrained weekly on newly resolved tickets via CI/CD pipeline

---

## Extending to new ticket categories

The agent is designed to be category-agnostic. To add a new ticket type:

1. Add the category name to `generate_data.py` and regenerate training data
2. Add a resolution handler in `handlers.py`
3. Retrain the classifier — no pipeline changes needed

This mirrors the generalized agentic framework built at Cisco to support future use cases without re-engineering.
