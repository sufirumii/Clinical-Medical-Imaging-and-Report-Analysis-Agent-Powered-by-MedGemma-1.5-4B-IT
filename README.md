# MedGemma AI Agent

## Structure
```
AI Agent/
├── api.py          ← FastAPI backend
├── model.py        ← MedGemma loader + inference
├── static/
│   └── index.html  ← Frontend UI
├── uploads/        ← temp image storage
└── requirements.txt
```

## Run
```bash
cd "/home/AI Agent"
python api.py
```
Then open http://localhost:7860
