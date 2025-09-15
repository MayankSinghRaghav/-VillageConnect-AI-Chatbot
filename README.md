# VillageConnect — AI Chatbot (Final project)

## Overview
VillageConnect is a conversational AI that answers village-level service queries.

## Setup
```
pip install -r requirements.txt
```

## Training
```
cd src
python train.py
```

## Chat CLI
```
python src/inference.py
```

## Run API
```
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

## Docker
```
docker build -t villageconnect .
docker run -p 8000:8000 -v $(pwd)/models:/app/models villageconnect
```
