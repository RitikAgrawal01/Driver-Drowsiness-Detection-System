import uvicorn
import sys
import os

# Adds 'model_server' directory to path so main.py can find model_loader.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model_server"))

if __name__ == "__main__":
    uvicorn.run("model_server.main:app", host="0.0.0.0", port=8001, reload=True)