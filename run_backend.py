import uvicorn
import sys
import os
from dotenv import load_dotenv

# 1. Force load the .env file from the project root and override any stale cache
root_dir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(root_dir, ".env"), override=True)

# 2. Add backend to python path
sys.path.insert(0, os.path.join(root_dir, "backend"))

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)