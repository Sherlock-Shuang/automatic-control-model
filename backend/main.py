from fastapi import FastAPI
from backend.api import homework
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="智能作业批改系统 MVP 版", version="1.0.0")

app.include_router(homework.router, prefix="/api", tags=["homework"])

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Autocontrol AI Assistant Backend Running"}

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
