from dotenv import load_dotenv
from pathlib import Path

# 1. 优先加载环境变量 (必须在导入 homework 之前，否则 ai_pipeline 获取不到 key)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from fastapi import FastAPI
from backend.api import homework
import uvicorn

app = FastAPI(title="自动控制原理作业辅助批改", version="2.0.0")

app.include_router(homework.router, prefix="/api", tags=["homework"])

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Autocontrol AI Assistant Backend Running"}

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
