import os
from dotenv import load_dotenv

# 1. 优先加载环境变量 (必须在导入 homework 之前，否则 ai_pipeline 获取不到 key)
load_dotenv()

# 2. 确保不会被本地代理干扰 API 调用
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['ALL_PROXY'] = ''

from fastapi import FastAPI
from backend.api import homework
import uvicorn

app = FastAPI(title="智能作业批改系统 MVP 版", version="1.0.0")

app.include_router(homework.router, prefix="/api", tags=["homework"])

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Autocontrol AI Assistant Backend Running"}

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
