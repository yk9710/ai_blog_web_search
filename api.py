"""FastAPI interface for the blog question-answering application."""
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, AIMessageChunk
from pydantic import BaseModel, field_validator
from starlette.responses import StreamingResponse

from blog_rag import AppConfig, add_documents_to_qdrant, ask_question, initialize_components

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')


def safe_error(error, config):
    detail = str(error)
    for key in (
        config.qwen_api_key, config.deepseek_api_key, config.qdrant_api_key,
        os.getenv('TAVILY_API_KEY', ''),
    ):
        if key:
            detail = detail.replace(key, '<密钥已隐藏>')
    return detail


@asynccontextmanager
async def lifespan(app: FastAPI):
    names = ('qwen_model', 'qwen_api_key', 'qwen_base_url', 'deepseek_model',
             'deepseek_api_key', 'deepseek_base_url', 'qdrant_host', 'qdrant_api_key')
    config = AppConfig(**{name: os.getenv(name.upper(), '').strip() for name in names})
    try:
        resources = await run_in_threadpool(initialize_components, config)
    except Exception as error:
        raise RuntimeError(f'服务初始化失败：{safe_error(error, config)}') from None
    app.state.config = config
    app.state.resources = resources
    app.state.resource_lock = threading.RLock()
    try:
        yield
    finally:
        await run_in_threadpool(resources.client.close)


app = FastAPI(title='AI 博客问答 API', lifespan=lifespan)


class ImportRequest(BaseModel):
    url: str

    @field_validator('url')
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError('请输入博客文章链接 。')
        return value.strip()


class QuestionRequest(BaseModel):
    question: str

    @field_validator('question')
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError('请输入你想询问的问题。')
        return value.strip()


@app.post('/api/import')
def import_blog(data: ImportRequest, request: Request):
    with request.app.state.resource_lock:
        try:
            return {'count': add_documents_to_qdrant(data.url, request.app.state.resources.db)}
        except Exception as error:
            raise HTTPException(
                status_code=400, detail=safe_error(error, request.app.state.config)
            ) from error

async def event_generator(question:str,request):
    async for chunk in request.app.state.resources.graph.astream({
        "messages": [HumanMessage(content=question)],
        "question": question,
        "search_query": question,
        "search_type": "unknown",
        "documents": [],
        "relevance": "unknown",
        "reason": "",
        "rewrite_count": 0,
        "answer": "",
        "sources": [],
    },stream_mode="messages",version="v2"):
        if chunk["type"]=="messages":
            token,metadata=chunk["data"]
            if metadata.get("langgraph_node") != "generate" or not isinstance(token.content, str):
                continue
            # messages 模式会重放节点返回的完整 AIMessage，只转发真正的 token 增量
            if not isinstance(token, AIMessageChunk):
                continue
            if token.content:
                yield f"data:{json.dumps(
                    {'content': token.content, 'role': 'assistant'}, ensure_ascii=False
                )}\n\n"
    yield "data:[DONE]\n\n"

@app.post('/api/ask')
def ask(data: QuestionRequest, request: Request):
    with request.app.state.resource_lock:
        try:
            return StreamingResponse(
                event_generator(data.question, request),
                media_type='text/event-stream',
                headers={"X-Accel-Buffering": "no"},
            )
        except Exception as error:
            raise HTTPException(
                status_code=400, detail=safe_error(error, request.app.state.config)
            ) from error


frontend = BASE_DIR / 'frontend' / 'dist'
if frontend.exists():
    app.mount('/assets', StaticFiles(directory=frontend / 'assets'), name='assets')

    @app.get('/', include_in_schema=False)
    def index():
        return FileResponse(frontend / 'index.html')
