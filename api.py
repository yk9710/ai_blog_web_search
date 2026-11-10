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

from blog_rag import (
    AppConfig,
    add_documents_to_qdrant,
    build_references,
    initialize_components,
)

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

STEP_TITLES = {
    'classify_intent': '理解问题',
    'retrieve': '检索博客',
    'grade': '评估资料',
    'rewrite': '优化检索问题',
    'search': '联网搜索',
    'generate': '生成回答',
}


def task_step(task):
    node = task['name']
    if node not in STEP_TITLES:
        return None
    step = {
        'type': 'step',
        'id': task['id'],
        'node': node,
        'title': STEP_TITLES[node],
        'status': 'loading',
    }
    if 'error' not in task:
        return step
    if task['error'] is not None:
        step['status'] = 'error'
        return step
    step['status'] = 'success'
    result = task['result']
    if node == 'classify_intent':
        step['description'] = '直接回答' if result['intent'] == 'direct' else '需要检索资料'
    elif node in ('retrieve', 'search'):
        unit = '个片段' if node == 'retrieve' else '条结果'
        step['description'] = f"找到 {len(result['documents'])} {unit}"
    elif node == 'grade':
        label = '资料充分' if result['relevance'] == 'yes' else '资料不足'
        step['description'] = f"{label}：{result['reason']}"
    elif node == 'rewrite':
        step['description'] = result['search_query']
    return step


class ImportRequest(BaseModel):
    url: str

    @field_validator('url')
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError('请输入博客文章链接。')
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

async def event_generator(question: str, request):
    state = {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "search_query": question,
        "documents": [],
        "relevance": "unknown",
        "rewrite_count": 0,
    }

    def sse(payload):
        return f"data:{json.dumps(payload, ensure_ascii=False)}\n\n"

    graph = request.app.state.resources.graph
    async for chunk in graph.astream(
        state, stream_mode=['messages', 'updates', 'tasks'], version='v2'
    ):
        if chunk["type"] == "messages":
            token, metadata = chunk["data"]
            if metadata.get("langgraph_node") != "generate":
                continue
            if not isinstance(token, AIMessageChunk) or not isinstance(token.content, str):
                continue
            reasoning = token.additional_kwargs.get("reasoning_content") or ""
            if reasoning:
                yield sse({"type": "reasoning", "reasoning": reasoning})
            if token.content:
                yield sse({"type": "content", "content": token.content})
        elif chunk['type'] == 'tasks':
            step = task_step(chunk['data'])
            if step:
                yield sse(step)
        elif chunk["type"] == "updates":
            result = chunk["data"].get("generate")
            if result is None:
                continue
            references = build_references(result.get("documents", []))
            yield sse({"type": "references", "documents": references})
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
