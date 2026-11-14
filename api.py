"""FastAPI interface for the blog question-answering application."""
import asyncio
import json
import os
import threading
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from weakref import WeakValueDictionary

from anyio import CancelScope
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
    app.state.import_lock = threading.Lock()
    app.state.thread_locks = WeakValueDictionary()
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
    thread_id:str

    @field_validator('question','thread_id')
    @classmethod
    def nonempty(cls, value,info):
        messages = {'question': '请输入你想询问的问题。', 'thread_id': '缺少会话标识。'}
        if not value.strip():
            raise ValueError(messages[info.field_name])
        return value.strip()


@app.post('/api/import')
def import_blog(data: ImportRequest, request: Request):
    with request.app.state.import_lock:
        try:
            return {'count': add_documents_to_qdrant(data.url, request.app.state.resources.db)}
        except Exception as error:
            raise HTTPException(
                status_code=400, detail=safe_error(error, request.app.state.config)
            ) from error

async def event_generator(question: str, request,thread_id:str):
    state = {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "search_query": question,
        "documents": [],
        "relevance": "unknown",
        "rewrite_count": 0,
    }

    config = {"configurable": {"thread_id": thread_id}}

    def sse(payload):
        return f"data:{json.dumps(payload, ensure_ascii=False)}\n\n"

    graph = request.app.state.resources.graph
    chunks = graph.astream(
        state,
        stream_mode=['messages', 'updates', 'tasks'],
        version='v2',
        config=config,
    )
    async with aclosing(chunks):
        async for chunk in chunks:
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
                yield sse({"type": "references", "documents": result["references"]})
    yield "data:[DONE]\n\n"


class ConversationStream(StreamingResponse):
    def __init__(self, data: QuestionRequest, request: Request):
        self.thread_id = data.thread_id
        self.thread_locks = request.app.state.thread_locks
        self.events = event_generator(data.question, request, data.thread_id)
        super().__init__(
            self.events,
            media_type='text/event-stream',
            headers={'X-Accel-Buffering': 'no'},
        )

    async def stream_response(self, send):
        # 同一事件循环内取锁，仍有请求持有或等待时复用同一把锁。
        lock = self.thread_locks.setdefault(self.thread_id, asyncio.Lock())
        async with lock:
            try:
                await super().stream_response(send)
            finally:
                # 客户端断开后也要先关闭生成器，再让下一轮读取 checkpoint。
                with CancelScope(shield=True):
                    await self.events.aclose()


@app.post('/api/ask')
def ask(data: QuestionRequest, request: Request):
    return ConversationStream(data, request)


frontend = BASE_DIR / 'frontend' / 'dist'
if frontend.exists():
    app.mount('/assets', StaticFiles(directory=frontend / 'assets'), name='assets')

    @app.get('/', include_in_schema=False)
    def index():
        return FileResponse(frontend / 'index.html')
