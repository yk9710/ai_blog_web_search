# AI 博客问答

React + Ant Design X 前端，FastAPI 后端。项目沿用千问嵌入、Qdrant 和 DeepSeek 的博客检索问答流程，支持可选的 Tavily 网络搜索。

## 启动

1. 准备 Python 3.10+、Node.js 20.19+ 或 22.12+，以及运行中的 Qdrant 服务。
2. 安装后端依赖并配置环境变量：

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env
   # 编辑 .env，填写可用的模型 ID、地址和 API 密钥
   uvicorn api:app --reload
   ```

3. 在另一个终端启动前端：

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

打开 Vite 输出的地址（默认 http://localhost:5173）。后端启动时读取 `.env`，初始化并共享模型和数据库资源，前端无需处理连接。前端开发服务器会将 `/api` 转发到 `http://127.0.0.1:8000`。FastAPI 的接口文档位于 http://127.0.0.1:8000/docs。

在对话框中单独发送一个完整的 http 或 https 文章链接即可导入；发送其他文字即可提问。首次使用或更换嵌入模型后，请重新导入博客；不同模型的旧知识库会保留。导入仅支持公开可访问的 HTML 或纯文本页面。问题可以使用博客检索，配置 `TAVILY_API_KEY` 后也可使用联网搜索。

构建后也可以由 FastAPI 提供页面：在 `frontend` 执行 `npm run build`，然后启动 `uvicorn api:app`，访问 http://127.0.0.1:8000。

模型和数据库资源在每个后端进程启动时初始化一次，并在进程关闭时释放数据库连接。若初始化失败，服务不会启动，请检查 `.env` 和外部服务；修改配置后需重启后端。多进程部署时，每个进程各自初始化一份资源。本项目面向本地使用，尚未提供身份认证，不应直接暴露到公网。

## 测试

```bash
pip install pytest
python -m pytest -q
cd frontend && npm run build
```
