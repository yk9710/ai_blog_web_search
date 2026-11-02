/**
 * 流式请求封装（SSE over fetch）
 * @param {string} url
 * @param {object} options
 * @param {object} options.body       请求体，会 JSON.stringify
 * @param {object} options.headers    额外请求头
 * @param {AbortSignal} options.signal 用于中断
 * @param {(content: string) => void} options.onMessage  每收到一段文本回调
 * @param {(err: Error) => void} options.onError
 * @param {() => void} options.onDone
 */
export async function fetchStream(url, {
    body,
    headers = {},
    signal,
    onMessage,
    onError,
    onDone,
} = {}) {
    let reader = null;
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                Accept: 'text/event-stream',
                ...headers,
            },
            body: JSON.stringify(body),
            signal,
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        if (!response.body) {
            throw new Error('当前环境不支持 ReadableStream');
        }

        reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
            const {done, value} = await reader.read();
            if (done) break;

            // stream: true 保证多字节字符（中文）不被截断
            buffer += decoder.decode(value, {stream: true});

            // 按空行切分 SSE 帧，最后一段可能不完整，留在 buffer
            const parts = buffer.split('\n\n');
            buffer = parts.pop();

            for (const part of parts) {
                const line = part.trim();
                if (!line) continue;

                // 一帧可能有多行 data:，按规范拼接
                const dataLines = line
                    .split('\n')
                    .filter((l) => l.startsWith('data:'))
                    .map((l) => l.slice(5).trimStart());

                if (dataLines.length === 0) continue; // 忽略 event/id/retry 等非 data 行
                const dataStr = dataLines.join('\n');

                // 结束标识
                if (dataStr === '[DONE]') {
                    onDone?.();
                    return;
                }

                try {
                    const json = JSON.parse(dataStr);
                    // 后端约定字段：content；若还有 done 标识也可在此判断
                    if (json.done) {
                        onDone?.();
                        return;
                    }
                    if (json.content) onMessage?.(json.content);
                } catch (e) {
                    // 非 JSON 帧：可选择当作纯文本透传，或忽略
                    onMessage?.(dataStr);
                }
            }
        }

        // 流自然结束但没收到 [DONE]
        onDone?.();
    } catch (err) {
        if (err.name === 'AbortError') {
            // 主动中断，不算错误
            onDone?.();
            return;
        }
        onError?.(err);
    } finally {
        try {
            reader?.releaseLock();
        } catch {
        }
    }
}