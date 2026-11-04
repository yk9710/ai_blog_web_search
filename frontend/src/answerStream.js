import {XStream} from '@ant-design/x-sdk'

export async function readAnswerStream(readableStream, onEvent) {
    for await (const chunk of XStream({readableStream})) {
        if (chunk.data === '[DONE]') return
        onEvent(JSON.parse(chunk.data))
    }
    throw new Error('流未正常结束')
}
