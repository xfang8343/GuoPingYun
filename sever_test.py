import asyncio
import websockets
import json

async def handler(websocket):
    print("客户端已连接")

    await websocket.send(json.dumps({
        "type": "vision-start-test",
        "data": {"test_type": 2}
    }))
    print("已发送启动指令 (文字识别)")

    # 接收客户端发来的消息并打印
    async for msg in websocket:
        data = json.loads(msg)
        msg_type = data.get("type")
        if msg_type == "vision-ocr-recognition":
            print(" 7", json.dumps(data, ensure_ascii=False, indent=2))
        elif msg_type == "heartbeat":
            print("收到心跳")
        else:
            print("其他消息:", msg_type)

async def main():
    async with websockets.serve(handler, "127.0.0.1", 5000):
        print("测试服务器已启动 ws://127.0.0.1:5000")
        await asyncio.Future()  # 永远运行

if __name__ == "__main__":
    asyncio.run(main())