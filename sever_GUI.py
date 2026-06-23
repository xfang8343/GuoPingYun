import sys
import json
import asyncio
import websockets
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTextEdit, QVBoxLayout, 
                             QWidget, QPushButton, QHBoxLayout, QLabel)
from PyQt6.QtCore import QObject, pyqtSignal, QThread

class WebSocketServer(QThread):
    message_received = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.clients = set()
        self.server = None
        self.loop = asyncio.new_event_loop()

    def run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def start_server(self):
        """主线程调用，向子线程提交启动任务"""
        asyncio.run_coroutine_threadsafe(self._start_server_task(), self.loop)

    async def _start_server_task(self):
        if self.server is None:
            try:
                self.server = await websockets.serve(self.handler, "0.0.0.0", 5000)
                self.message_received.emit("【系统】WebSocket 服务已启动 (0.0.0.0:5000)")
            except Exception as e:
                self.message_received.emit(f"【错误】无法启动服务: {e}")
        else:
            self.message_received.emit("【系统】服务已经在运行中")

    def stop_server(self):
        """主线程调用，向子线程提交关闭任务"""
        asyncio.run_coroutine_threadsafe(self._stop_server_task(), self.loop)

    async def _stop_server_task(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
            self.message_received.emit("【系统】WebSocket 服务已关闭")

    async def handler(self, websocket):
        self.clients.add(websocket)
        self.message_received.emit("【系统】客户端连接成功")
        try:
            async for message in websocket:
                self.message_received.emit(f"【接收】{message}")
                data = json.loads(message)
                if data.get("type") == "heartbeat":
                    await websocket.send(json.dumps({"type": "heartbeat"}))
        finally:
            self.clients.remove(websocket)
            self.message_received.emit("【系统】客户端已断开")

    def send_msg(self, msg_dict):
        if self.clients:
            asyncio.run_coroutine_threadsafe(self._broadcast(msg_dict), self.loop)

    async def _broadcast(self, msg_dict):
        msg = json.dumps(msg_dict)
        for client in self.clients:
            await client.send(msg)

class ServerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Debug Server - 已添加文字识别")
        self.resize(600, 650) # 适当调大高度以容纳新按钮

        # 布局初始化
        layout = QVBoxLayout()
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        
        btn_layout = QHBoxLayout()
        self.btn_serve_on = QPushButton("开启服务")
        self.btn_serve_off = QPushButton("关闭服务")
        btn_layout.addWidget(self.btn_serve_on)
        btn_layout.addWidget(self.btn_serve_off)

        task_layout = QVBoxLayout()
        
        # 功能按钮
        self.btn_start_face = QPushButton("启动人脸识别")
        self.btn_stop_face = QPushButton("停止人脸识别")
        self.btn_start_obj = QPushButton("启动物体识别")
        self.btn_stop_obj = QPushButton("停止物体识别")
        self.btn_start_ocr = QPushButton("启动文字识别")
        self.btn_stop_ocr = QPushButton("停止文字识别")
        
        task_layout.addWidget(self.btn_start_face)
        task_layout.addWidget(self.btn_stop_face)
        task_layout.addWidget(self.btn_start_obj)
        task_layout.addWidget(self.btn_stop_obj)
        task_layout.addWidget(self.btn_start_ocr)
        task_layout.addWidget(self.btn_stop_ocr)
        
        layout.addWidget(QLabel("实时消息流:"))
        layout.addWidget(self.log_box)
        layout.addLayout(btn_layout)
        layout.addLayout(task_layout)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # 服务线程启动
        self.server_thread = WebSocketServer()
        self.server_thread.message_received.connect(self.log_box.append)
        self.server_thread.start()

        # 事件绑定
        self.btn_serve_on.clicked.connect(self.server_thread.start_server)
        self.btn_serve_off.clicked.connect(self.server_thread.stop_server)
        
        # 业务指令 (0:人脸, 1:物体, 2:文字)
        self.btn_start_face.clicked.connect(lambda: self.server_thread.send_msg({"type": "vision-start-test", "data": {"test_type": 0}}))
        self.btn_stop_face.clicked.connect(lambda: self.server_thread.send_msg({"type": "vision-end-test", "data": {"test_type": 0}}))
        self.btn_start_obj.clicked.connect(lambda: self.server_thread.send_msg({"type": "vision-start-test", "data": {"test_type": 1}}))
        self.btn_stop_obj.clicked.connect(lambda: self.server_thread.send_msg({"type": "vision-end-test", "data": {"test_type": 1}}))
        self.btn_start_ocr.clicked.connect(lambda: self.server_thread.send_msg({"type": "vision-start-test", "data": {"test_type": 2}}))
        self.btn_stop_ocr.clicked.connect(lambda: self.server_thread.send_msg({"type": "vision-end-test", "data": {"test_type": 2}}))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ServerWindow()
    window.show()
    sys.exit(app.exec())