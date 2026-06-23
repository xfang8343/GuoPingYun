"""
send.py — 视觉识别结果发送模块
职责：统一封装协议格式，向服务端发送 JSON 结果
与识别算法完全隔离，算法侧只需调用本模块的函数即可

可扩展：新增识别类型时，在本文件添加对应的 build_xxx / send_xxx 函数
"""

import json
import asyncio
import logging

logger = logging.getLogger("Send")

# ──────────────────────────────────────────────
# 协议类型常量（唯一定义处，算法侧从这里 import）
# ──────────────────────────────────────────────
MSG_FACE   = "vision-face-recognition"
MSG_OBJECT = "vision-obj-recognition"
MSG_OCR    = "vision-ocr-recognition"
MSG_DIST   = "vision-dis-measurement"


# ──────────────────────────────────────────────
# 内部工具：带重传的发送
# ──────────────────────────────────────────────
async def _send_with_retry(websocket, payload: dict, retries: int = 3, delay: float = 0.5):
    """
    带重传机制的底层发送函数
    :param websocket: 已连接的 websocket 对象
    :param payload:   要发送的字典（将被序列化为 JSON）
    :param retries:   最大重试次数
    :param delay:     每次重试间隔（秒）
    """
    msg = json.dumps(payload, ensure_ascii=False)
    for attempt in range(1, retries + 1):
        try:
            await websocket.send(msg)
            logger.debug(f"[Send] 发送成功（第{attempt}次）: {payload['type']}")
            return True
        except Exception as e:
            logger.warning(f"[Send] 第{attempt}次发送失败: {e}")
            if attempt < retries:
                await asyncio.sleep(delay)
    logger.error(f"[Send] 已达最大重试次数({retries})，放弃发送: {payload['type']}")
    return False


def _send_threadsafe(websocket, loop, payload: dict, retries: int = 3):
    """
    供同步线程（如 CV 线程）调用的线程安全发送入口
    内部通过 run_coroutine_threadsafe 提交到异步事件循环
    """
    asyncio.run_coroutine_threadsafe(
        _send_with_retry(websocket, payload, retries=retries),
        loop
    )


# ──────────────────────────────────────────────
# 协议构建函数（build_xxx）
# 只负责组装 dict，不涉及网络，便于单元测试
# ──────────────────────────────────────────────
def build_face_payload(faces: list) -> dict:
    """
    构建人脸识别上报包
    :param faces: list of dict，每项包含:
        class_id (str), confidence (float),
        bbox: {x1, y1, x2, y2} (int)
    """
    return {
        "type": MSG_FACE,
        "data": {
            "face_count": len(faces),
            "faces": [
                {
                    "class_id": str(f["class_id"]),
                    "confidence": round(float(f["confidence"]), 4),
                    "bbox": {
                        "x1": int(f["bbox"]["x1"]),
                        "y1": int(f["bbox"]["y1"]),
                        "x2": int(f["bbox"]["x2"]),
                        "y2": int(f["bbox"]["y2"]),
                    }
                }
                for f in faces
            ]
        }
    }


def build_object_payload(objects: list) -> dict:
    """
    构建物体识别上报包
    :param objects: list of dict，每项包含:
        class_id (str), confidence (float),
        bbox: {x1, y1, x2, y2} (int)
    """
    return {
        "type": MSG_OBJECT,
        "data": {
            "object_count": len(objects),
            "objects": [
                {
                    "class_id": str(o["class_id"]),
                    "confidence": round(float(o["confidence"]), 4),
                    "bbox": {
                        "x1": int(o["bbox"]["x1"]),
                        "y1": int(o["bbox"]["y1"]),
                        "x2": int(o["bbox"]["x2"]),
                        "y2": int(o["bbox"]["y2"]),
                    }
                }
                for o in objects
            ]
        }
    }


def build_ocr_payload(texts: list) -> dict:
    """
    构建文字识别上报包
    :param texts: list of dict，每项包含:
        text (str), language (str, 如 'zh'/'en')
    """
    return {
        "type": MSG_OCR,
        "data": {
            "text_count": len(texts),
            "texts": [
                {
                    "text": str(t["text"]),
                    "language": str(t.get("language", "zh"))
                }
                for t in texts
            ]
        }
    }


def build_distance_payload(value: float, unit: str = "m") -> dict:
    """构建距离测量上报包"""
    return {
        "type": MSG_DIST,
        "data": {
            "distance_value": round(float(value), 4),
            "distance_unit": str(unit)
        }
    }


# ──────────────────────────────────────────────
# 对外发送函数（send_xxx）
# 算法侧调用这些函数，传入结构化数据即可
# ──────────────────────────────────────────────
def send_face(websocket, loop, faces: list, retries: int = 3):
    """发送人脸识别结果（供同步线程调用）"""
    _send_threadsafe(websocket, loop, build_face_payload(faces), retries)


def send_object(websocket, loop, objects: list, retries: int = 3):
    """发送物体识别结果（供同步线程调用）"""
    _send_threadsafe(websocket, loop, build_object_payload(objects), retries)


def send_ocr(websocket, loop, texts: list, retries: int = 3):
    """发送文字识别结果（供同步线程调用）"""
    _send_threadsafe(websocket, loop, build_ocr_payload(texts), retries)


def send_distance(websocket, loop, value: float, unit: str = "m", retries: int = 3):
    """发送距离测量结果（供同步线程调用）"""
    _send_threadsafe(websocket, loop, build_distance_payload(value, unit), retries)


# ── 异步版本（供 async 上下文直接 await 调用）──
async def async_send_face(websocket, faces: list, retries: int = 3):
    return await _send_with_retry(websocket, build_face_payload(faces), retries)

async def async_send_object(websocket, objects: list, retries: int = 3):
    return await _send_with_retry(websocket, build_object_payload(objects), retries)

async def async_send_ocr(websocket, texts: list, retries: int = 3):
    return await _send_with_retry(websocket, build_ocr_payload(texts), retries)

async def async_send_distance(websocket, value: float, unit: str = "m", retries: int = 3):
    return await _send_with_retry(websocket, build_distance_payload(value, unit), retries)