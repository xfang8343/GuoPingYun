"""
vision_core.py — 机器人视觉识别主逻辑
职责：摄像头采集、人脸/物体/OCR 识别、绘制预览
发送结果：统一调用 send.py 提供的接口，不直接操作 WebSocket 协议格式
"""

import asyncio
import websockets
import json
import logging
import cv2
import numpy as np
import time
import os
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from rapidocr_onnxruntime import RapidOCR

# ── 导入 send.py 的发送接口 ──────────────────────
from send import send_face, send_object, send_ocr, send_distance

# ── 模型加载 ────────────────────────────────────
model      = YOLO('yolov5n.pt')
ocr_engine = RapidOCR()

# ── 日志配置 ────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RobotClient")

# ── 连接配置 ────────────────────────────────────
SERVER_URI         = "ws://127.0.0.1:5000"
RECONNECT_INTERVAL = 5
HEARTBEAT_INTERVAL = 3

# ── 协议控制常量 ─────────────────────────────────
MSG_TYPE_HEARTBEAT = "heartbeat"
MSG_TYPE_START     = "vision-start-test"
MSG_TYPE_END       = "vision-end-test"

# ── 性别模型配置 ─────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR        = os.path.join(BASE_DIR, "convert", "models")
GENDER_PROTO     = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
GENDER_MODEL     = os.path.join(MODEL_DIR, "gender_net.caffemodel")
GENDER_LIST      = ["male", "female"]
GENDER_MEAN      = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_THRESHOLD = 0.5   # 低于此置信度输出 unknown

# ── 全局状态 ─────────────────────────────────────
is_running_face_test = False
current_test_type    = -1   # -1:无, 0:人脸, 1:物体, 2:OCR, 3:距离

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ── 中文字体加载 ──────────────────────────────────
def load_chinese_font(font_size=20):
    font_paths = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                continue
    print("[警告] 未找到中文字体，中文可能无法显示")
    return ImageFont.load_default()

FONT = load_chinese_font(18)


def cv2_draw_chinese(img, text, position, text_color=(0, 255, 0), bg_color=None, font=None):
    """在 OpenCV BGR 图像上绘制中英文文本（通过 PIL 中转）"""
    if font is None:
        font = FONT

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(pil_img)

    bbox   = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0] + 4
    text_h = bbox[3] - bbox[1] + 4

    x, y = position
    if bg_color is not None:
        draw.rectangle([x, y, x + text_w, y + text_h], fill=bg_color[::-1])

    draw.text((x + 2, y + 2), text, font=font, fill=text_color[::-1])
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ── 性别模型加载 ──────────────────────────────────
def load_gender_net():
    """加载 gender_net，失败时返回 None 并打印警告"""
    if not os.path.exists(GENDER_PROTO) or not os.path.exists(GENDER_MODEL):
        logger.warning(
            f"性别模型文件缺失，请确认路径:\n  {GENDER_PROTO}\n  {GENDER_MODEL}"
        )
        return None
    try:
        net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
        logger.info("性别分类模型加载成功")
        return net
    except Exception as e:
        logger.warning(f"性别模型加载失败: {e}")
        return None


# ── 性别预测 ──────────────────────────────────────
def predict_gender(img_bgr, x, y, w, h, gender_net):
    """
    裁剪人脸区域送入 gender_net 推理
    返回: ('male'|'female'|'unknown', confidence)
    """
    pad   = 20
    ih, iw = img_bgr.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(iw, x + w + pad)
    y2 = min(ih, y + h + pad)

    face_crop = img_bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        return "unknown", 0.0

    try:
        blob  = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), GENDER_MEAN, swapRB=False)
        gender_net.setInput(blob)
        preds = gender_net.forward()          # shape: (1, 2) [male, female]
        conf  = float(preds[0].max())
        label = GENDER_LIST[preds[0].argmax()]
        if conf < GENDER_THRESHOLD:
            return "unknown", conf
        return label, conf
    except Exception as e:
        logger.warning(f"性别推理异常: {e}")
        return "unknown", 0.0


# ── CV 主循环（在独立线程中运行）────────────────────
def run_cv_logic(websocket, loop):
    global is_running_face_test, current_test_type

    cap = cv2.VideoCapture(0)
    logger.info(f"CV 线程启动，当前模式: {current_test_type}")

    target_fps     = 10
    frame_duration = 1.0 / target_fps

    # ── 模式 0 时才加载性别模型 ──────────────────
    gender_net = None
    if current_test_type == 0:
        gender_net = load_gender_net()

    try:
        while is_running_face_test:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # ── 模式 0：人脸识别 + 性别分类 ──────────
            if current_test_type == 0:
                if not hasattr(run_cv_logic, 'face_cache'):
                    run_cv_logic.face_cache = None
                if not hasattr(run_cv_logic, 'face_last_send_time'):
                    run_cv_logic.face_last_send_time = 0

                frame_height, frame_width = frame.shape[:2]
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

                if len(faces) > 0:
                    (x, y, w, h) = faces[0]

                    # 性别预测
                    if gender_net is not None:
                        gender_label, gender_conf = predict_gender(
                            frame, x, y, w, h, gender_net
                        )
                    else:
                        gender_label, gender_conf = "unknown", 0.0

                    run_cv_logic.face_cache = (x, y, w, h, frame_height, gender_label, gender_conf)

                    # 绘制检测框 + 性别标签
                    box_color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                    frame = cv2_draw_chinese(
                        frame,
                        f"{gender_label} {gender_conf:.2f}",
                        (x, max(0, y - 25)),
                        text_color=(0, 0, 0),
                        bg_color=box_color,
                        font=FONT
                    )
                else:
                    run_cv_logic.face_cache = None

                now = time.time()
                if now - run_cv_logic.face_last_send_time >= 0.2:
                    run_cv_logic.face_last_send_time = now
                    if run_cv_logic.face_cache is not None:
                        x, y, w, h, fh, gender_label, gender_conf = run_cv_logic.face_cache
                        send_face(websocket, loop, faces=[{
                            "class_id":   gender_label,   # male / female / unknown
                            "confidence": gender_conf,
                            "bbox": {
                                "x1": x,
                                "y1": fh - y - h,
                                "x2": x + w,
                                "y2": fh - y
                            }
                        }])
                        logger.info(f"人脸发送: class_id={gender_label}, conf={gender_conf:.2f}")

            # ── 模式 1：物体识别 ──────────────────────
            elif current_test_type == 1:
                if not hasattr(run_cv_logic, 'obj_cache'):
                    run_cv_logic.obj_cache = []
                if not hasattr(run_cv_logic, 'obj_last_send_time'):
                    run_cv_logic.obj_last_send_time = 0

                frame_height, frame_width = frame.shape[:2]
                results = model.predict(frame, verbose=False)[0]

                detected_objects = []
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf  = box.conf.item()
                    label = model.names[box.cls.item()]
                    logger.info(f"当前class: {label}")

                    detected_objects.append({
                        "class_id":   label,
                        "confidence": conf,
                        "bbox": {
                            "x1": x1,
                            "y1": frame_height - y2,
                            "x2": x2,
                            "y2": frame_height - y1
                        }
                    })

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                run_cv_logic.obj_cache = detected_objects

                now = time.time()
                if now - run_cv_logic.obj_last_send_time >= 0.2:
                    run_cv_logic.obj_last_send_time = now
                    send_object(websocket, loop, objects=run_cv_logic.obj_cache)

            # ── 模式 2：文字识别（OCR）───────────────
            elif current_test_type == 2:
                if not hasattr(run_cv_logic, 'ocr_cache'):
                    run_cv_logic.ocr_cache = []
                if not hasattr(run_cv_logic, 'last_send_time'):
                    run_cv_logic.last_send_time = 0
                if not hasattr(run_cv_logic, 'last_ocr_time'):
                    run_cv_logic.last_ocr_time = 0

                now = time.time()

                if now - run_cv_logic.last_ocr_time >= 0.5:
                    run_cv_logic.last_ocr_time = now
                    ocr_result, _ = ocr_engine(frame)

                    new_cache = []
                    if ocr_result:
                        for line in ocr_result:
                            box_pts, text_content, score = line[0], line[1], line[2]
                            if score < 0.5:
                                continue
                            cjk   = sum(1 for c in text_content if '\u4e00' <= c <= '\u9fff')
                            latin = sum(1 for c in text_content if 'a' <= c.lower() <= 'z')
                            lang  = "zh" if cjk >= latin else "en"
                            new_cache.append({
                                'box':  box_pts,
                                'text': text_content,
                                'score': score,
                                'lang': lang
                            })
                    run_cv_logic.ocr_cache = new_cache

                if now - run_cv_logic.last_send_time >= 1:
                    run_cv_logic.last_send_time = now
                    send_ocr(websocket, loop, texts=[
                        {"text": item['text'], "language": item['lang']}
                        for item in run_cv_logic.ocr_cache
                    ])

                for item in run_cv_logic.ocr_cache:
                    box_pts      = item['box']
                    text_content = item['text']
                    score        = item['score']

                    xs = [p[0] for p in box_pts]
                    ys = [p[1] for p in box_pts]
                    x_min, y_min = int(min(xs)), int(min(ys))
                    x_max, y_max = int(max(xs)), int(max(ys))

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    frame = cv2_draw_chinese(
                        frame,
                        f"{text_content} {score:.2f}",
                        (x_min, y_min - 25),
                        text_color=(0, 0, 0),
                        bg_color=(0, 255, 0),
                        font=FONT
                    )

                cv2.putText(frame, "OCR Running...", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # ── 模式 3：距离测量（预留）──────────────
            elif current_test_type == 3:
                pass

            # ── 显示与帧率控制 ──────────────────────
            cv2.imshow("Robot Camera View", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                is_running_face_test = False

            elapsed    = time.time() - start_time
            sleep_time = frame_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("视觉测试已停止，摄像头资源已释放")


# ── 消息接收处理 ─────────────────────────────────
async def receive_handler(websocket):
    global is_running_face_test, current_test_type

    async for message in websocket:
        msg_data = json.loads(message)
        msg_type = msg_data.get("type")
        data     = msg_data.get("data", {})

        if msg_type == MSG_TYPE_HEARTBEAT:
            logger.info("收到心跳包")

        elif msg_type == MSG_TYPE_START:
            current_test_type    = int(data.get("test_type"))
            is_running_face_test = True
            logger.info(f"启动视觉测试，模式代码: {current_test_type}")

            loop = asyncio.get_running_loop()
            asyncio.create_task(asyncio.to_thread(run_cv_logic, websocket, loop))

        elif msg_type == MSG_TYPE_END:
            current_test_type    = -1
            is_running_face_test = False
            logger.info("停止视觉测试")


# ── 心跳发送 ────────────────────────────────────
async def send_heartbeat(websocket):
    while True:
        try:
            await websocket.send(json.dumps({"type": MSG_TYPE_HEARTBEAT}))
            await asyncio.sleep(HEARTBEAT_INTERVAL)
        except Exception:
            break


# ── 主入口：自动重连 ─────────────────────────────
async def run_client():
    while True:
        try:
            async with websockets.connect(SERVER_URI) as websocket:
                await asyncio.gather(
                    send_heartbeat(websocket),
                    receive_handler(websocket)
                )
        except Exception as e:
            logger.error(f"连接失败: {e}，{RECONNECT_INTERVAL}秒后重试...")
            await asyncio.sleep(RECONNECT_INTERVAL)


if __name__ == "__main__":
    asyncio.run(run_client())