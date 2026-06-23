# Internship — GuoPingYun机器人视觉识别系统

本项目为机器人视觉识别系统的客户端与数据处理模块，支持人脸识别、物体识别与文字识别（OCR），通过 WebSocket 与机器人本体通信，实时上报识别结果。

## 系统架构

```
机器人本体（摄像头）
        ↓
  vision_core.py        ← 视觉识别算法（人脸 / 物体 / OCR）
        ↓
    send.py             ← 数据发送模块（统一封装 JSON 协议格式）
        ↓
  机器人本体服务端       ← 接收指定格式识别结果
```

## 文件说明

### 客户端运行脚本

 `vision_core.py`   人脸识别、物体识别、文字识别算法主脚本，运行前请修改脚本内的服务端地址 `SERVER_URI` 
 `send.py`          数据发送模块，统一封装协议格式，向服务端发送 JSON 识别结果，与识别算法完全解耦 
 `c_pic.py`         拍摄一张图片并保存为 `captured_640x480.jpg`，用于发送给服务端进行图像校正 

### 原始数据集及服务端格式信息提取（通信协议具体参见https://docs.qq.com/aio/DUnBxcFRJdU5hcUVt?p=hyJZdbRjeXjLsJZTTtxsvg）

#### 人脸识别 — `./lfw`


 `./lfw/images/`                      人脸识别原始图片数据集（LFW） 
 `./lfw/result.txt`                   原始数据集标注信息 
 `./lfw/face_detection_export.txt`    提取后的服务端指定格式信息 

服务端格式示例：
```
原图片路径  检测到人脸数量  检测框左下x,y  检测框右上x,y
./lfw/images/AJ_Cook_0001.jpg 1 69.00 68.00 183.00 182.00
```

#### 物体识别 — `./KITTI`

| 路径 | 说明 |
|------|------|
| `./KITTI/images/` | 物体识别原始图片数据集（KITTI） |
| `./KITTI/labels/` | 原始数据集标注文件 |
| `./KITTI/merged_annotations.txt` | 提取后的服务端指定格式信息 |

服务端格式示例：
```
原图片路径  是否检测到物体  class_id  检测框左下x,y  检测框右上x,y
./KITTI/images/000000.png 1 3 712.40 62.08 810.73 227.00
```

### 格式提取与检测框验证 — `./convert`

| 文件 | 说明 |
|------|------|
| `convert_face.py` | 从 LFW 数据集提取服务端指定格式信息，输出至 `face_detection_export.txt` |
| `convert_or.py` | 从 KITTI 数据集提取服务端指定格式信息，输出至 `merged_annotations.txt` |
| `face_draw_box.py` | 将人脸检测框绘制到原图验证，结果保存至 `./face_box_test/` |
| `or_draw_box.py` | 将物体检测框绘制到原图验证，结果保存至 `./or_box_test/` |

---

## 快速开始

**1. 安装依赖**
```bash
pip install -r requirements.txt
```

**2. 修改服务端地址**

打开 `vision_core.py`，修改以下行为实际服务端地址：
```python
SERVER_URI = "ws://your_server_ip:port"
```

**3. 运行视觉识别客户端**
```bash
python vision_core.py
```

**4. 图像校正（可选）**
```bash
python c_pic.py
```

---

## 识别结果 JSON 格式

所有识别结果由 `send.py` 统一封装后上报，格式如下：

**人脸识别**
```json
{
  "type": "vision-face-recognition",
  "data": {
    "face_count": 1,
    "faces": [
      { "class_id": "face_detected", "confidence": 0.95,
        "bbox": { "x1": 120, "y1": 80, "x2": 160, "y2": 220 } }
    ]
  }
}
```

**物体识别**
```json
{
  "type": "vision-obj-recognition",
  "data": {
    "object_count": 1,
    "objects": [
      { "class_id": "cup", "confidence": 0.92,
        "bbox": { "x1": 120, "y1": 80, "x2": 160, "y2": 220 } }
    ]
  }
}
```

**文字识别（OCR）**
```json
{
  "type": "vision-ocr-recognition",
  "data": {
    "text_count": 1,
    "texts": [
      { "text": "欢迎使用机器人视觉测试系统", "language": "zh" }
    ]
  }
}
```
