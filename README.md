# 机器人视觉识别系统

## 项目结构
```
GuoPingYun/
├── vision_core.py          # 视觉识别主逻辑（摄像头采集 + 识别 + 预览）
├── send.py                 # WebSocket 发送接口（封装协议格式）
│
├── convert/
│   ├── convert_face.py     # 人脸数据集信息提取 → face_detection_export.txt
│   ├── convert_or.py       # 物体数据集信息提取 → merged_annotations.txt
│   ├── face_draw_box.py    # 绘制人脸检测框验证，输出到 ./face_box_test/
│   ├── or_draw_box.py      # 绘制物体检测框验证，输出到 ./or_box_test/
│   └── models/
│       ├── gender_deploy.prototxt   # 性别分类模型结构
│       └── gender_net.caffemodel    # 性别分类模型权重（~44MB）
│
├── lfw/
│   ├── images/             # 人脸识别原始图片数据集
│   ├── result.txt          # 原始数据集信息
│   └── face_detection_export.txt   # 提取后的服务端指定格式
│
└── KITTI/
    ├── images/             # 物体识别原始图片数据集
    ├── labels/             # 原始标注文件（YOLO格式 *.txt）
    └── merged_annotations.txt      # 提取后的服务端指定格式
```

---

## 客户端脚本

| 脚本 | 说明 |
|------|------|
| `c_pic.py` | 拍摄一张图片，保存为 `captured_640x480.jpg`，用于发送给服务端做校正 |
| `c2_ocr.py` | 包含人脸识别、物体识别、文字识别，注意修改服务端地址 |

---

## 视觉识别主脚本

### vision_core.py

摄像头采集与视觉识别主逻辑，通过 WebSocket 接收服务端指令，启动对应识别模式，结果通过 `send.py` 发送回服务端。

**支持的识别模式：**

| 模式代码 | 功能 | 说明 |
|----------|------|------|
| `0` | 人脸识别 + 性别分类 | 检测人脸并识别性别（male / female / unknown），结果体现在 `class_id` 字段 |
| `1` | 物体识别 | 使用 YOLOv5n 检测画面中的物体，输出类别与置信度 |
| `2` | 文字识别（OCR） | 使用 RapidOCR 识别画面中的中英文文字 |
| `3` | 距离测量 | 预留，暂未实现 |

**性别分类模型：**
- 使用 OpenCV DNN 加载 `gender_net.caffemodel`
- 模型文件路径：`./convert/models/`
- 置信度低于 `0.5` 时输出 `unknown`
- 仅在模式 `0` 激活时加载，不占用其他模式资源

### send.py

封装 WebSocket 协议格式，提供 `send_face` / `send_object` / `send_ocr` / `send_distance` 接口，`vision_core.py` 通过调用这些接口发送识别结果，不直接操作协议细节。

---

## 原始数据集及服务端格式说明

### 人脸识别（LFW）

提取脚本：`convert/convert_face.py`
输出文件：`lfw/face_detection_export.txt`

**格式：**
```
图片路径,是否检测到人脸,性别,左下x,左下y,右上x,右上y
```

**示例：**
```
./lfw/images/AJ_Cook_0001.jpg,1,female,69.00,68.00,183.00,182.00
./lfw/images/AJ_Cook_0002.jpg,0,0,0,0,0,0
```

> 坐标系：原点在图像**左下角**，x 向右，y 向上（像素坐标）
> 未检测到人脸时，性别与坐标字段均填 `0`

---

### 物体识别（KITTI）

提取脚本：`convert/convert_or.py`
输出文件：`KITTI/merged_annotations.txt`

**格式：**
```
图片路径 是否检测到物体 classid 左下x 左下y 右上x 右上y
```

**示例：**
```
./KITTI/images/000000.png 1 3 712.40 62.08 810.73 227.00
```

> 坐标系：原点在图像**左下角**，x 向右，y 向上（像素坐标）

---

## 检测框绘制验证

| 脚本 | 输入 | 输出 |
|------|------|------|
| `convert/face_draw_box.py` | `lfw/face_detection_export.txt` + `lfw/images/` | `convert/face_box_test/` |
| `convert/or_draw_box.py` | `KITTI/merged_annotations.txt` + `KITTI/images/` | `convert/or_box_test/` |

支持随机抽取或手动指定图片，在原图上绘制红色检测框用于人工验证坐标转换是否正确。
```
