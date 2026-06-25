import os
import cv2
import urllib.request

# =========================
# 路径配置
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIR   = os.path.join(BASE_DIR, "lfw", "images")
OUTPUT_FILE = os.path.join(BASE_DIR, "lfw", "face_detection_export.txt")

# OpenCV gender model 权重文件路径
MODEL_DIR        = os.path.join(BASE_DIR, "models")
GENDER_PROTO     = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
GENDER_MODEL     = os.path.join(MODEL_DIR, "gender_net.caffemodel")

# 下载地址
GENDER_PROTO_URL  = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt"
GENDER_MODEL_URL  = "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/gender_net.caffemodel"

GENDER_LIST = ["male", "female"]

# 人脸检测均值（OpenCV gender model 要求）
MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# 自动下载模型文件
# =========================
def download_if_missing(url, dest_path):
    if os.path.exists(dest_path):
        return
    print(f"正在下载: {os.path.basename(dest_path)} ...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"下载完成: {dest_path}")
    except Exception as e:
        print(f"下载失败: {url}\n错误: {e}")
        print("请手动下载并放置到:", dest_path)
        raise


# =========================
# bbox 转换（左上原点 → 左下原点）
# =========================
def to_bottom_left_bbox(x, y, w, h, img_h):
    x1 = x
    y1 = img_h - (y + h)
    x2 = x + w
    y2 = img_h - y
    return x1, y1, x2, y2


# =========================
# 性别预测
# =========================
def predict_gender(img_bgr, x, y, w, h, gender_net):
    """
    裁剪人脸区域，送入 gender_net 预测性别
    返回 'male' 或 'female'
    """
    # 适当扩展人脸裁剪区域，提升识别准确率
    pad   = 20
    img_h, img_w = img_bgr.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_w, x + w + pad)
    y2 = min(img_h, y + h + pad)

    face_crop = img_bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        return "male"   # 裁剪异常时给默认值

    blob = cv2.dnn.blobFromImage(
        face_crop, 1.0, (227, 227),
        MODEL_MEAN, swapRB=False
    )
    gender_net.setInput(blob)
    preds = gender_net.forward()           # shape: (1, 2)  [male, female]
    gender = GENDER_LIST[preds[0].argmax()]
    return gender


# =========================
# 主程序
# =========================
def main():
    # 下载模型（若不存在）
    download_if_missing(GENDER_PROTO_URL, GENDER_PROTO)
    download_if_missing(GENDER_MODEL_URL, GENDER_MODEL)

    # 加载人脸检测器
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # 加载性别分类模型
    gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
    print("模型加载完成")

    # 遍历所有子目录下的图片（LFW 按人名分子目录存放）
    image_paths = []
    for root, _, files in os.walk(IMAGE_DIR):
        for fname in files:
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                image_paths.append(os.path.join(root, fname))
    image_paths.sort()

    total   = len(image_paths)
    success = 0
    no_face = 0
    skipped = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for idx, img_path in enumerate(image_paths, 1):

            # 输出相对路径，格式统一为 ./lfw/images/子目录/文件名
            rel_path = "./lfw/images/" + os.path.relpath(
                img_path, IMAGE_DIR
            ).replace(os.sep, "/")

            img = cv2.imread(img_path)
            if img is None:
                print(f"[{idx}/{total}] [SKIP] {rel_path}")
                skipped += 1
                continue

            img_h, img_w = img.shape[:2]

            faces = face_cascade.detectMultiScale(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                scaleFactor=1.1,
                minNeighbors=5
            )

            # ── 没检测到人脸 ──
            if len(faces) == 0:
                out.write(f"{rel_path},0,0,0,0,0,0\n")
                print(f"[{idx}/{total}] [NO FACE] {rel_path}")
                no_face += 1
                continue

            # ── 选最大人脸 ──
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, fw, fh = faces[0]

            # ── 性别预测 ──
            gender = predict_gender(img, x, y, fw, fh, gender_net)

            # ── 坐标转换（左下原点） ──
            x1, y1, x2, y2 = to_bottom_left_bbox(x, y, fw, fh, img_h)

            # ── 写入结果 ──
            out.write(
                f"{rel_path},1,{gender},"
                f"{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}\n"
            )
            print(f"[{idx}/{total}] [OK] {rel_path}  gender={gender}")
            success += 1

    print(f"\nDONE -> {OUTPUT_FILE}")
    print(f"成功: {success}  无人脸: {no_face}  跳过: {skipped}  共: {total}")


if __name__ == "__main__":
    main()