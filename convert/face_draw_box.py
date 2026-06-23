import os
import random
from PIL import Image, ImageDraw

# =========================
# 配置
# =========================
ANNOTATION_FILE = "./lfw/face_detection_export.txt"
IMAGE_DIR = "./lfw/images"
OUTPUT_DIR = "./face_box_test"

RANDOM_SELECT = True
SAMPLE_NUM = 55

CUSTOM_IMAGES = [

    "Reese_Witherspoon_0001.jpg",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# 读取标注
# =========================
def load_annotations(annotation_file):
    data = {}

    with open(annotation_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            # 格式：
            # path flag x1 y1 x2 y2
            if len(parts) != 6:
                continue

            img_path = parts[0]
            flag = int(parts[1])

            try:
                x1 = float(parts[2])
                y1 = float(parts[3])
                x2 = float(parts[4])
                y2 = float(parts[5])
            except:
                continue

            if img_path not in data:
                data[img_path] = []

            data[img_path].append((flag, x1, y1, x2, y2))

    return data


# =========================
# 坐标转换（左下 -> PIL左上）
# =========================
def convert_coords(x1, y1, x2, y2, img_h):
    left = x1
    right = x2
    top = img_h - y2
    bottom = img_h - y1
    return left, top, right, bottom


# =========================
# 绘制
# =========================
def draw_boxes(img_path, boxes, save_path):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    img_w, img_h = img.size

    for flag, x1, y1, x2, y2 in boxes:

        # 如果未检测到人脸，跳过画框
        if flag == 0:
            continue

        left, top, right, bottom = convert_coords(x1, y1, x2, y2, img_h)

        # 红色检测框
        draw.rectangle(
            [left, top, right, bottom],
            outline="red",
            width=2
        )

    img.save(save_path)


# =========================
# 主函数
# =========================
def main():

    data = load_annotations(ANNOTATION_FILE)

    all_images = list(data.keys())

    # =========================
    # 选择图片
    # =========================
    if RANDOM_SELECT:
        selected_images = random.sample(
            all_images,
            min(SAMPLE_NUM, len(all_images))
        )
    else:
        selected_images = [
            os.path.join(IMAGE_DIR, name)
            for name in CUSTOM_IMAGES
        ]

    print(f"共选择 {len(selected_images)} 张图片")

    for img_path in selected_images:

        if not os.path.exists(img_path):
            print("不存在:", img_path)
            continue

        if img_path not in data:
            print("无标注:", img_path)
            continue

        boxes = data[img_path]

        filename = os.path.basename(img_path)
        save_path = os.path.join(OUTPUT_DIR, f"box_{filename}")

        draw_boxes(img_path, boxes, save_path)

        print("已保存:", save_path)

    print("\nDONE ->", OUTPUT_DIR)


if __name__ == "__main__":
    main()