import os
import random
from PIL import Image, ImageDraw


ANNOTATION_FILE = "./merged_annotations1.txt"
IMAGE_DIR = "./KITTI/images"
OUTPUT_DIR = "./or_box_test"

# 是否随机抽取图片
RANDOM_SELECT = False

# 如果不随机，则手动指定测试图片（写图片名即可）
CUSTOM_IMAGES = [
    "000001.png","000003.png",
    "000005.png","000099.png",
]

# 随机抽取数量
SAMPLE_NUM = 5


# =========================
# 读取标注文件
# =========================
def load_annotations(annotation_file):
    """
    格式：
    img_path 1 class_id lbx lby rtx rty
    """
    data = {}

    with open(annotation_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 7:
                continue

            img_path = parts[0]
            class_id = parts[2]

            try:
                lbx = float(parts[3])
                lby = float(parts[4])
                rtx = float(parts[5])
                rty = float(parts[6])
            except:
                continue

            if img_path not in data:
                data[img_path] = []

            data[img_path].append((class_id, lbx, lby, rtx, rty))

    return data


# =========================
# 坐标转换（左下 -> PIL左上）
# =========================
def convert_to_pil_coords(lbx, lby, rtx, rty, img_h):
    """
    PIL坐标系：左上角为原点，y向下
    你的坐标：左下角为原点，y向上
    """
    left = lbx
    right = rtx

    top = img_h - rty
    bottom = img_h - lby

    return left, top, right, bottom


# =========================
# 绘制检测框
# =========================
def draw_boxes(image_path, boxes, output_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size

    for cls_id, lbx, lby, rtx, rty in boxes:
        left, top, right, bottom = convert_to_pil_coords(
            lbx, lby, rtx, rty, img_h
        )

        # 画红框
        draw.rectangle(
            [left, top, right, bottom],
            outline="red",
            width=2
        )

        # 写类别
        draw.text((left, top), str(cls_id), fill="red")

    img.save(output_path)


# =========================
# 主流程
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_annotations(ANNOTATION_FILE)

    # 选择测试图片
    all_images = list(data.keys())

    if RANDOM_SELECT:
        selected_images = random.sample(all_images, min(SAMPLE_NUM, len(all_images)))
    else:
        selected_images = [
            os.path.join(IMAGE_DIR, name) for name in CUSTOM_IMAGES
        ]

    print(f"共选择 {len(selected_images)} 张图片进行测试")

    for img_path in selected_images:

        if not os.path.exists(img_path):
            print(f"图片不存在: {img_path}")
            continue

        if img_path not in data:
            print(f"无标注: {img_path}")
            continue

        boxes = data[img_path]

        filename = os.path.basename(img_path)
        save_path = os.path.join(OUTPUT_DIR, f"box_{filename}")

        draw_boxes(img_path, boxes, save_path)

        print(f"已保存: {save_path}")

    print("全部处理完成！")


if __name__ == "__main__":
    main()