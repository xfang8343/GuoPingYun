import os
import glob
from PIL import Image

def convert_bbox_center_to_corners(center_x, center_y, width, height, img_w, img_h):
    """
    将 YOLO 归一化坐标（原点左上，x右y下）转换为像素坐标的左下角和右上角
    目标坐标系：原点在图像左下角，x 向右，y 向上
    返回：左下x, 左下y, 右上x, 右上y（像素坐标）
    """
    # 第一步：还原为左上角原点的像素坐标
    left   = (center_x - width  / 2.0) * img_w
    top    = (center_y - height / 2.0) * img_h
    right  = (center_x + width  / 2.0) * img_w
    bottom = (center_y + height / 2.0) * img_h

    # clamp 到图像范围内，防止标注越界导致坐标异常
    left   = max(0.0, min(left,   img_w))
    right  = max(0.0, min(right,  img_w))
    top    = max(0.0, min(top,    img_h))
    bottom = max(0.0, min(bottom, img_h))

    # 第二步：转换到左下角原点坐标系（y 向上）
    lbx = left
    lby = img_h - bottom   # 原来的下边变成左下角的 y（靠近原点，值小）
    rtx = right
    rty = img_h - top      # 原来的上边变成右上角的 y（远离原点，值大）

    return lbx, lby, rtx, rty


def find_image_path(base_name, image_dir):
    """
    优化1：按优先级尝试多种扩展名，返回第一个存在的图片路径；找不到则返回 None
    """
    for ext in ('.png', '.jpg', '.jpeg'):
        img_path = os.path.join(image_dir, f"{base_name}{ext}")
        if os.path.exists(img_path):
            return img_path
    return None


def process_files(input_pattern, output_file, image_dir="./KITTI/images"):
    """
    从标签文件读取 YOLO 归一化坐标，根据对应图片真实分辨率转换为左下角+右上角像素坐标
    """
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    file_list = glob.glob(input_pattern)
    file_list.sort()

    skipped = 0
    # 缓存已读取过的图片尺寸，避免重复 IO
    img_size_cache = {}

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file_path in file_list:
            base_name = os.path.splitext(os.path.basename(file_path))[0]

            # 自动匹配扩展名
            img_path = find_image_path(base_name, image_dir)
            if img_path is None:
                print(f"警告: 找不到图片 {base_name}（已尝试 .png/.jpg/.jpeg），跳过。")
                skipped += 1
                continue

            # 命中缓存则直接取，否则读取并写入缓存
            if img_path not in img_size_cache:
                try:
                    with Image.open(img_path) as img:
                        img_size_cache[img_path] = img.size
                except Exception as e:
                    print(f"警告: 无法读取图片 {img_path}，跳过。错误: {e}")
                    skipped += 1
                    continue

            img_w, img_h = img_size_cache[img_path]

            with open(file_path, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) != 5:
                        print(f"警告: {file_path} 行格式错误，跳过: {line}")
                        continue

                    class_id = parts[0]
                    try:
                        cx = float(parts[1])
                        cy = float(parts[2])
                        w  = float(parts[3])
                        h  = float(parts[4])
                    except ValueError:
                        print(f"警告: {file_path} 数据无法转换，跳过: {line}")
                        continue

                    lbx, lby, rtx, rty = convert_bbox_center_to_corners(
                        cx, cy, w, h, img_w, img_h
                    )

                    # 输出：图片路径 1 类别 左下x 左下y 右上x 右上y    //‘1’表示检测到图片中有物体
                    out_f.write(
                        f"{img_path} 1 {class_id} "
                        f"{lbx:.2f} {lby:.2f} {rtx:.2f} {rty:.2f}\n"
                    )

    print(f"转换完成，共处理 {len(file_list) - skipped} 张图片，结果保存至: {output_file}")
    if skipped > 0:
        print(f"警告: {skipped} 张图片无法读取或找不到，已跳过")


if __name__ == "__main__":
    input_files = "./KITTI/labels/*.txt"
    output      = "./KITTI/merged_annotations.txt"
    process_files(input_files, output)