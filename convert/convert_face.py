import os
import cv2

# =========================
# 路径配置
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIR = os.path.join(BASE_DIR, "lfw", "images")
OUTPUT_FILE = os.path.join(BASE_DIR, "lfw", "face_detection_export.txt")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# =========================
# bbox转换（左下原点）
# =========================
def to_bottom_left_bbox(x, y, w, h, img_h):
    x1 = x
    y1 = img_h - (y + h)
    x2 = x + w
    y2 = img_h - y
    return x1, y1, x2, y2


# =========================
# 主程序
# =========================
def main():

    images = [f for f in os.listdir(IMAGE_DIR)
              if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    images.sort()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

        for img_name in images:

            img_path = os.path.join(IMAGE_DIR, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print("[SKIP]", img_name)
                continue

            h, w = img.shape[:2]

            faces = face_cascade.detectMultiScale(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                1.1,
                5
            )

            # =========================
            # 没检测到人脸
            # =========================
            if len(faces) == 0:
                out.write(f"./lfw/images/{img_name} 0 -1 -1 -1 -1\n")
                continue

            # =========================
            # 选最大人脸
            # =========================
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, fw, fh = faces[0]

            x1, y1, x2, y2 = to_bottom_left_bbox(x, y, fw, fh, h)

            # =========================
            # 输出（无性别字段）
            # =========================
            out.write(
                f"./lfw/images/{img_name} 1 "
                f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n"
            )

            print("[OK]", img_name)

    print("\nDONE ->", OUTPUT_FILE)


if __name__ == "__main__":
    main()