import cv2

def capture_and_save():
    # 打开默认摄像头（索引0）
    cap = cv2.VideoCapture(2)
    
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
    
    # 读取一帧
    ret, frame = cap.read()
    
    if not ret:
        print("错误：无法读取帧")
        cap.release()
        return
    
    # 获取图像的尺寸（高度和宽度）
    height, width = frame.shape[:2]
    
    # 构造包含尺寸信息的文件名（例如：captured_640x480.jpg）
    filename = f"captured_{width}x{height}.jpg"
    
    # 保存图片到当前目录
    success = cv2.imwrite(filename, frame)
    
    if success:
        print(f"图片已保存为：{filename} (尺寸：{width} x {height})")
    else:
        print("保存图片失败")
    
    # 释放摄像头
    cap.release()

if __name__ == "__main__":
    capture_and_save()