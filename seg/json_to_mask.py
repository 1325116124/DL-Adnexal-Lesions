import os
import cv2
import numpy as np
import json
from PIL import Image


def get_image_size(image_path):
    """
    获取图像的高度和宽度
    """
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    return height, width


def extract_points(data, image_width, image_height):
    """
    从JSON数据中提取多边形的坐标点，并根据图像尺寸进行缩放
    """
    points_list = []
    for item in data:
        content = item.get('content', [])
        rect_mask = item.get('rectMask', {})
        rect_width = rect_mask.get('width', 1)
        rect_height = rect_mask.get('height', 1)
        points = []
        for point in content:
            x = point.get('x')
            y = point.get('y')
            if x is not None and y is not None:
                # 缩放坐标
                scaled_x = int(x * image_width / rect_width)
                scaled_y = int(y * image_height / rect_height)
                points.append([scaled_x, scaled_y])
        if points:
            points = np.array(points, dtype=np.int32)
            points_list.append(points)
    print(f"提取的坐标点: {points_list}")  # 打印提取的坐标点
    return points_list


def json_to_mask(json_path, image_path, output_dir):
    """
    将标注的JSON文件转换为掩码图像并保存。

    参数:
    json_path (str): 标注的JSON文件路径
    image_path (str): 对应的原图路径
    output_dir (str): 保存掩码图像的目录路径
    """
    # 获取图像的高度和宽度
    height, width = get_image_size(image_path)
    print(f"图像尺寸: 高度 {height}, 宽度 {width}")  # 打印图像尺寸
    # 创建全黑的掩码图像
    mask = np.zeros((height, width), dtype=np.uint8)

    try:
        # 读取JSON文件
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except UnicodeDecodeError:
            try:
                with open(json_path, 'r', encoding='latin-1') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"无法读取文件 {json_path}: {e}")
                return

        # 提取多边形坐标点，并根据图像尺寸进行缩放
        points_list = extract_points(data, width, height)

        # 检查坐标点是否超出图像范围
        for points in points_list:
            for point in points:
                if point[0] < 0 or point[0] >= width or point[1] < 0 or point[1] >= height:
                    print(f"坐标点 {point} 超出图像范围！")

        # 填充多边形区域
        for points in points_list:
            cv2.fillPoly(mask, [points], 255)

        # 保存掩码图像
        mask_image = Image.fromarray(mask)
        mask_filename = os.path.splitext(os.path.basename(json_path))[0] + '_mask.png'
        mask_path = os.path.join(output_dir, mask_filename)
        mask_image.save(mask_path)
    except Exception as e:
        print(f"处理文件 {json_path} 时出错: {e}")


def process_directory(json_dir, image_dir, output_dir):
    """
    处理整个目录下的JSON文件和对应的图像文件。

    参数:
    json_dir (str): 标注的JSON文件目录路径
    image_dir (str): 原图目录路径
    output_dir (str): 保存掩码图像的目录路径
    """
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(json_dir):
        if file.endswith('.json'):  # 只处理JSON文件
            json_path = os.path.join(json_dir, file)
            base_name = os.path.splitext(file)[0]
            # 尝试匹配.png和.jpg格式的图像文件
            image_extensions = ['.png', '.jpg']
            found_image = False
            for ext in image_extensions:
                image_file = base_name + ext
                image_path = os.path.join(image_dir, image_file)
                if os.path.exists(image_path):
                    json_to_mask(json_path, image_path, output_dir)
                    found_image = True
                    break
            if not found_image:
                print(f"未找到对应的图像文件，跳过 {json_path}")

if __name__ == "__main__":
    json_directory = "/data2/yanghong/model/segment/json"  # 替换为你的JSON文件目录路径
    image_directory = "/data2/yanghong/model/segment/pic2"  # 替换为你的图像文件目录路径
    output_directory = "paper_pic"  # 替换为保存掩码图像的目录路径

    process_directory(json_directory, image_directory, output_directory)
    