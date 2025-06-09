import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import pydicom  # DICOM处理库


def load_model(model_path, device):
    """加载分割模型（假设模型定义在model2.py中）"""
    from model2 import UNet  # 确保model2.py与当前文件同目录
    model = UNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


class TICGenerator:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = load_model(model_path, self.device)
        self.model.eval()

    def process_dcm(self, dicom_dir, window_size=5, threshold=3.0):
        """处理单个目录下的`造影动态图.dcm`文件，生成TIC数据"""
        # 定位目标DICOM文件
        target_dcm = dicom_dir / "造影动态图.dcm"
        if not target_dcm.exists():
            raise FileNotFoundError(f"未找到文件：{target_dcm}")

        # 读取DICOM文件（支持多帧）
        ds = pydicom.dcmread(target_dcm)
        # 提取多帧像素数据（shape: [frames, H, W]）
        if len(ds.pixel_array.shape) == 2:  # 单帧扩展为多帧
            frames_pixel = ds.pixel_array[np.newaxis, :, :]
        else:
            frames_pixel = ds.pixel_array

        # 转换像素值（Rescale Slope/Intercept）并归一化
        frames = []
        for frame_pixel in frames_pixel:
            pixel_array = frame_pixel * ds.RescaleSlope + ds.RescaleIntercept  # 像素值转换
            pixel_array = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # 归一化到0-255
            frame = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2BGR)  # 转换为BGR格式（适配后续处理）
            frames.append(frame)

        # 获取帧率（优先从DICOM标签获取，否则默认25）
        fps = ds.get("FrameRate", 25)

        # 计算每帧的平均灰度值（通过分割模型过滤）
        raw_values = []
        for frame in frames:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转为灰度图
            frame_gray = frame_gray.astype(np.float32) / 255.0  # 归一化到0-1
            frame_gray = cv2.resize(frame_gray, (224, 224))  # 调整尺寸到模型输入大小
            frame_tensor = torch.from_numpy(frame_gray).unsqueeze(0).unsqueeze(0).to(self.device)  # 构造输入张量

            with torch.no_grad():
                mask = self.model(frame_tensor)  # 模型预测分割掩码
            mask = mask.squeeze().cpu().numpy() > 0.5  # 二值化掩码

            masked_frame = frame_gray * mask  # 应用掩码过滤
            if mask.sum() > 0:  # 避免除零错误
                mean_value = np.mean(masked_frame[mask > 0])
            else:
                mean_value = 0.0  # 无有效区域时设为0
            raw_values.append(mean_value)

        # 异常值过滤（基于标准差）
        if len(raw_values) > 0:
            mean = np.mean(raw_values)
            std = np.std(raw_values)
            tic_values = [x if abs(x - mean) < threshold * std else mean for x in raw_values]

            # 滑动窗口平滑（仅当窗口大小有效时）
            if window_size > 1 and len(tic_values) > window_size:
                window = np.ones(window_size) / window_size
                tic_values = np.convolve(tic_values, window, mode='same')

        return tic_values, fps

    def plot_tic_curve(self, tic_values, fps, output_path, show_raw=False, raw_values=None):
        """绘制TIC曲线并保存"""
        time_points = np.arange(len(tic_values)) / fps  # 时间轴（秒）

        plt.figure(figsize=(10, 6))
        plt.plot(time_points, tic_values, 'b-', label='平滑后曲线')
        if show_raw and raw_values is not None:
            plt.plot(time_points, raw_values, 'r--', alpha=0.3, label='原始数据')
            plt.legend()

        plt.title('Time-Intensity Curve (TIC)')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Intensity')
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.grid(True, alpha=0.3)

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # 配置参数（根据实际路径修改）
    model_path = "/data2/yanghong/model/segment/checkpoints/best_model.pth"  # 分割模型路径
    txt_path = Path("/home/data/yanghong/valset_out.txt.txt")  # 存放目标目录名称的txt文件路径（与当前文件同目录）
    data_dir = Path("data")            # 存放数据的根目录（与当前文件同目录下的data目录）
    output_dir = Path("valset_out_tic") # 输出TIC图像的目录
    output_dir.mkdir(exist_ok=True)    # 创建输出目录（若不存在）

    # 读取txt中的目标目录名称（每行一个名称）
    with open(txt_path, 'r', encoding='utf-8') as f:
        target_names = [line.strip() for line in f.readlines() if line.strip()]  # 过滤空行

    generator = TICGenerator(model_path)

    # 遍历每个目标目录，处理其中的`造影动态图.dcm`
    for name in target_names:
        # 构造目标目录路径
        dicom_dir = data_dir / name
        if not dicom_dir.is_dir():
            print(f"警告：目录 {dicom_dir} 不存在，跳过处理")
            continue

        # 处理DICOM文件并生成TIC曲线
        try:
            tic_values, fps = generator.process_dcm(dicom_dir)
        except FileNotFoundError as e:
            print(f"警告：{e}，跳过处理")
            continue

        # 生成输出图像路径（命名为txt中的名称）
        output_path = output_dir / f"{name}_tic_curve.png"
        generator.plot_tic_curve(tic_values, fps, output_path, show_raw=True, raw_values=tic_values)  # 显示原始数据
        print(f"已生成：{output_path}")