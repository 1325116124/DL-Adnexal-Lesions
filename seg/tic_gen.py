import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from model2 import UNet  # 假设您有一个分割模型


def load_model(model_path, device):
    model = UNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


class TICGenerator:
    def __init__(self, model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = load_model(model_path, device)
        self.model.eval()
        self.device = device

    def process_video(self, video_path, window_size=5, threshold=3.0):
        """处理视频并生成TIC曲线数据
        Args:
            window_size: 滑动窗口大小
            threshold: 异常值过滤阈值(标准差倍数)
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        tic_values = []
        raw_values = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 转换为灰度图并预处理
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = frame_gray.astype(np.float32) / 255.0
            frame_gray = cv2.resize(frame_gray, (224, 224))
            frame_tensor = torch.from_numpy(frame_gray).unsqueeze(0).unsqueeze(0).float().to(self.device)

            # 使用模型预测分割区域
            with torch.no_grad():
                mask = self.model(frame_tensor)
            mask = mask.squeeze().cpu().numpy() > 0.5

            # 计算分割区域的平均灰度值
            masked_frame = frame_gray * mask
            mean_value = np.mean(masked_frame[mask > 0])
            raw_values.append(mean_value)

        cap.release()

        # 异常值过滤
        if len(raw_values) > 0:
            mean = np.mean(raw_values)
            std = np.std(raw_values)
            tic_values = [x if abs(x - mean) < threshold * std else mean for x in raw_values]

            # 滑动窗口平均
            if window_size > 1 and len(tic_values) > window_size:
                window = np.ones(window_size) / window_size
                tic_values = np.convolve(tic_values, window, mode='same')

        return tic_values, fps

    def plot_tic_curve(self, tic_values, fps, output_path=None, show_raw=False, raw_values=None):
        """绘制TIC曲线并保存
        Args:
            show_raw: 是否显示原始数据
            raw_values: 原始数据(当show_raw为True时需要)
        """
        time_points = np.arange(len(tic_values)) / fps

        plt.figure(figsize=(10, 6))
        plt.plot(time_points, tic_values, 'b-', label='平滑后曲线')
        if show_raw and raw_values is not None:
            plt.plot(time_points, raw_values, 'r--', alpha=0.3, label='原始数据')
            plt.legend()
        plt.title('Time-Intensity Curve (TIC)')
        plt.xlabel('Time (s)')
        plt.ylabel('Intensity')
        plt.xticks(fontsize=11)
        # 设置y轴刻度间隔为0.5
        y_min = np.min(tic_values)
        y_max = np.max(tic_values)
        plt.yticks(np.arange(np.floor(y_min), y_max + 0.075, 0.075), fontsize=11)
        plt.grid(True, alpha=0.3)

        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()
        plt.close()


if __name__ == "__main__":
    model_path = "/data2/yanghong/model/segment/checkpoints/best_model.pth"
    video_directory = Path("/home/data/yanghong/AI返修-2025.2/其他中心-良性")
    output_directory = Path("tic_images4")
    output_directory.mkdir(exist_ok=True)

    generator = TICGenerator(model_path)

    for video_path in video_directory.glob("*.mp4"):
        tic_data, fps = generator.process_video(video_path)
        output_path = output_directory / f"{video_path.stem}_tic_curve.png"
        generator.plot_tic_curve(tic_data, fps, output_path)

    