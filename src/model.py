import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from datetime import datetime
from SkipGram import WebSessionEmbedder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
from tqdm import tqdm


class SNNBlock(nn.Module):
    """残差网络块，使用1x1卷积实现残差连接"""

    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1):
        super(SNNBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                               padding=(kernel_size - 1) // 2, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               padding=(kernel_size - 1) // 2, stride=stride)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 1x1卷积实现残差连接（特别是当输入输出通道数不同时）
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = F.relu(out)

        return out


class ChannelAttention(nn.Module):
    """通道注意力机制"""

    def __init__(self, in_features):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_features, in_features // 2)  # 降维
        self.fc2 = nn.Linear(in_features // 2, in_features)  # 升维

    def forward(self, x):
        # x: [B, C, L]
        b, c, _ = x.size()

        # [B, C, 1] -> [B, C]
        y = self.avg_pool(x).squeeze(-1)

        # [B, C] -> [B, C/2] -> [B, C]
        y = self.fc1(y)
        y = torch.tanh(y)
        y = self.fc2(y)

        # [B, C, 1]
        attention = torch.softmax(y, dim=1).unsqueeze(-1)

        # [B, C, L] * [B, C, 1] -> [B, C, L]
        out = x * attention

        return out


class SNNCD(nn.Module):
    """基于一维卷积神经网络的爬虫检测模型"""

    def __init__(self, time_length_channels=4, embed_channels=32,
                 seq_length=256, hidden_channels=8, num_blocks=8):
        super(SNNCD, self).__init__()

        # 时间长度分支
        self.time_length_branch = nn.Sequential(
            SNNBlock(time_length_channels, hidden_channels, kernel_size=7),
            *[SNNBlock(hidden_channels, hidden_channels, kernel_size=7)
              for _ in range(num_blocks)]
        )

        # 请求嵌入分支
        self.embed_branch = nn.Sequential(
            SNNBlock(embed_channels, hidden_channels, kernel_size=7),
            *[SNNBlock(hidden_channels, hidden_channels, kernel_size=7)
              for _ in range(num_blocks)]
        )

        # 自适应平均池化
        self.avg_pool = nn.AdaptiveAvgPool1d(64)

        # 通道注意力机制
        self.channel_attention = ChannelAttention(hidden_channels)

        # 全连接分类层
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 2)
        )

    def forward(self, time_length, embeddings):
        # 时间长度分支
        x1 = self.time_length_branch(time_length)  # [B, hidden, seq_len]
        x1 = self.channel_attention(x1)  # 应用通道注意力
        x1 = self.avg_pool(x1)  # [B, hidden, 64]
        x1 = x1.mean(dim=1)  # [B, 64]

        # 请求嵌入分支
        x2 = self.embed_branch(embeddings)  # [B, hidden, seq_len]
        x2 = self.channel_attention(x2)  # 应用通道注意力
        x2 = self.avg_pool(x2)  # [B, hidden, 64]
        x2 = x2.mean(dim=1)  # [B, 64]

        # 融合两个分支的特征（这里使用加权平均）
        x = (x1 + x2) / 2  # [B, 64]

        # 分类
        logits = self.classifier(x)

        return logits


class CrawlerDataset(Dataset):
    """爬虫检测数据集"""

    def __init__(self, time_length_data, embedding_data, labels):
        self.time_length_data = time_length_data  # [N, 4, 256]
        self.embedding_data = embedding_data  # [N, 32, 256]
        self.labels = labels  # [N]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'time_length': self.time_length_data[idx],
            'embedding': self.embedding_data[idx],
            'label': self.labels[idx]
        }


def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001, device='cuda'):
    """训练模型函数"""
    # 将模型移至GPU
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_val_acc = 0.0

    # 使用tqdm创建epoch进度条
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # 创建训练批次进度条
        train_pbar = tqdm(train_loader, desc="Training", ncols=100, leave=True)

        for batch in train_pbar:
            time_length = batch['time_length'].to(device)
            embedding = batch['embedding'].to(device)
            labels = batch['label'].to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(time_length, embedding)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # 更新进度条信息
            current_train_loss = loss.item()
            current_train_acc = 100.0 * (predicted == labels).sum().item() / labels.size(0)
            train_pbar.set_postfix({
                'loss': f'{current_train_loss:.4f}',
                'acc': f'{current_train_acc:.2f}%'
            })

        train_acc = 100.0 * train_correct / train_total
        train_loss = train_loss / len(train_loader)

        # 关闭训练进度条
        train_pbar.close()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # 创建验证批次进度条
        val_pbar = tqdm(val_loader, desc="Validation", ncols=100, leave=True)

        with torch.no_grad():
            for batch in val_pbar:
                time_length = batch['time_length'].to(device)
                embedding = batch['embedding'].to(device)
                labels = batch['label'].to(device)

                outputs = model(time_length, embedding)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # 更新进度条信息
                current_val_loss = loss.item()
                current_val_acc = 100.0 * (predicted == labels).sum().item() / labels.size(0)
                val_pbar.set_postfix({
                    'loss': f'{current_val_loss:.4f}',
                    'acc': f'{current_val_acc:.2f}%'
                })

        val_acc = 100.0 * val_correct / val_total
        val_loss = val_loss / len(val_loader)

        # 关闭验证进度条
        val_pbar.close()

        # 更新学习率
        scheduler.step(val_loss)

        # 打印完整的epoch结果
        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/best_snncd_model_balanced.pth')
            print(f'Model saved with accuracy: {val_acc:.2f}%')

    return model

# 设置中文字体
def set_chinese_font():
    # 针对不同操作系统添加可能的字体路径
    if os.name == 'nt':  # Windows
        potential_fonts = [
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            'C:/Windows/Fonts/simsun.ttc',  # 宋体
            'C:/Windows/Fonts/msyh.ttc'  # 微软雅黑
        ]
    else:
        print("警告: 未找到系统中的中文字体，将尝试使用其他方法")
        return None
    # 检查这些字体是否存在
    available_fonts = [f for f in potential_fonts if os.path.exists(f)]

    if available_fonts:
        # 使用第一个可用的字体
        font_path = available_fonts[0]
        print(f"使用中文字体: {font_path}")

        # 添加字体
        font_prop = fm.FontProperties(fname=font_path)

        # 设置默认字体
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

        return font_prop
    else:
        print("警告: 未找到系统中的中文字体，将尝试使用其他方法")
        return None


def evaluate_model_detailed(model, test_loader, device='cuda', save_path=None):
    """详细评估模型函数，使用sklearn生成综合报告，支持中文显示"""
    # 设置中文字体
    chinese_font = set_chinese_font()

    model = model.to(device)
    model.eval()

    all_preds = []
    all_probs = []  # 存储预测的概率值
    all_labels = []

    with torch.no_grad():
        # 添加测试进度条
        test_pbar = tqdm(test_loader, desc="Evaluating model", leave=True)
        for batch in test_pbar:
            time_length = batch['time_length'].to(device)
            embedding = batch['embedding'].to(device)
            labels = batch['label'].to(device)

            outputs = model(time_length, embedding)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            # 计算当前批次的准确率
            current_acc = 100.0 * (predicted == labels).sum().item() / labels.size(0)
            test_pbar.set_postfix({'acc': f'{current_acc:.2f}%'})

            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probabilities[:, 1].cpu().numpy())  # 取正类概率
            all_labels.extend(labels.cpu().numpy())

    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 基本指标
    accuracy = np.mean(all_preds == all_labels) * 100

    # 分类报告
    class_names = ['human', 'bot']
    report = classification_report(all_labels, all_preds, target_names=class_names)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # 计算ROC曲线和AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # 计算PR曲线和平均精度
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    avg_precision = average_precision_score(all_labels, all_probs)

    # 输出结果
    print(f"测试准确率: {accuracy:.2f}%")
    print("\n分类报告:")
    print(report)
    print("\n混淆矩阵:")
    print(conf_matrix)
    print(f"\nROC AUC: {roc_auc:.4f}")
    print(f"平均精度分数: {avg_precision:.4f}")

    # 如果指定了保存路径，则保存可视化结果
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)

        if chinese_font:
            plt.xlabel('预测标签', fontproperties=chinese_font)
            plt.ylabel('真实标签', fontproperties=chinese_font)
            plt.title('混淆矩阵', fontproperties=chinese_font)
            # 设置刻度标签的字体
            for label in plt.gca().get_xticklabels():
                label.set_fontproperties(chinese_font)
            for label in plt.gca().get_yticklabels():
                label.set_fontproperties(chinese_font)
        else:
            # 如果没有中文字体，使用英文替代
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
        plt.close()

        # 绘制ROC曲线
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        if chinese_font:
            plt.xlabel('假正例率', fontproperties=chinese_font)
            plt.ylabel('真正例率', fontproperties=chinese_font)
            plt.title('接收者操作特征曲线 (ROC)', fontproperties=chinese_font)
            # 设置图例字体
            legend = plt.legend(loc="lower right")
            for text in legend.get_texts():
                text.set_fontproperties(chinese_font)
        else:
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc="lower right")

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'roc_curve.png'))
        plt.close()

        # 绘制PR曲线
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                 label=f'PR curve (AP = {avg_precision:.2f})')
        plt.axhline(y=sum(all_labels) / len(all_labels), color='red', linestyle='--',
                    label=f'Baseline (positive rate = {sum(all_labels) / len(all_labels):.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        if chinese_font:
            plt.xlabel('召回率', fontproperties=chinese_font)
            plt.ylabel('精确率', fontproperties=chinese_font)
            plt.title('精确率-召回率曲线 (PR)', fontproperties=chinese_font)
            # 设置图例字体
            legend = plt.legend(loc="lower left")
            for text in legend.get_texts():
                text.set_fontproperties(chinese_font)
        else:
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'pr_curve.png'))
        plt.close()


        # 将分类报告保存为CSV
        report_dict = classification_report(all_labels, all_preds,
                                            target_names=class_names,
                                            output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(os.path.join(save_path, 'classification_report.csv'))

        # 保存预测结果
        pred_df = pd.DataFrame({
            '真实标签': all_labels,
            '预测标签': all_preds,
            '爬虫概率': all_probs
        })
        pred_df.to_csv(os.path.join(save_path, 'predictions.csv'), index=False)

        print(f"评估报告和可视化结果已保存至 {save_path}")

    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'probabilities': all_probs,
        'true_labels': all_labels,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }

def load_and_preprocess_data(data_path, embedder):
    """
    加载并预处理数据

    参数:
    - data_path: 包含JSON格式数据的文件路径
    - embedder: WebSessionEmbedder对象，用于获取请求嵌入向量

    返回:
    - 训练集、验证集和测试集的数据
    """
    # 加载JSON数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 初始化数据存储
    time_length_data = []
    embedding_data = []
    labels = []

    # 处理每个会话
    session_pbar = tqdm(data, desc="Processing sessions", leave=True)
    for session in session_pbar:
        if len(session['requests']) == 0:
            continue  # 跳过没有请求的会话
        is_bot : bool = session['is_bot']

        # 更新进度条信息
        session_type = "Bot" if is_bot else "Human"
        num_requests = len(session['requests'])
        session_pbar.set_postfix({
            'type': session_type,
            'requests': num_requests
        })

        # 提取请求时间和长度
        timestamps = []
        sizes = []
        requests = session['requests']

        # 处理每个请求
        for req in requests:
            timestamps.append(datetime.fromisoformat(req['timestamp']))
            sizes.append(req['size'])

        # 如果序列长度大于256，需要分块处理
        while len(timestamps) > 256:
            # 先取前256个元素作为第一个块
            current_timestamps = timestamps[:256]
            current_sizes = sizes[:256]
            current_requests = requests[:256]

            # 剩余部分
            timestamps = timestamps[256:]
            sizes = sizes[256:]
            requests = requests[256:]

            # 处理原始会话
            # 计算相对时间（相对于第一个请求的时间）
            start_time = current_timestamps[0]
            relative_times = [(t - start_time).total_seconds() for t in current_timestamps]

            # 计算时间差
            time_diffs = [0]  # 第一个时间差设为0
            for i in range(1, len(relative_times)):
                time_diffs.append(relative_times[i] - relative_times[i - 1])

            # 计算长度差
            size_diffs = [0]  # 第一个长度差设为0
            for i in range(1, len(current_sizes)):
                size_diffs.append(current_sizes[i] - current_sizes[i - 1])

            # 创建4通道时间长度序列
            time_length_channels = [
                relative_times,
                time_diffs,
                current_sizes,
                size_diffs
            ]

            # 转换为numpy数组
            time_length_array = np.array(time_length_channels, dtype=np.float32)  # [4, 256]

            # 获取原始会话的请求嵌入向量
            embeddings = []
            for req in current_requests:  # 处理原始会话的请求
                # 使用嵌入器获取嵌入向量
                embed = embedder.get_request_embedding(req)
                embeddings.append(embed.cpu().numpy())

            # 将嵌入向量堆叠，形成 [32, 256] 的矩阵
            embedding_array = np.array(embeddings, dtype=np.float32).T  # [32, 256]

            # 添加到数据列表
            time_length_data.append(time_length_array)
            embedding_data.append(embedding_array)
            labels.append(1 if is_bot else 0)

        # 现在处理剩余部分的嵌入向量
        # 数据集设定最少请求数为11，这里维持限制
        if len(requests) > 10:
            remaining_embeddings = []
            for req in requests:
                # 使用嵌入器获取嵌入向量
                embed = embedder.get_request_embedding(req)
                remaining_embeddings.append(embed.cpu().numpy())

            # 将剩余部分的嵌入向量合并成一个大的特征矩阵
            repeat_times = 256 // len(remaining_embeddings)
            remainder = 256 % len(remaining_embeddings)
            remaining_embeddings_padded = remaining_embeddings * repeat_times + remaining_embeddings[:remainder]

            # 将嵌入向量堆叠，形成 [32, 256] 的矩阵
            remaining_embedding_array = np.array(remaining_embeddings_padded, dtype=np.float32).T  # [32, 256]

            # 计算剩余部分的时间长度特征
            # 计算相对时间（相对于第一个请求的时间）
            remaining_start_time = timestamps[0]
            remaining_relative_times = [(t - remaining_start_time).total_seconds() for t in timestamps]

            # 计算时间差
            remaining_time_diffs = [0]  # 第一个时间差设为0
            for i in range(1, len(remaining_relative_times)):
                remaining_time_diffs.append(remaining_relative_times[i] - remaining_relative_times[i - 1])

            # 计算长度差
            remaining_size_diffs = [0]  # 第一个长度差设为0
            for i in range(1, len(sizes)):
                remaining_size_diffs.append(sizes[i] - sizes[i - 1])

            # 创建4通道时间长度序列
            remaining_time_length_channels = [
                remaining_relative_times,
                remaining_time_diffs,
                sizes,
                remaining_size_diffs
            ]

            # 确保所有通道长度为256
            for i in range(len(remaining_time_length_channels)):
                channel = remaining_time_length_channels[i]
                if len(channel) < 256:
                    # 计算需要重复的次数
                    repeat_times = 256 // len(channel)
                    remainder = 256 % len(channel)

                    # 循环填充
                    channel = channel * repeat_times + channel[:remainder]
                    remaining_time_length_channels[i] = channel

            # 转换为numpy数组
            remaining_time_length_array = np.array(remaining_time_length_channels, dtype=np.float32)  # [4, 256]

            # 添加剩余部分到数据列表
            time_length_data.append(remaining_time_length_array)
            embedding_data.append(remaining_embedding_array)
            labels.append(1 if is_bot else 0)
    # 如果剩余部分长度小于等于128，则丢弃

    # 转换为numpy数组
    time_length_data = np.array(time_length_data)
    embedding_data = np.array(embedding_data)
    labels = np.array(labels, dtype=np.int64)

    # 打乱数据集
    indices = np.random.permutation(len(labels))
    time_length_data = time_length_data[indices]
    embedding_data = embedding_data[indices]
    labels = labels[indices]

    # 分割训练集、验证集和测试集
    num_samples = len(labels)
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)

    train_time_length = torch.tensor(time_length_data[:train_size])
    train_embeddings = torch.tensor(embedding_data[:train_size])
    train_labels = torch.tensor(labels[:train_size])

    val_time_length = torch.tensor(time_length_data[train_size:train_size + val_size])
    val_embeddings = torch.tensor(embedding_data[train_size:train_size + val_size])
    val_labels = torch.tensor(labels[train_size:train_size + val_size])

    test_time_length = torch.tensor(time_length_data[train_size + val_size:])
    test_embeddings = torch.tensor(embedding_data[train_size + val_size:])
    test_labels = torch.tensor(labels[train_size + val_size:])

    return (
        (train_time_length, train_embeddings, train_labels),
        (val_time_length, val_embeddings, val_labels),
        (test_time_length, test_embeddings, test_labels)
    )


def main():
    # 检查CUDA是否可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 加载嵌入器
    print("加载嵌入器模型...")
    embedder = WebSessionEmbedder(
        use_fields=['method', 'decoded_path', 'status']
    )
    embedder.load_model("models_balanced")

    # 加载数据
    print("加载和预处理数据...")
    data_path = "../dataset/all_zip.json"
    (train_time_length, train_embeddings, train_labels), \
        (val_time_length, val_embeddings, val_labels), \
        (test_time_length, test_embeddings, test_labels) = load_and_preprocess_data(data_path, embedder)

    print(f"数据集大小 - 训练: {len(train_labels)}, 验证: {len(val_labels)}, 测试: {len(test_labels)}")

    # 创建数据集和数据加载器
    train_dataset = CrawlerDataset(train_time_length, train_embeddings, train_labels)
    val_dataset = CrawlerDataset(val_time_length, val_embeddings, val_labels)
    test_dataset = CrawlerDataset(test_time_length, test_embeddings, test_labels)

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    model = SNNCD(time_length_channels=4, embed_channels=32,
                  seq_length=256, hidden_channels=8, num_blocks=8)

    # 训练模型
    print("开始训练模型...")
    model = train_model(model, train_loader, val_loader,
                        num_epochs=40, lr=0.001, device=device)

    # 评估模型
    # 创建保存模型评估结果的目录
    results_dir = "evaluation_results_balanced_precise"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print("进行详细模型评估...")
    eval_results = evaluate_model_detailed(model, test_loader, device, save_path=results_dir)

    # 可以进一步分析结果
    print(f"最终测试准确率: {eval_results['accuracy']:.2f}%")
    print(f"ROC AUC: {eval_results['roc_auc']:.4f}")
    print(f"平均精度: {eval_results['avg_precision']:.4f}")



if __name__ == "__main__":
    main()
