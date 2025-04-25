import ijson
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, Counter
import random
import os
from tqdm import tqdm
import pickle
from typing import List, Dict, Tuple
import datetime
from urllib.parse import urlparse, parse_qs


class WebSessionEmbedder:
    """
    Web会话嵌入框架主类，用于处理会话数据、训练Skip-gram模型、生成嵌入
    """

    def __init__(self,
                 window_size: int = 5,
                 embedding_dim: int = 32,
                 min_count: int = 5,
                 use_fields: List[str] = None,
                 negative_samples: int = 5,
                 learning_rate: float = 0.001,
                 batch_size: int = 64,
                 epochs: int = 10,
                 device: str = None):
        """
        初始化Web会话嵌入器

        Args:
            window_size: 上下文窗口大小
            embedding_dim: 嵌入向量维度
            min_count: 最小词频阈值，低于此值的请求特征将被忽略
            use_fields: 用于特征提取的请求字段列表
            negative_samples: 负采样数量
            learning_rate: 学习率
            batch_size: 批处理大小
            epochs: 训练轮数
        """
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.min_count = min_count
        self.negative_samples = negative_samples
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        # 默认使用的字段
        self.use_fields = use_fields or ['method', 'decoded_path', 'status', 'user_agent']

        # 词汇表和嵌入相关
        self.vocab = None
        self.token2idx = None
        self.idx2token = None
        self.token_freqs = None
        self.model = None
        self.feature_extractor = FeatureExtractor()


        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")

    def load_data(self, file_paths: List[str]) -> List[Dict]:
        """
        加载会话数据

        Args:
            file_paths: 数据集文件路径

        Returns:
            处理后的会话数据列表
        """
        sessions = []
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    objects = ijson.items(f, 'item')
                    for session in objects:
                        sessions.append(session)
                except IOError:
                    continue

        print(f"加载了 {len(sessions)} 个有效会话")
        return sessions

    def build_vocabulary(self, sessions: List[Dict]) -> None:
        """
        构建词汇表

        Args:
            sessions: 会话数据列表
        """
        token_counter = Counter()

        # 遍历所有会话的请求
        for session in tqdm(sessions, desc="构建词汇表"):
            requests = session.get('requests', [])
            for request in requests:
                # 提取特征并添加到计数器
                features = self.feature_extractor.extract_features(request, self.use_fields)
                for feature in features:
                    token_counter[feature] += 1

        # 过滤低频词
        filtered_tokens = [token for token, count in token_counter.items()
                           if count >= self.min_count]

        # 构建词汇映射
        self.token2idx = {token: idx for idx, token in enumerate(filtered_tokens)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        self.vocab = filtered_tokens
        self.token_freqs = {token: count for token, count in token_counter.items()
                            if token in self.token2idx}

        print(f"词汇表大小: {len(self.vocab)}")

    def prepare_skipgram_data(self, sessions: List[Dict]) -> List[Tuple[int, int]]:
        """
        准备Skip-gram训练数据

        Args:
            sessions: 会话数据列表

        Returns:
            训练样本列表，每个样本为(目标词索引, 上下文词索引)的元组
        """
        skipgram_data = []

        for session in tqdm(sessions, desc="准备训练数据"):
            requests = session.get('requests', [])

            # 从每个请求中提取特征
            session_features = []
            for request in requests:
                features = self.feature_extractor.extract_features(request, self.use_fields)
                session_features.extend([f for f in features if f in self.token2idx])

            # 如果特征太少，跳过该会话
            if len(session_features) < self.window_size + 1:
                continue

            # 生成目标词和上下文词对
            for i in range(self.window_size, len(session_features) - self.window_size, 2):
                target_token = session_features[i]
                if target_token not in self.token2idx:
                    continue

                target_idx = self.token2idx[target_token]

                # 在窗口大小内随机选择上下文
                context_indices = list(range(max(0, i - self.window_size), i)) + \
                                  list(range(i + 1, min(i + 1 + self.window_size, len(session_features))))

                for j in context_indices:
                    context_token = session_features[j]
                    if context_token not in self.token2idx:
                        continue

                    context_idx = self.token2idx[context_token]
                    skipgram_data.append((target_idx, context_idx))

        print(f"生成了 {len(skipgram_data)} 个训练样本")
        return skipgram_data

    def initialize_model(self) -> None:
        """
        初始化Skip-gram模型
        """
        vocab_size = len(self.vocab)
        self.model = SkipGramModel(vocab_size, self.embedding_dim)
        self.model.to(self.device)  # 将模型迁移到GPU

    def train(self, training_data: List[Tuple[int, int]], save_path: str = "web_session_model") -> None:
        """
        训练Skip-gram模型

        Args:
            training_data: 训练数据
            save_path: 模型保存路径
        """
        # 准备数据集
        dataset = SkipGramDataset(training_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 初始化模型和优化器
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # 训练循环
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}")

            for batch_idx, (target_idx, context_idx, neg_samples) in enumerate(progress_bar):
                # 将数据迁移到GPU
                target_idx = target_idx.to(self.device)
                context_idx = context_idx.to(self.device)
                neg_samples = neg_samples.to(self.device)
                # 前向传播
                loss = self.model(target_idx, context_idx, neg_samples)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix(loss=f"{total_loss / (batch_idx + 1):.4f}")

        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)

        # 保存模型和配置
        self.save_model(save_path)

        print(f"模型训练完成并保存到 {save_path}")

    def save_model(self, save_path: str) -> None:
        """
        保存模型和相关配置

        Args:
            save_path: 保存路径
        """
        # 保存模型权重
        torch.save(self.model.state_dict(), f"{save_path}/model.pth")

        # 保存词汇表和映射
        with open(f"{save_path}/vocab.pkl", 'wb') as f:
            pickle.dump({
                'token2idx': self.token2idx,
                'idx2token': self.idx2token,
                'vocab': self.vocab,
                'token_freqs': self.token_freqs,
                'use_fields': self.use_fields,
                'embedding_dim': self.embedding_dim
            }, f)

    def load_model(self, load_path: str) -> None:
        """
        加载已保存的模型和配置

        Args:
            load_path: 模型加载路径
        """
        # 加载词汇表和映射
        with open(f"{load_path}/vocab.pkl", 'rb') as f:
            vocab_data = pickle.load(f)
            self.token2idx = vocab_data['token2idx']
            self.idx2token = vocab_data['idx2token']
            self.vocab = vocab_data['vocab']
            self.token_freqs = vocab_data['token_freqs']
            self.use_fields = vocab_data['use_fields']
            self.embedding_dim = vocab_data['embedding_dim']

        # 初始化模型
        vocab_size = len(self.vocab)
        self.model = SkipGramModel(vocab_size, self.embedding_dim)

        # 加载模型权重
        self.model.load_state_dict(torch.load(f"{load_path}/model.pth", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        print(f"模型从 {load_path} 加载完成")

    def get_session_embedding(self, session: Dict) -> torch.Tensor:
        """
        获取会话的嵌入表示

        Args:
            session: 单个会话数据

        Returns:
            会话嵌入向量
        """
        requests = session.get('requests', [])

        # 提取所有请求的特征
        session_features = []
        for request in requests:
            features = self.feature_extractor.extract_features(request, self.use_fields)
            valid_features = [f for f in features if f in self.token2idx]
            session_features.extend(valid_features)

        # 如果没有有效特征，返回零向量
        if not session_features:
            return torch.zeros(self.embedding_dim)

        # 获取特征对应的索引
        feature_indices = [self.token2idx[feature] for feature in session_features
                           if feature in self.token2idx]

        # 如果没有有效索引，返回零向量
        if not feature_indices:
            return torch.zeros(self.embedding_dim)

        # 获取嵌入并平均
        with torch.no_grad():
            indices_tensor = torch.tensor(feature_indices).to(self.device)
            embeddings = self.model.get_target_embedding(indices_tensor)
            session_embedding = torch.mean(embeddings, dim=0)

        return session_embedding.cpu()

    def get_request_embedding(self, request: Dict, device='cuda') -> torch.Tensor:
        """
        获取单个请求的嵌入表示，使用GPU加速

        Args:
            request: 单个请求数据
            device: 计算设备，默认为'cuda'

        Returns:
            请求嵌入向量
        """
        # 确保模型在正确的设备上
        if next(self.model.parameters()).device.type != device:
            self.model = self.model.to(device)

        features = self.feature_extractor.extract_features(request, self.use_fields)
        valid_features = [f for f in features if f in self.token2idx]

        # 如果没有有效特征，返回零向量
        if not valid_features:
            return torch.zeros(self.embedding_dim, device=device)

        # 获取特征对应的索引
        feature_indices = [self.token2idx[feature] for feature in valid_features]

        # 将特征索引也移至GPU
        feature_indices_tensor = torch.tensor(feature_indices, device=device)

        # 获取嵌入并平均
        with torch.no_grad():
            embeddings = self.model.get_target_embedding(feature_indices_tensor)
            request_embedding = torch.mean(embeddings, dim=0)

        return request_embedding

    def find_similar_sessions(self, query_session: Dict, all_sessions: List[Dict], top_k: int = 5) -> List[
        Tuple[Dict, float]]:
        """
        查找与给定会话最相似的其他会话

        Args:
            query_session: 查询会话
            all_sessions: 所有候选会话列表
            top_k: 返回最相似的会话数量

        Returns:
            相似会话列表，每项为(会话, 相似度)的元组
        """
        query_embedding = self.get_session_embedding(query_session)

        similarities = []
        for session in all_sessions:
            if session == query_session:
                continue

            session_embedding = self.get_session_embedding(session)
            # 计算余弦相似度
            similarity = torch.cosine_similarity(query_embedding.unsqueeze(0),
                                                 session_embedding.unsqueeze(0)).item()
            similarities.append((session, similarity))

        # 按相似度降序排序并返回top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class FeatureExtractor:
    """
    请求特征提取器
    """

    def __init__(self):
        """
        初始化特征提取器
        """
        pass

    def extract_features(self, request: Dict, use_fields: List[str]) -> List[str]:
        """
        从单个请求中提取特征

        Args:
            request: 请求数据
            use_fields: 要使用的字段列表

        Returns:
            提取的特征列表
        """
        features = []

        for field in use_fields:
            if field not in request:
                continue

            value = request[field]

            # 跳过空值
            if value is None:
                continue

            # 根据不同字段类型进行特征提取
            if field == 'method':
                features.append(f"METHOD:{value}")

            elif field == 'status':
                features.append(f"STATUS:{value}")
                status_group = value // 100
                features.append(f"STATUS_GROUP:{status_group}xx")

            elif field == 'decoded_path':
                # 提取路径组件
                path_features = self._extract_path_features(value)
                features.extend(path_features)

            elif field == 'user_agent':
                # 提取用户代理特征
                ua_features = self._extract_ua_features(value)
                features.extend(ua_features)

            elif field == 'timestamp':
                # 提取时间特征
                time_features = self._extract_time_features(value)
                features.extend(time_features)

            else:
                # 其他字段直接添加
                features.append(f"{field.upper()}:{value}")

        return features

    def _extract_path_features(self, path: str) -> List[str]:
        """
        从URL路径中提取特征

        Args:
            path: URL路径

        Returns:
            路径特征列表
        """
        features = []

        # 提取路径的第一级目录
        parts = path.strip('/').split('/')
        if parts and parts[0]:
            features.append(f"PATH_DIR1:{parts[0]}")

        # 如果路径有多个部分，提取第二级目录
        if len(parts) > 1 and parts[1]:
            features.append(f"PATH_DIR2:{parts[1]}")

        # 提取文件扩展名
        if '.' in parts[-1]:
            ext = parts[-1].split('.')[-1]
            if ext:
                features.append(f"PATH_EXT:{ext}")

        # 解析查询参数
        parsed_url = urlparse(path)
        if parsed_url.query:
            query_params = parse_qs(parsed_url.query)
            for param in query_params:
                features.append(f"QUERY_PARAM:{param}")

        return features

    def _extract_ua_features(self, user_agent: str) -> List[str]:
        """
        从用户代理字符串中提取特征

        Args:
            user_agent: 用户代理字符串

        Returns:
            用户代理特征列表
        """
        features = []

        # 检测是否为搜索引擎爬虫
        crawler_patterns = [
            'bot', 'spider', 'crawl', 'slurp', 'search',
            'fetch', 'apache-httpclient', 'python-requests'
        ]

        ua_lower = user_agent.lower()
        for pattern in crawler_patterns:
            if pattern in ua_lower:
                features.append(f"UA_CRAWLER:{pattern}")
                break

        # 提取浏览器信息
        browsers = ['chrome', 'firefox', 'safari', 'edge', 'msie', 'opera']
        for browser in browsers:
            if browser in ua_lower:
                features.append(f"UA_BROWSER:{browser}")
                break

        # 提取操作系统信息
        os_patterns = [
            ('windows', 'UA_OS:windows'),
            ('mac', 'UA_OS:mac'),
            ('linux', 'UA_OS:linux'),
            ('android', 'UA_OS:android'),
            ('ios', 'UA_OS:ios')
        ]

        for pattern, feature in os_patterns:
            if pattern in ua_lower:
                features.append(feature)
                break

        # 提取设备类型
        device_patterns = [
            ('mobile', 'UA_DEVICE:mobile'),
            ('tablet', 'UA_DEVICE:tablet')
        ]

        for pattern, feature in device_patterns:
            if pattern in ua_lower:
                features.append(feature)
                break

        # 如果没有匹配到移动或平板，假设为桌面
        if not any(pattern in ua_lower for pattern, _ in device_patterns):
            features.append('UA_DEVICE:desktop')

        return features

    def _extract_time_features(self, timestamp: str) -> List[str]:
        """
        从时间戳中提取特征

        Args:
            timestamp: 时间戳字符串

        Returns:
            时间特征列表
        """
        features = []

        try:
            # 解析时间戳
            dt = datetime.datetime.fromisoformat(timestamp)

            # 提取小时 (0-23)
            hour = dt.hour
            features.append(f"TIME_HOUR:{hour}")

            # 时段分类
            if 6 <= hour < 12:
                features.append("TIME_PERIOD:morning")
            elif 12 <= hour < 18:
                features.append("TIME_PERIOD:afternoon")
            elif 18 <= hour < 22:
                features.append("TIME_PERIOD:evening")
            else:
                features.append("TIME_PERIOD:night")

            # 提取星期几 (0-6, 0是星期一)
            weekday = dt.weekday()
            features.append(f"TIME_WEEKDAY:{weekday}")

            # 工作日/周末
            if weekday < 5:  # 0-4为周一至周五
                features.append("TIME_WORKDAY:yes")
            else:
                features.append("TIME_WEEKEND:yes")

        except (ValueError, TypeError):
            pass

        return features


class SkipGramModel(nn.Module):
    """
    Skip-gram模型实现
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        初始化Skip-gram模型

        Args:
            vocab_size: 词汇表大小
            embedding_dim: 嵌入向量维度
        """
        super(SkipGramModel, self).__init__()

        # 目标词嵌入层
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # 上下文词嵌入层
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """
        初始化模型权重
        """
        # 使用均匀分布初始化嵌入权重
        nn.init.uniform_(self.target_embeddings.weight, -0.5 / self.target_embeddings.embedding_dim,
                         0.5 / self.target_embeddings.embedding_dim)
        nn.init.uniform_(self.context_embeddings.weight, -0.5 / self.context_embeddings.embedding_dim,
                         0.5 / self.context_embeddings.embedding_dim)

    def forward(self, target_idx: torch.Tensor, context_idx: torch.Tensor, negative_idx: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            target_idx: 目标词索引
            context_idx: 上下文词索引
            negative_idx: 负采样词索引

        Returns:
            损失值
        """
        # 获取目标词嵌入
        target_emb = self.target_embeddings(target_idx)  # [batch_size, embedding_dim]

        # 获取正样本上下文词嵌入
        context_emb = self.context_embeddings(context_idx)  # [batch_size, embedding_dim]

        # 计算正样本得分 (点积)
        positive_score = torch.sum(target_emb * context_emb, dim=1)  # [batch_size]

        # 获取负样本上下文词嵌入
        negative_emb = self.context_embeddings(negative_idx)  # [batch_size, neg_samples, embedding_dim]

        # 计算负样本得分 (点积)
        negative_score = torch.bmm(negative_emb, target_emb.unsqueeze(2)).squeeze()  # [batch_size, neg_samples]

        # 使用 BCEWithLogitsLoss (包含sigmoid)
        positive_loss = torch.mean(torch.nn.functional.logsigmoid(positive_score))
        negative_loss = torch.mean(torch.nn.functional.logsigmoid(-negative_score))

        # 总损失
        loss = -(positive_loss + negative_loss)

        return loss

    def get_target_embedding(self, idx: torch.Tensor) -> torch.Tensor:
        """
        获取目标词嵌入

        Args:
            idx: 词索引

        Returns:
            嵌入向量
        """
        return self.target_embeddings(idx)

    def get_context_embedding(self, idx: torch.Tensor) -> torch.Tensor:
        """
        获取上下文词嵌入

        Args:
            idx: 词索引

        Returns:
            嵌入向量
        """
        return self.context_embeddings(idx)


class SkipGramDataset(Dataset):
    """
    Skip-gram模型数据集
    """

    def __init__(self, data: List[Tuple[int, int]], neg_samples: int = 5):
        """
        初始化数据集

        Args:
            data: 训练数据，每项为(目标词索引, 上下文词索引)的元组
            neg_samples: 负采样数量
        """
        self.data = data
        self.neg_samples = neg_samples

        # 计算词频分布，用于负采样
        self._build_sampling_table()

    def _build_sampling_table(self):
        """
        构建负采样表
        """
        # 获取所有唯一的索引
        indices = set()
        for target_idx, context_idx in self.data:
            indices.add(target_idx)
            indices.add(context_idx)

        self.vocab_size = len(indices)

        # 使用均匀分布作为采样分布
        self.sampling_table = list(indices)

    def __len__(self) -> int:
        """
        返回数据集大小
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            目标词索引，上下文词索引，负采样词索引的元组
        """
        target_idx, context_idx = self.data[idx]

        # 负采样
        negative_samples = []
        while len(negative_samples) < self.neg_samples:
            neg_idx = random.choice(self.sampling_table)
            # 确保负样本不是正样本
            if neg_idx != context_idx and neg_idx != target_idx:
                negative_samples.append(neg_idx)

        return (
            torch.tensor(target_idx, dtype=torch.long),
            torch.tensor(context_idx, dtype=torch.long),
            torch.tensor(negative_samples, dtype=torch.long)
        )


def main():
    """
    主函数示例
    """
    # 初始化嵌入器
    embedder = WebSessionEmbedder(
        window_size=2,  # 2 * 2 + 1 = 5
        embedding_dim=32,
        min_count=3,
        use_fields=['method', 'decoded_path', 'status'],
        negative_samples=5,
        learning_rate=0.001,
        batch_size=64,
        epochs=3
    )

    # 加载数据
    sessions = embedder.load_data(["../dataset/all_zip.json"])

    # 构建词汇表
    embedder.build_vocabulary(sessions)

    # 准备训练数据
    training_data = embedder.prepare_skipgram_data(sessions)

    # 初始化模型
    embedder.initialize_model()

    # 训练模型
    embedder.train(training_data, save_path="models_balanced")

    # 示例：获取会话嵌入
    if sessions:
        session_embedding = embedder.get_session_embedding(sessions[0])
        print(f"会话嵌入维度: {session_embedding.shape}")

        # 查找相似会话
        # similar_sessions = embedder.find_similar_sessions(sessions[0], sessions, top_k=5)
        # print(f"找到 {len(similar_sessions)} 个相似会话")


if __name__ == "__main__":
    main()
