import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
import torch
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import ijson
from CrawlerDetect import is_crawler
import os
import hashlib


class CrawlerDetectorSVM:
    def __init__(self, use_gpu=True):
        """
        初始化爬虫检测器

        参数:
            use_gpu: 是否使用GPU (如果有可用的CUDA)
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        print(f"使用设备: {self.device}")

        # 初始化SVM分类器
        self.svm_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True, kernel='rbf', C=1.0, gamma='scale'))
        ])

        # 特征名称，用于解释模型
        self.feature_names = [
            "requests_per_second", "avg_interval_seconds", "interval_std_dev",
            "head_request_ratio", "night_access_ratio", "has_systematic_pattern",
            "path_repetition_ratio", "missing_referer_ratio_adjusted",
            "max_bigram_ratio", "ratio_404", "avg_size", "std_dev_size"
        ]

        # 记录训练历史
        self.history = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }

    def extract_features(self, sources: List[str], threshold: float) -> Tuple:
        """
        从会话数据中提取特征

        参数:
            sessions: 会话数据列表

        返回:
            特征矩阵
        """
        cache_key = f"{','.join(sorted(sources))}_{threshold}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_dir = "cache"
        cache_file = os.path.join(cache_dir, f"features_{cache_hash}.pkl")

        # 如果缓存目录不存在则创建
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # 检查是否存在缓存
        if os.path.exists(cache_file):
            print(f"正在从缓存加载特征...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        features = []
        labels = []

        print("提取特征中...")
        for source in sources:
            with open(source, 'r', encoding='utf-8') as f:
                sessions = ijson.items(f, 'item')
                for session in tqdm(sessions, desc="特征提取"):
                    if session.get("request_count", 0) <= 10:
                        continue
                    # 使用提供的函数进行特征分析
                    label, analysis_results = is_crawler(session, threshold=threshold)
                    labels.append(label)
                    # 提取需要的特征
                    feature_vector = []
                    for feature in self.feature_names:
                        feature_vector.append(analysis_results.get(feature, 0))
                    # 会话基本统计信息
                    feature_vector.append(session.get("request_count", 0))
                    feature_vector.append(session.get("duration_seconds", 0))
                    for i in range(len(feature_vector)):
                        if feature_vector[i] is None or not np.isfinite(float(feature_vector[i])):
                            feature_vector[i] = 0.0
                            # 处理极端值
                        if abs(float(feature_vector[i])) > 1e15:
                            feature_vector[i] = np.sign(float(feature_vector[i])) * 1e15
                    features.append(feature_vector)

        # 将特征列表转换为numpy数组
        # 将结果转换为numpy数组
        result = (np.array(features, dtype=np.float32), np.array(labels, dtype=np.int8))

        # 保存到缓存
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)

        return result

    def train(self, X, y, test_size=0.3, random_state=42):
        """
        训练SVM分类器

        参数:
            X: 特征矩阵
            y: 标签向量
            test_size: 测试集比例
            random_state: 随机种子

        返回:
            训练和测试的准确率
        """
        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"训练数据: {X_train.shape[0]} 样本, 测试数据: {X_test.shape[0]} 样本")

        # 如果使用GPU，将数据转移到GPU
        if self.use_gpu:
            X_train_gpu = torch.tensor(X_train, device=self.device)
            y_train_gpu = torch.tensor(y_train, device=self.device)
            # 由于scikit-learn不直接支持GPU，这里仅作为数据预处理步骤
            # 将GPU中的数据转回CPU供scikit-learn使用
            X_train = X_train_gpu.cpu().numpy()
            y_train = y_train_gpu.cpu().numpy()

        # 训练模型
        print("训练模型中...")
        self.svm_pipeline.fit(X_train, y_train)

        # 预测和评估
        y_pred = self.svm_pipeline.predict(X_test)
        y_prob = self.svm_pipeline.predict_proba(X_test)[:, 1]

        # 计算各种评估指标
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_prob)

        # 记录历史
        self.history['accuracy'].append(accuracy)
        self.history['precision'].append(report['1']['precision'])  # 爬虫为正类
        self.history['recall'].append(report['1']['recall'])
        self.history['f1'].append(report['1']['f1-score'])
        self.history['auc'].append(auc)

        print(f"训练准确率: {self.svm_pipeline.score(X_train, y_train):.4f}")
        print(f"测试准确率: {accuracy:.4f}")
        print(f"AUC值: {auc:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("\n混淆矩阵:")
        print(conf_matrix)

        # 返回测试准确率
        return accuracy, auc

    def cross_validate(self, X, y, cv=5):
        """
        使用交叉验证评估模型

        参数:
            X: 特征矩阵
            y: 标签向量
            cv: 交叉验证折数

        返回:
            交叉验证分数
        """
        print(f"执行{cv}折交叉验证...")
        cv_scores = cross_val_score(self.svm_pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        print(f"交叉验证准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        return cv_scores

    def grid_search(self, X, y, param_grid=None, cv=5):
        """
        执行网格搜索以优化超参数

        参数:
            X: 特征矩阵
            y: 标签向量
            param_grid: 参数网格
            cv: 交叉验证折数

        返回:
            最优参数
        """
        if param_grid is None:
            param_grid = {
                'svm__C': [0.1, 1, 10, 100],
                'svm__gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
                'svm__kernel': ['rbf', 'poly', 'sigmoid']
            }

        print("执行网格搜索以优化超参数...")
        grid = GridSearchCV(
            self.svm_pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=2
        )

        grid_result = grid.fit(X, y)

        print(f"最优参数: {grid.best_params_}")
        print(f"最佳交叉验证得分: {grid.best_score_:.4f}")

        # 更新模型为最佳参数
        self.svm_pipeline = grid.best_estimator_

        return grid.best_params_

    def predict(self, sessions):
        """
        预测会话是否为爬虫

        参数:
            sessions: 单个会话或会话列表

        返回:
            预测结果和概率
        """
        # 确保输入是列表
        if not isinstance(sessions, list):
            sessions = [sessions]

        # 提取特征
        X = self.extract_features(sessions)

        # 预测
        y_pred = self.svm_pipeline.predict(X)
        y_prob = self.svm_pipeline.predict_proba(X)[:, 1]

        # 返回预测结果和概率
        return y_pred, y_prob

    def get_feature_importance(self):
        """
        获取特征重要性分析（基于SVM权重）
        注意: 这种方法对于非线性核（如RBF）不是最理想的，但提供了初步的理解

        返回:
            特征重要性字典
        """
        # 对于非线性SVM，可以近似计算特征重要性
        # 这里使用支持向量的平均贡献作为替代
        if hasattr(self.svm_pipeline['svm'], 'coef_'):
            # 线性核的情况
            importance = np.abs(self.svm_pipeline['svm'].coef_[0])
            importance_dict = dict(zip(self.feature_names, importance))
        else:
            # 非线性核的情况 - 使用基于排列的特征重要性
            # 这里仅提供一个占位符，实际实现需要额外的计算
            importance_dict = {name: 0 for name in self.feature_names}
            print("注意: 对于非线性核SVM，特征重要性计算是近似的")

        return importance_dict

    def save_model(self, file_path):
        """
        保存模型到文件

        参数:
            file_path: 保存路径
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self.svm_pipeline, f)
        print(f"模型已保存到 {file_path}")

    def load_model(self, file_path):
        """
        从文件加载模型

        参数:
            file_path: 模型文件路径
        """
        with open(file_path, 'rb') as f:
            self.svm_pipeline = pickle.load(f)
        print(f"模型已从 {file_path} 加载")

    def plot_results(self):
        """
        绘制训练结果图表
        """
        if not self.history['accuracy']:
            print("没有训练历史可供绘制")
            return

        plt.figure(figsize=(15, 10))

        # 绘制准确率
        plt.subplot(2, 2, 1)
        plt.plot(self.history['accuracy'], marker='o')
        plt.title('准确率')
        plt.xlabel('训练轮次')
        plt.ylabel('准确率')
        plt.grid(True)

        # 绘制精确度、召回率和F1值
        plt.subplot(2, 2, 2)
        plt.plot(self.history['precision'], marker='o', label='精确度')
        plt.plot(self.history['recall'], marker='s', label='召回率')
        plt.plot(self.history['f1'], marker='^', label='F1值')
        plt.title('精确度, 召回率和F1值')
        plt.xlabel('训练轮次')
        plt.legend()
        plt.grid(True)

        # 绘制AUC
        plt.subplot(2, 2, 3)
        plt.plot(self.history['auc'], marker='o', color='r')
        plt.title('AUC')
        plt.xlabel('训练轮次')
        plt.ylabel('AUC')
        plt.grid(True)

        # 特征重要性
        plt.subplot(2, 2, 4)
        importance = self.get_feature_importance()
        features = list(importance.keys())
        values = list(importance.values())
        sorted_idx = np.argsort(values)
        plt.barh([features[i] for i in sorted_idx], [values[i] for i in sorted_idx])
        plt.title('特征重要性')
        plt.tight_layout()
        plt.show()


# 使用示例
def main():
    # 创建并训练模型
    detector = CrawlerDetectorSVM(use_gpu=True)

    # 提取特征
    X, labels = detector.extract_features(['../dataset/train.json', '../dataset/test.json'], 1.8)

    # 训练模型
    detector.train(X, labels)

    # 交叉验证
    # detector.cross_validate(X, labels, cv=5)

    # 网格搜索优化
    # detector.grid_search(X, labels)

    # 保存模型
    detector.save_model('models/crawler_detector_svm.pkl')

    # 额外的完整数据集评估 - 保存所有样本的预测概率
    # print("\n对完整数据集进行评估和概率分析...")
    # # 使用交叉验证获取完整数据集的预测
    # from sklearn.model_selection import cross_val_predict
    # y_pred = cross_val_predict(detector.svm_pipeline, X, labels, cv=5)
    # y_prob = cross_val_predict(detector.svm_pipeline, X, labels, cv=5, method='predict_proba')
    #
    # # 创建结果DataFrame
    # results_df = pd.DataFrame({
    #     'true_label': labels,
    #     'predicted_label': y_pred,
    #     'prob_class_0': y_prob[:, 0],
    #     'prob_class_1': y_prob[:, 1],
    #     'correct_prediction': labels == y_pred
    # })
    #
    # # 保存完整结果
    # os.makedirs('./prediction_results', exist_ok=True)
    # results_df.to_csv('./prediction_results/full_dataset_predictions.csv', index=False)
    # print("已保存完整数据集的预测结果到: ./prediction_results/full_dataset_predictions.csv")
    #
    # print("模型训练和评估完成！")


if __name__ == "__main__":
    main()
