from collections import deque

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, precision_score, f1_score
from sklearn.datasets import fetch_openml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, MultivariateNormal
import os
from scipy import stats
import requests
from io import BytesIO
import zipfile
import warnings
from typing import List, Dict, Tuple, Optional, Union
import seaborn as sns
from sklearn.impute import SimpleImputer
from scipy.interpolate import UnivariateSpline
from scipy.stats import entropy
import random

# 设置随机种子确保结果可复现
np.random.seed(42)
torch.manual_seed(42)

# 忽略警告
warnings.filterwarnings('ignore')


class DataProcessor:
    """数据处理类，负责数据加载、基础数据选取和归一化"""

    # "D:\projects\pythonProject\cm1.csv"
    # r"D:\projects\pythonProject\PROMISE-backup-master\PROMISE-backup-master\bug-source_data\jedit\jedit-3.2.csv"
    #  D:\projects\pythonProject\uci_heart_disease_clean.csv
    def __init__(self,
                 data_path: str = r"D:\projects\pythonProject\processed_data\statlog_german_credit_processed_full.csv",
                 dataset_name: str = None):
        """
        初始化数据处理器

        参数:
            data_path: 数据路径，如果为None则从Promise数据集下载
            dataset_name: 要使用的Promise数据集名称，如"cm1", "jm1"等
        """
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.data = None
        self.X = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.feature_names = None  # 新增：存储特征名称

    def load_data(self) -> pd.DataFrame:
        """加载Promise数据集"""
        if self.data_path:
            # 从本地加载数据
            self.data = pd.read_csv(self.data_path)
            if 'target' in self.data.columns:
                self.data['target'] = self.data['target'].apply(lambda x: 1 if x > 1 else x)
            elif 'name' in self.data.columns:
                self.data = self.data.drop(['name'], axis=1)
                self.data['bug'] = self.data['bug'].apply(lambda x: 1 if x > 1 else x)
        else:
            # 从Promise数据集下载
            Data = fetch_openml(name=self.dataset_name, version=1, as_frame=True)
            X = pd.DataFrame(Data.data, columns=Data.feature_names)
            Y = Data.target

            # 自动检测并转换目标变量格式
            Y = self._convert_target_variable(Y)

            # 合并特征和目标变量
            self.data = pd.concat([X, Y], axis=1)

        print(f"数据加载完成，形状: {self.data.shape}")
        return self.data

    def _convert_target_variable(self, Y: pd.Series) -> pd.Series:
        """
        自动检测并转换目标变量格式为0/1

        参数:
            Y: 目标变量Series

        返回:
            转换为0/1格式的目标变量Series
        """
        # 获取唯一值并转换为小写
        unique_values = Y.unique()
        unique_values_lower = [str(val).lower() for val in unique_values]

        # 检查常见的二元分类标签格式
        if set(unique_values_lower) == {'yes', 'no'}:
            print("检测到yes/no格式，转换为0/1")
            return Y.map({'yes': 1, 'no': 0, 'YES': 1, 'NO': 0, 'Yes': 1, 'No': 0})
        elif set(unique_values_lower) == {'true', 'false'}:
            print("检测到true/false格式，转换为0/1")
            return Y.map({'true': 1, 'false': 0, 'TRUE': 1, 'FALSE': 0, 'True': 1, 'False': 0})
        elif set(unique_values_lower) == {'1', '0'}:
            print("检测到1/0格式，直接转换为数值类型")
            return Y.astype(int)
        elif set(unique_values_lower) == {'1', '0', 'yes', 'no'}:  # 处理混合格式
            print("检测到混合格式，转换为0/1")
            return Y.map({'yes': 1, 'no': 0, 'YES': 1, 'NO': 0, 'Yes': 1, 'No': 0, '1': 1, '0': 0})
        elif set(unique_values_lower) == {'1', '0', 'true', 'false'}:  # 处理混合格式
            print("检测到混合格式，转换为0/1")
            return Y.map({'true': 1, 'false': 0, 'TRUE': 1, 'FALSE': 0, 'True': 1, 'False': 0, '1': 1, '0': 0})
        else:
            print(f"警告: 无法识别目标变量格式，唯一值: {unique_values}")
            print("将尝试直接转换为数值类型，可能会导致错误")
            try:
                return Y.astype(int)
            except:
                print("错误: 无法将目标变量转换为数值类型")
                print("请检查数据集或手动指定目标变量转换方式")
                return Y


    def prepare_data(self, test_size: float = 2 / 10, random_state: int = 42) -> None:
        """
        准备训练集和测试集

        参数:
            test_size: 测试集比例
            random_state: 随机种子
        """
        if self.data is None:
            raise ValueError("请先加载数据")

        # 假设最后一列是目标变量
        target_column = self.data.columns[-1]
        feature_columns = [col for col in self.data.columns if col != target_column]

        self.feature_names = feature_columns

        X = self.data[feature_columns].values
        X = self.scaler.fit_transform(X)
        self.X = X
        y = self.data[target_column].values
        # 使用SMOTE处理不平衡
        # smote = SMOTE(random_state=42)
        # X, y = smote.fit_resample(X, y)

        # 分割数据集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"训练集大小: {self.X_train.shape}, 测试集大小: {self.X_test.shape}")

        train_defect_count = np.sum(self.y_train == 1)
        test_defect_count = np.sum(self.y_test == 1)

        print(f"训练集中的缺陷样本数量: {train_defect_count}")
        print(f"测试集中的缺陷样本数量: {test_defect_count}")


class NormalizingFlow(nn.Module):
    """归一化流模型，用于将复杂数据分布转换为简单分布"""

    def __init__(self, dim: int, num_layers: int = 5):
        """
        初始化归一化流模型

        参数:
            dim: 数据维度
            num_layers: 流层数
        """
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.transforms = nn.ModuleList([self._create_transform(dim) for _ in range(num_layers)])
        self.base_dist = MultivariateNormal(torch.zeros(dim), torch.eye(dim))

    def _create_transform(self, dim: int) -> nn.Module:
        """创建单个流变换层"""
        # 使用仿射耦合层
        return AffineCouplingLayer(dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，将数据从复杂分布转换为简单分布

        参数:
            x: 输入数据

        返回:
            z: 转换后的数据
            log_det: 对数行列式
        """
        z = x
        log_det = torch.zeros(z.shape[0])

        for transform in self.transforms:
            z, ld = transform(z)
            log_det += ld

        return z, log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        反向传播，将简单分布转换为复杂分布

        参数:
            z: 来自简单分布的数据

        返回:
            x: 转换后的复杂分布数据
        """
        x = z

        for transform in reversed(self.transforms):
            x = transform.inverse(x)

        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算输入数据的对数概率

        参数:
            x: 输入数据

        返回:
            log_prob: 对数概率
        """
        z, log_det = self.forward(x)
        log_prob = self.base_dist.log_prob(z) + log_det
        return log_prob


class AffineCouplingLayer(nn.Module):
    """仿射耦合层，归一化流中的一种常用变换"""

    def __init__(self, dim: int, hidden_dim: int = 64):
        """
        初始化仿射耦合层

        参数:
            dim: 数据维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.dim = dim
        self.mask = self._create_mask(dim)

        # 计算分割点 - 修改：处理任意维度
        split_dim = dim // 2

        # 创建s和t网络 - 修改：使用正确的输入维度
        self.s_net = nn.Sequential(
            nn.Linear(split_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, dim - split_dim),  # 修改：输出维度调整
            nn.Tanh()
        )

        self.t_net = nn.Sequential(
            nn.Linear(split_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, dim - split_dim)  # 修改：输出维度调整
        )

    def _create_mask(self, dim: int) -> torch.Tensor:
        """创建掩码，将输入分为两部分"""
        mask = torch.zeros(dim)
        split_dim = dim // 2
        mask[:split_dim] = 1
        return mask

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        参数:
            x: 输入数据

        返回:
            z: 转换后的数据
            log_det: 对数行列式
        """
        # 应用掩码
        split_dim = self.dim // 2
        x0 = x[:, :split_dim]  # 修改：明确分割输入
        x1 = x[:, split_dim:]  # 修改：明确分割输入

        # 计算s和t
        s = self.s_net(x0)
        t = self.t_net(x0)

        # 应用变换
        z1 = x1 * torch.exp(s) + t
        z = torch.cat([x0, z1], dim=1)  # 修改：正确拼接结果

        # 计算对数行列式
        log_det = torch.sum(s, dim=1)

        return z, log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        反向传播

        参数:
            z: 输入数据

        返回:
            x: 转换后的数据
        """
        # 应用掩码
        split_dim = self.dim // 2
        z0 = z[:, :split_dim]  # 修改：明确分割输入
        z1 = z[:, split_dim:]  # 修改：明确分割输入

        # 计算s和t
        s = self.s_net(z0)
        t = self.t_net(z0)

        # 应用逆变换
        x1 = (z1 - t) * torch.exp(-s)
        x = torch.cat([z0, x1], dim=1)  # 修改：正确拼接结果

        return x


class DistributionAnalyzer:
    """分布分析器，负责计算分布差异和影响量化分析"""

    def __init__(self, flow_model: NormalizingFlow = None):
        """
        初始化分布分析器

        参数:
            flow_model: 归一化流模型
        """
        self.flow_model = flow_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_flow_model(self, flow_model: NormalizingFlow) -> None:
        """设置归一化流模型"""
        self.flow_model = flow_model
        self.flow_model.to(self.device)

    def train_flow_model(self, X_train: np.ndarray, batch_size: int = 64,
                         epochs: int = 100, lr: float = 0.001) -> None:
        """
        训练归一化流模型

        参数:
            X_train: 训练数据
            batch_size: 批量大小
            epochs: 训练轮数
            lr: 学习率
        """
        if self.flow_model is None:
            raise ValueError("请先设置归一化流模型")

        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)

        # 定义优化器
        optimizer = optim.Adam(self.flow_model.parameters(), lr=lr)

        # 训练循环
        for epoch in range(epochs):
            perm = torch.randperm(len(X_train_tensor))
            total_loss = 0

            for i in range(0, len(X_train_tensor), batch_size):
                batch_idx = perm[i:i + batch_size]
                batch = X_train_tensor[batch_idx]

                # 前向传播
                optimizer.zero_grad()
                log_prob = self.flow_model.log_prob(batch)
                loss = -log_prob.mean()

                # 反向传播
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch)

            # 打印训练信息
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(X_train_tensor)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        print("归一化流模型训练完成")

    def _dequantize_data(self, X: np.ndarray) -> np.ndarray:
        """
        去量化处理：使用样条插值将离散数据转换为连续数据

        参数:
            X: 要处理的数据

        返回:
            去量化后的数据
        """
        df = pd.DataFrame(X)
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            # 检查是否为离散数据
            pass


# 定义DQN网络
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# 定义强化学习环境
class WeightOptimizationEnv:
    def __init__(self, X_train, y_train, X_test, y_test, feature_names=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.num_features = X_train.shape[1]
        self.weights = np.ones(self.num_features) / self.num_features  # 初始权重
        self.state = self.weights.copy()
        self.action_space = self.num_features
        self.observation_space = self.num_features

        # 新增：存储特征名称
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(self.num_features)]

    def step(self, action):
        # 根据动作调整权重
        new_weights = self.weights.copy()
        new_weights[action] += 0.1  # 简单的权重调整
        new_weights = new_weights / np.sum(new_weights)  # 归一化

        # 计算新的距离和奖励
        reward = self.calculate_reward(new_weights)

        # 更新状态
        self.weights = new_weights
        self.state = self.weights.copy()

        return self.state, reward, False, {}

    def reset(self):
        self.weights = np.ones(self.num_features) / self.num_features
        self.state = self.weights.copy()
        return self.state

    def calculate_reward(self, weights):
        # 这里简单使用最近邻分类准确率作为奖励
        nbrs = NearestNeighbors(
            n_neighbors=1,
            metric='minkowski',
            p=2,
            metric_params={'w': weights}  # 修正：通过metric_params传递权重
        )
        nbrs.fit(self.X_train, self.y_train)
        distances, indices = nbrs.kneighbors(self.X_test)
        y_pred = self.y_train[indices.flatten()]
        accuracy = np.mean(y_pred == self.y_test)

        return accuracy

    def get_feature_weights(self):
        """获取特征名称和对应权重的映射"""
        return {name: weight for name, weight in zip(self.feature_names, self.weights)}


# 训练DQN代理
def train_dqn(env, episodes=2000, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
    input_dim = env.observation_space
    output_dim = env.action_space
    model = DQN(input_dim, output_dim)
    target_model = DQN(input_dim, output_dim)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    memory = deque(maxlen=10000)

    for episode in range(episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        total_reward = 0

        for _ in range(100):
            if np.random.rand() <= epsilon:
                action = np.random.choice(env.action_space)
            else:
                q_values = model(state)
                action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            memory.append((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward

            if len(memory) >= 32:
                minibatch = random.sample(memory, 32)
                states, actions, rewards, next_states, dones = zip(*minibatch)

                states = torch.cat(states)
                actions = torch.tensor(actions, dtype=torch.long)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.cat(next_states)
                dones = torch.tensor(dones, dtype=torch.float32)

                q_values = model(states)
                next_q_values = target_model(next_states)
                max_next_q_values = torch.max(next_q_values, dim=1)[0]
                target_q_values = q_values.clone()
                target_q_values[torch.arange(len(actions)), actions] = rewards + (1 - dones) * gamma * max_next_q_values

                loss = criterion(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

    # 返回特征名称和权重的映射
    return env.get_feature_weights()

# 计算加权的分布差异指标
def calculate_weighted_js_divergence(X_train, X_test, weights):
    num_features = X_train.shape[1]
    weighted_js_divergence = 0

    for i in range(num_features):
        train_feature = X_train[:, i]
        test_feature = X_test[:, i]

        # 计算JS散度
        js_div = jensenshannon(train_feature, test_feature)

        # 获取对应特征的权重
        weight = weights[list(weights.keys())[i]]

        # 计算加权的JS散度
        weighted_js_divergence += weight * js_div

    return weighted_js_divergence


if __name__ == "__main__":
    # 数据处理
    data_processor = DataProcessor()
    data_processor.load_data()
    data_processor.prepare_data()

    X_train = data_processor.X_train
    y_train = data_processor.y_train
    X_test = data_processor.X_test
    y_test = data_processor.y_test
    feature_names = data_processor.feature_names  # 获取特征名称

    num_experiments = 5
    all_weights = []

    for i in range(num_experiments):
        print(f"\n实验 {i + 1}/{num_experiments}")

        # 重新初始化环境和模型
        env = WeightOptimizationEnv(X_train, y_train, X_test, y_test, feature_names)
        weights = train_dqn(env)
        all_weights.append(weights)

        # 打印本次实验的权重
        print(f"实验 {i + 1} 的权重:")
        for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            print(f"{name}: {weight:.4f}")

    # 计算平均权重和标准差
    average_weights = {}
    std_weights = {}

    for name in feature_names:
        weights_for_feature = [exp[name] for exp in all_weights]
        average_weights[name] = np.mean(weights_for_feature)
        std_weights[name] = np.std(weights_for_feature)

    # 打印平均结果
    print("\n多次实验的平均权重:")
    for name, weight in sorted(average_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {weight:.4f} ")

    # 计算加权的分布差异指标
    weighted_js_divergence = calculate_weighted_js_divergence(X_train, X_test, average_weights)
    print(f"\n加权的分布差异指标 (JS散度): {weighted_js_divergence:.4f}")

    # 可视化权重 - 使用特征名称
    plt.figure(figsize=(12, 8))
    plt.bar(average_weights.keys(), average_weights.values())
    plt.xlabel('特征名称')
    plt.ylabel('权重')
    plt.title('最优特征权重分布')
    plt.xticks(rotation=90)  # 旋转x轴标签以便更好显示
    plt.tight_layout()  # 调整布局
    plt.show()
