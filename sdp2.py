from collections import deque
## 使用普通方法寻找优化训练集
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors, KernelDensity, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
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
# 原有导入保持不变...
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset
from WMCAVE2 import wm_cvae_data_augmentation
# 新增CIU配置（对齐原论文参数）
CONFIG = {
    "max_iterations": 100,
    "converge_threshold": 10,
    "p": 0.75,
    "k_range": np.arange(1.0, 2.1, 0.1),
    "test_size": 0.2,
    "n_runs": 10,
    "mlp_lr": 0.001,
    "batch_size": 32,
    "random_seed": 42
}
# 固定随机种子（与原CIU代码一致）
np.random.seed(CONFIG["random_seed"])
torch.manual_seed(CONFIG["random_seed"])
torch.cuda.manual_seed(CONFIG["random_seed"])
torch.cuda.manual_seed_all(CONFIG["random_seed"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
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
    def __init__(self, data_path: str = r"D:\projects\pythonProject\PROMISE-backup-master\PROMISE-backup-master\bug-source_data\ant\ant-1.3.csv", dataset_name: str = "pc2"):
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

    def load_data(self) -> pd.DataFrame:
        """加载Promise数据集"""
        if self.data_path:
            # 从本地加载数据
            self.data = pd.read_csv(self.data_path)
            if 'target' in self.data.columns:
                self.data['target'] = self.data['target'].apply(lambda x: 1 if x > 1 else x)
            elif '2' in self.data.columns:
                self.data['2'] = self.data['2'].apply(lambda x: 1 if x > 1 else x)
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


    def prepare_data(self, test_size: float = 1/5, random_state: int = 42) -> None:
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

        X = self.data[feature_columns].values
        X = self.scaler.fit_transform(X)
        self.X =  X
        y = self.data[target_column].values


        # 分割数据集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"训练集大小: {self.X_train.shape}, 测试集大小: {self.X_test.shape}")

        # 统计训练集和测试集中的缺陷样本数量
        train_defect_count = np.sum(self.y_train == 1)
        test_defect_count = np.sum(self.y_test == 1)

        print(f"训练集中的缺陷样本数量: {train_defect_count}")
        print(f"测试集中的缺陷样本数量: {test_defect_count}")



class NormalizingFlow(nn.Module):
    """归一化流模型，用于将复杂数据分布转换为简单分布"""

    def __init__(self, dim: int, num_layers: int = 5, device=None):
        """
        初始化归一化流模型

        参数:
            dim: 数据维度
            num_layers: 流层数
        """
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.transforms = nn.ModuleList([self._create_transform(dim) for _ in range(num_layers)])
        self.to(self.device)  # 将模型参数移动到设备

        # 创建基础分布并确保其参数在正确设备上
        self.base_dist = MultivariateNormal(
            torch.zeros(dim, device=self.device),  # 均值向量
            torch.eye(dim, device=self.device)  # 协方差矩阵
        )

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
        x = x.to(self.device)
        z = x
        # 确保log_det在正确的设备上
        log_det = torch.zeros(z.shape[0], device=self.device)

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
        z = z.to(self.device)
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

    def __init__(self, dim: int, hidden_dim: int = 64, device=None):
        """
        初始化仿射耦合层
        参数:
            dim: 数据维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        self.dim = dim
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 创建掩码并移动到指定设备
        self.mask = self._create_mask(dim).to(self.device)

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
        self.to(self.device)

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
        x = x.to(self.device)
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
        z = z.to(self.device)
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

#定义DQN网络

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(DQN, self).__init__()
        self.device = device  # 保存设备信息
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.to(device)  # 初始化时就迁移到指定设备

    def forward(self, x):
        x = x.to(self.device)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 定义强化学习环境
class WeightOptimizationEnv:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.num_features = X_train.shape[1]
        self.weights = np.ones(self.num_features) / self.num_features  # 初始权重
        self.state = self.weights.copy()
        self.action_space = self.num_features
        self.observation_space = self.num_features

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
        # 使用欧式距离计算奖励
        # 这里使用加权特征重要性来计算距离，而不是直接作为距离度量的参数

        # 对特征进行加权
        X_train_weighted = self.X_train * np.sqrt(weights)
        X_test_weighted = self.X_test * np.sqrt(weights)

        # 使用标准欧氏距离
        nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nbrs.fit(X_train_weighted, self.y_train)

        # 获取最近邻并计算准确率
        distances, indices = nbrs.kneighbors(X_test_weighted)
        y_pred = self.y_train[indices.flatten()]
        accuracy = np.mean(y_pred == self.y_test)

        return accuracy


# 训练DQN代理
def train_dqn(env, episodes=1000, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
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

    return env.weights


# 定义加权距离计算函数
def weighted_euclidean_distance(x1, x2, weights):
    """
    计算加权欧氏距离

    参数:
        x1: 第一个样本
        x2: 第二个样本
        weights: 特征权重

    返回:
        加权欧氏距离
    """
    return np.sqrt(np.sum(weights * (x1 - x2) ** 2))

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
                         epochs: int = 400, lr: float = 0.001) -> None:
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
            if df[col].nunique() < 20 and df[col].dtype != object:
                # 获取唯一值及其频率
                unique_values = df[col].unique()
                counts = df[col].value_counts().sort_index()

                # 创建样条插值函数
                x = np.array(counts.index)
                y = np.array(counts.values)

                # 确保至少有3个点用于样条插值
                if len(x) >= 3:
                    try:
                        # 平滑插值
                        spl = UnivariateSpline(x, y, s=len(x) / 2)

                        # 生成新的连续值
                        new_x = np.linspace(x.min(), x.max(), 100)
                        new_y = spl(new_x)

                        # 确保所有值非负
                        new_y = np.maximum(new_y, 0)

                        # 检查是否所有值都是零
                        if np.sum(new_y) == 0:
                            print(f"警告: 列 {col} 的插值结果全为零，跳过去量化处理")
                            continue

                        # 归一化概率密度
                        new_y = new_y / new_y.sum()

                        # 从插值分布中采样
                        indices = np.random.choice(len(new_x), size=len(df), p=new_y)
                        new_values = new_x[indices]

                        # 更新数据
                        df[col] = new_values
                    except Exception as e:
                        print(f"处理列 {col} 时出错: {e}")
                        continue

        return df.values

    def transform_data(self, X: np.ndarray) -> np.ndarray:
        """
        使用归一化流模型转换数据

        参数:
            X: 输入数据

        返回:
            transformed_X: 转换后的数据
        """
        if self.flow_model is None:
            raise ValueError("请先设置归一化流模型")

        # 去量化处理
        X = self._dequantize_data(X)

        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X).to(self.device)

        # 转换数据
        self.flow_model.eval()
        with torch.no_grad():
            z, _ = self.flow_model(X_tensor)

        return z.cpu().numpy()

    def calculate_distribution_differences(self, X_train: np.ndarray, X_test: np.ndarray,
                                           method: str = None ) -> float:
        """
        计算训练集和测试集之间的分布差异

        参数:
            X_train: 训练数据
            X_test: 测试数据
            method: 差异计算方法，可选"kl_divergence", "js_divergence", "mmd"

        返回:
            difference: 分布差异值
        """
        # 如果有归一化流模型，先转换数据
        if self.flow_model is not None:
            X_train_transformed = self.transform_data(X_train)
            X_test_transformed = self.transform_data(X_test)
        else:
            X_train_transformed = X_train
            X_test_transformed = X_test

        # 根据选择的方法计算分布差异
        if method == "kl_divergence":
            return self._calculate_kl_divergence(X_train_transformed, X_test_transformed)
        elif method == "energy_distance":
            return self._calculate_energy_distance(X_train_transformed, X_test_transformed)
        elif method == "mmd":
            return self._calculate_mmd(X_train_transformed, X_test_transformed)
        elif method == "jsd":
            return self._calculate_JSD(X_train_transformed, X_test_transformed)
        elif method == "weighted_jsd":
            return self._calculate_weighted_JSD(X_train_transformed, X_test_transformed)
        else:
            raise ValueError(f"不支持的方法: {method}")

    def _calculate_kl_divergence(self, X1: np.ndarray, X2: np.ndarray) -> float:
        """计算KL散度"""
        # 对每个特征分别计算KL散度并求和
        kl_div = 0
        n_features = X1.shape[1]

        for i in range(n_features):
            # 估计概率密度
            p = stats.gaussian_kde(X1[:, i])
            q = stats.gaussian_kde(X2[:, i])

            # 生成评估点
            x_min = min(X1[:, i].min(), X2[:, i].min())
            x_max = max(X1[:, i].max(), X2[:, i].max())
            x = np.linspace(x_min, x_max, 100)

            # 计算概率密度值
            p_vals = p(x)
            q_vals = q(x)

            # 避免除零错误
            mask = (p_vals > 1e-10) & (q_vals > 1e-10)
            if np.sum(mask) > 0:
                kl_div += np.sum(p_vals[mask] * np.log(p_vals[mask] / q_vals[mask])) * (x_max - x_min) / 100

        return kl_div

    def _calculate_mmd(self, X1: np.ndarray, X2: np.ndarray, gamma: float = 1.0) -> float:
        """计算最大均值差异(MMD)"""
        X1_tensor = torch.FloatTensor(X1).to(self.device)
        X2_tensor = torch.FloatTensor(X2).to(self.device)

        # 计算MMD
        n = X1_tensor.size(0)
        m = X2_tensor.size(0)

        # 计算核矩阵
        XX = self._compute_rbf_kernel(X1_tensor, X1_tensor, gamma)
        YY = self._compute_rbf_kernel(X2_tensor, X2_tensor, gamma)
        XY = self._compute_rbf_kernel(X1_tensor, X2_tensor, gamma)

        # 计算MMD
        mmd = torch.mean(XX) + torch.mean(YY) - 2 * torch.mean(XY)

        return mmd.item()

    def _compute_rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor, gamma: float) -> torch.Tensor:
        """计算RBF核矩阵"""
        XX = torch.sum(X * X, dim=1).view(-1, 1)
        YY = torch.sum(Y * Y, dim=1).view(1, -1)
        dist = XX + YY - 2 * torch.matmul(X, Y.t())
        return torch.exp(-gamma * dist)

    def _calculate_energy_distance(self, X1: np.ndarray, X2: np.ndarray) -> float:
        """计算能量距离"""
        n = X1.shape[0]
        m = X2.shape[0]

        # 计算所有可能的欧几里得距离
        XX = np.zeros((n, n))
        YY = np.zeros((m, m))
        XY = np.zeros((n, m))

        for i in range(n):
            for j in range(n):
                XX[i, j] = np.linalg.norm(X1[i] - X1[j])

        for i in range(m):
            for j in range(m):
                YY[i, j] = np.linalg.norm(X2[i] - X2[j])

        for i in range(n):
            for j in range(m):
                XY[i, j] = np.linalg.norm(X1[i] - X2[j])

        # 计算能量距离
        term1 = (2.0 / (n * m)) * np.sum(XY)
        term2 = (1.0 / (n * n)) * np.sum(XX)
        term3 = (1.0 / (m * m)) * np.sum(YY)

        return term1 - term2 - term3

    def _calculate_JSD(self, X: np.ndarray, Y: np.ndarray, bandwidth=0.5, n_samples=500):
        """计算两个样本集之间的JS散度"""
        n_features = X.shape[1]
        total_js = 0

        # 生成用于评估密度的样本点（使用两个分布的混合）
        combined = np.vstack([X, Y])
        sample_points = combined[np.random.choice(len(combined), n_samples)]

        for i in range(n_features):
            # 对每个特征进行KDE
            kde_p = KernelDensity(bandwidth=bandwidth).fit(X[:, i].reshape(-1, 1))
            kde_q = KernelDensity(bandwidth=bandwidth).fit(Y[:, i].reshape(-1, 1))

            # 计算对数密度
            log_p = kde_p.score_samples(sample_points[:, i].reshape(-1, 1))
            log_q = kde_q.score_samples(sample_points[:, i].reshape(-1, 1))

            # 转换为概率密度
            p = np.exp(log_p)
            q = np.exp(log_q)

            # 归一化（确保积分为1）
            p = p / np.sum(p)
            q = q / np.sum(q)

            # 计算中点分布
            m = 0.5 * (p + q)

            # 计算JS散度
            kl_pm = np.sum(p * np.log(p / m, out=np.zeros_like(p), where=(p > 0) & (m > 0)))
            kl_qm = np.sum(q * np.log(q / m, out=np.zeros_like(q), where=(q > 0) & (m > 0)))
            js = 0.5 * (kl_pm + kl_qm)

            total_js += js

        return total_js / n_features  # 返回平均JS散度

    def _calculate_weighted_JSD(self, X: np.ndarray, Y: np.ndarray,  bandwidth=0.5, n_samples=500):
        """计算两个样本集之间的JS散度"""
        n_features = X.shape[1]
        total_js = 0

        # 生成用于评估密度的样本点（使用两个分布的混合）
        combined = np.vstack([X, Y])
        sample_points = combined[np.random.choice(len(combined), n_samples)]

        for i in range(n_features):
            # 对每个特征进行KDE
            kde_p = KernelDensity(bandwidth=bandwidth).fit(X[:, i].reshape(-1, 1))
            kde_q = KernelDensity(bandwidth=bandwidth).fit(Y[:, i].reshape(-1, 1))

            # 计算对数密度
            log_p = kde_p.score_samples(sample_points[:, i].reshape(-1, 1))
            log_q = kde_q.score_samples(sample_points[:, i].reshape(-1, 1))

            # 转换为概率密度
            p = np.exp(log_p)
            q = np.exp(log_q)

            # 归一化（确保积分为1）
            p = p / np.sum(p)
            q = q / np.sum(q)

            # 计算中点分布
            m = 0.5 * (p + q)

            # 计算JS散度
            kl_pm = np.sum(p * np.log(p / m, out=np.zeros_like(p), where=(p > 0) & (m > 0)))
            kl_qm = np.sum(q * np.log(q / m, out=np.zeros_like(q), where=(q > 0) & (m > 0)))
            js = 0.5 * (kl_pm + kl_qm)
            weight = optimal_weights[i]
            total_js += js * weight

        return total_js   # 返回平均JS散度

    import numpy as np
    from typing import Dict
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, recall_score, confusion_matrix,
                                 precision_score, f1_score)
    from sklearn.model_selection import GridSearchCV  # 导入网格搜索
    from sklearn.exceptions import UndefinedMetricWarning
    import warnings

    # 忽略无关警告（如少数类无预测样本时的警告）
    warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

    def analyze_impact(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray,
                       y_test: np.ndarray, model_type: str = "random_forest",
                       use_hyper_opt: bool = True,  # 新增：是否启用超参数优化
                       cv: int = 5,  # 新增：交叉验证折数
                       scoring: str = "f1"  # 新增：超参优化的评分指标
                       ) -> Dict[str, float]:
        """
        评估分布差异对模型性能的影响（新增超参数优化）

        参数:
            X_train: 训练数据
            X_test: 测试数据
            y_train: 训练标签
            y_test: 测试标签
            model_type: 模型类型，可选"random_forest", "logistic_regression", "svm"
            use_hyper_opt: 是否启用超参数优化（默认True）
            cv: 交叉验证折数（默认5，样本量小时可设为3）
            scoring: 超参优化的评分指标（默认f1，适配不平衡数据）

        返回:
            metrics: 模型性能指标
        """
        # ========== 1. 定义各模型的超参数网格（核心新增） ==========
        param_grids = {
            "random_forest": {
                "n_estimators": [200, 500, 1000],  # 决策树数量
                "max_depth": [None, 10, 20, 30],  # 树最大深度
                "min_samples_split": [2, 5, 10],  # 分裂所需最小样本数
            },
        }

        # ========== 2. 创建基础模型 ==========
        if model_type == "random_forest":
            base_model = RandomForestClassifier(
                n_estimators=1000,
                max_depth=None,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1  # CPU多线程加速
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        # ========== 3. 超参数优化（核心新增） ==========
        if use_hyper_opt:
            # 初始化网格搜索（n_jobs=-1：使用所有CPU核心加速）
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grids[model_type],
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
            )
            # 训练并寻找最优超参数
            grid_search.fit(X_train, y_train)
            # 使用最优参数的模型
            model = grid_search.best_estimator_
        else:
            # 不启用调优，使用基础模型（兼容原有逻辑）
            model = base_model
            model.fit(X_train, y_train)

        # ========== 4. 预测与指标计算（原有逻辑，仅少量优化） ==========
        y_pred = model.predict(X_test)
        # 计算指标（增加分母为0的防护）
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        return {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1_score": f1
        }

    def weighted_minkowski_distance(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray, p: float) -> float:
        """
        计算加权Minkowski距离

        参数:
            x: 第一个样本
            y: 第二个样本
            weights: 特征权重向量
            p: Minkowski参数 (p=1: Manhattan, p=2: Euclidean)

        返回:
            加权Minkowski距离
        """
        # 确保输入有效
        if len(x) != len(y) or len(x) != len(weights):
            raise ValueError("输入向量和权重必须具有相同的维度")

        # 计算加权Minkowski距离
        return np.sum(weights * np.abs(x - y) ** p) ** (1 / p)

    def optimize_training_set_selection(self, X_pool: np.ndarray, y_pool: np.ndarray,
                                        X_test: np.ndarray, num_samples,
                                        method: str = "closest_distribution",
                                        distance_metric: str = "euclidean",
                                        use_stratified: bool = False,
                                        use_approximate: bool = False,
                                        p: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:


        if method == "random":
            # 随机选择
            indices = np.random.choice(len(X_pool), size=num_samples, replace=False)
            return X_pool[indices], y_pool[indices]
        elif method == "closest_distribution":
            # 如果有归一化流模型，使用它转换数据
            if self.flow_model is not None:
                X_pool_transformed = self.transform_data(X_pool)
                X_test_transformed = self.transform_data(X_test)
            else:
                X_pool_transformed = X_pool
                X_test_transformed = X_test

            # 计算距离
            if use_approximate:
                # 使用近似最近邻加速（仅支持部分度量）
                supported_metrics = ['euclidean', 'weighted_jsd', 'minkowski']
                if distance_metric not in supported_metrics:
                    raise ValueError(f"近似模式下不支持该距离度量: {distance_metric}")

                nbrs = NearestNeighbors(
                    n_neighbors=min(len(X_test_transformed), 100),
                    algorithm='ball_tree',
                    metric=distance_metric,
                    p=p  # 用于Minkowski距离
                ).fit(X_test_transformed)

                distances, _ = nbrs.kneighbors(X_pool_transformed)
                avg_distances = np.mean(distances, axis=1)
            else:
                # 精确计算距离
                if distance_metric == "euclidean":
                    # 向量化欧氏距离计算
                    diff = X_pool_transformed[:, np.newaxis] - X_test_transformed
                    distances = np.mean(np.linalg.norm(diff, axis=2), axis=1)

                elif distance_metric == "minkowski":
                    # Minkowski距离 (推广的Lp范数)
                    diff = X_pool_transformed[:, np.newaxis] - X_test_transformed
                    distances = np.mean(np.sum(np.abs(diff) ** p, axis=2) ** (1 / p), axis=1)


                elif distance_metric == "weighted_jsd":
                    # 加权JS散度（Weighted Jensen-Shannon Divergence）
                    # JS散度范围：[0, ln2]，值越小表示分布越相似

                    # 1. 数据归一化到[epsilon, 1-epsilon]区间，避免log(0)
                    epsilon = 1e-10

                    # 对每个样本的特征进行归一化，使其和为1（转换为概率分布）
                    def normalize_to_distribution(X, weights):
                        # 应用特征权重
                        X_weighted = X * weights[np.newaxis, :]
                        # 归一化每行（每个样本）为概率分布
                        X_sum = X_weighted.sum(axis=1, keepdims=True)
                        # 处理全零行
                        X_sum[X_sum == 0] = 1e-10
                        X_dist = X_weighted / X_sum
                        # 数值稳定性处理
                        X_dist = np.clip(X_dist, epsilon, 1 - epsilon)
                        return X_dist

                    X_pool_dist = normalize_to_distribution(X_pool_transformed, optimal_weights)
                    X_test_dist = normalize_to_distribution(X_test_transformed, optimal_weights)

                    # 2. 向量化计算JS散度
                    # 对于每个池化样本，计算与所有测试样本的JS散度，然后取平均
                    num_pool = len(X_pool_dist)
                    num_test = len(X_test_dist)

                    # 扩展维度以便广播
                    P = X_pool_dist[:, np.newaxis, :]  # (num_pool, 1, n_features)
                    Q = X_test_dist[np.newaxis, :, :]  # (1, num_test, n_features)

                    # 计算混合分布 M = 0.5*(P + Q)
                    M = 0.5 * (P + Q)

                    # 计算加权KL散度：KL(P||M) = sum(w_i * P_i * log(P_i/M_i))
                    kl_pm = np.sum(P * np.log(P / M), axis=2)
                    kl_qm = np.sum(Q * np.log(Q / M), axis=2)

                    # JS散度 = 0.5*(KL(P||M) + KL(Q||M))
                    jsd = 0.5 * (kl_pm + kl_qm)

                    # 计算每个池化样本与所有测试样本的平均JS散度
                    distances = np.mean(jsd, axis=1)
                else:
                    raise ValueError(f"不支持的距离度量: {distance_metric}")

                avg_distances = distances

            # 选择样本（保持原有逻辑不变）
            sorted_indices = np.argsort(avg_distances)

            if use_stratified and len(np.unique(y_pool)) > 1 and num_samples < len(X_pool):
                # 分层抽样逻辑（保持原有代码不变）
                stratified_indices = []
                for label in np.unique(y_pool):
                    label_indices = sorted_indices[y_pool[sorted_indices] == label]
                    label_samples = int(num_samples * len(label_indices) / len(X_pool))
                    label_samples = max(1, label_samples)
                    stratified_indices.extend(label_indices[:label_samples])

                if len(stratified_indices) < num_samples:
                    remaining = list(set(sorted_indices) - set(stratified_indices))
                    stratified_indices.extend(remaining[:num_samples - len(stratified_indices)])

                indices = np.array(stratified_indices)
            else:
                indices = sorted_indices[:num_samples]

            return X_pool[indices], y_pool[indices]
        else:
            raise ValueError(f"不支持的方法: {method}")

class Visualizer:
    """可视化工具类，用于数据和分析结果的可视化"""

    def plot_feature_importance(self, feature_importance: pd.DataFrame, top_n: int = 10) -> None:
        """
        绘制特征重要性图

        参数:
            feature_importance: 特征重要性数据框
            top_n: 显示前n个特征
        """
        plt.figure(figsize=(10, 6))
        top_features = feature_importance.sort_values('importance', ascending=False).head(top_n)
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title('特征重要性')
        plt.tight_layout()
        plt.show()

    def plot_distribution_comparison(self, X_train: np.ndarray, X_test: np.ndarray,
                                     feature_names: List[str], feature_indices: List[int] = None) -> None:
        """
        绘制训练集和测试集特征分布比较图

        参数:
            X_train: 训练数据
            X_test: 测试数据
            feature_names: 特征名称列表
            feature_indices: 要绘制的特征索引列表，如果为None则绘制前5个特征
        """
        if feature_indices is None:
            feature_indices = list(range(min(5, X_train.shape[1])))

        plt.figure(figsize=(15, 10))

        for i, idx in enumerate(feature_indices):
            plt.subplot(2, 3, i + 1)
            sns.kdeplot(X_train[:, idx], label='trainset')
            sns.kdeplot(X_test[:, idx], label='testset')
            plt.title(f'{feature_names[idx]} distribution')
            plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_performance_comparison(self, metrics_before: Dict[str, float],
                                    metrics_after: Dict[str, float]) -> None:
        """
        绘制优化前后模型性能比较图

        参数:
            metrics_before: 优化前的性能指标
            metrics_after: 优化后的性能指标
        """
        metrics = list(metrics_before.keys())
        values_before = list(metrics_before.values())
        values_after = list(metrics_after.values())

        x = np.arange(len(metrics))
        width = 0.35

        plt.figure(figsize=(10, 6))
        plt.bar(x - width / 2, values_before, width, label='优化前')
        plt.bar(x + width / 2, values_after, width, label='优化后')

        plt.ylabel('分数')
        plt.title('模型性能比较')
        plt.xticks(x, metrics)
        plt.legend()

        plt.tight_layout()
        plt.show()


def knn_data_filter(X_train, y_train, n_neighbors=3, confidence_threshold=0.5):
    """
    使用KNN筛选训练集：剔除与多数邻近样本类别不一致的噪声样本
    Args:
        X_train: 训练特征（n_samples, n_features）
        y_train: 训练标签（n_samples,）
        n_neighbors: KNN的邻居数量（默认5）
        confidence_threshold: 类别一致性阈值（默认0.5，即超过一半邻居类别一致则保留）
    Returns:
        X_filtered: 筛选后的训练特征
        y_filtered: 筛选后的训练标签
    """
    print(f"\n=== 开始KNN数据筛选（K={n_neighbors}）===")
    print(f"筛选前训练集：{X_train.shape}，类别分布：缺陷{sum(y_train)}，无缺陷{len(y_train) - sum(y_train)}")

    # 初始化KNN分类器（使用距离加权）
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    knn.fit(X_train, y_train)

    # 对每个训练样本，找到其K个邻居并统计类别
    neighbor_indices = knn.kneighbors(X_train, return_distance=False)  # (n_samples, n_neighbors)
    keep_mask = []  # 保留样本的掩码

    for i in range(len(X_train)):
        # 获取第i个样本的邻居标签（排除自身）
        neighbor_labels = y_train[neighbor_indices[i]]
        # 统计邻居中各类别的占比
        positive_ratio = sum(neighbor_labels == 1) / n_neighbors
        negative_ratio = sum(neighbor_labels == 0) / n_neighbors

        # 若样本类别与占比最高的邻居类别一致，则保留
        sample_label = y_train[i]
        if (sample_label == 1 and positive_ratio >= confidence_threshold) or \
                (sample_label == 0 and negative_ratio >= confidence_threshold):
            keep_mask.append(True)
        else:
            keep_mask.append(False)

    # 筛选样本
    keep_mask = np.array(keep_mask)
    X_filtered = X_train[keep_mask]
    y_filtered = y_train[keep_mask]


    return X_filtered, y_filtered

# ====================== CIU核心组件（适配sdp2架构） ======================
class AdaptiveMLP(nn.Module):
    """自适应MLP分类器（来自CIU论文）"""
    def __init__(self, input_dim, device):
        super(AdaptiveMLP, self).__init__()
        self.device = device
        if input_dim <= 10:
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 2)
            )
        elif input_dim <= 20:
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            )
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        return self.layers(x)


def train_mlp(model, X_train, y_train, X_val=None, y_val=None, device=None):
    """训练MLP（适配sdp2的device逻辑）"""
    device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["mlp_lr"])

    best_auc = 0.0
    best_model_state = None

    for epoch in range(50):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证集评估
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(torch.tensor(X_val, dtype=torch.float32).to(device))
                val_probs = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
                val_auc = roc_auc_score(y_val, val_probs)
            if val_auc > best_auc:
                best_auc = val_auc
                best_model_state = model.state_dict().copy()

    # 加载最优模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, best_auc if X_val is not None else 0.0


def predict_mlp(model, X, device=None):
    """MLP预测（输出少数类概率）"""
    device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
    return probs


def clustering_iterative_undersampling(X_major, X_minor, y_minor, device=None):
    """CIU核心算法（适配sdp2数据结构）"""
    device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_minor = len(X_minor)
    m_major = len(X_major)
    first_k = True
    input_dim = X_major.shape[1]

    # 步骤1：k调优
    best_k = 1.0
    best_score = 0.0
    for k in CONFIG["k_range"]:
        l = int(k * n_minor)
        if l > m_major:
            print(f"k={k}时l={l}>m={m_major}，停止遍历更大k，当前最优k={best_k}")
            break
        l = max(1, l)

        # K-means聚类
        kmeans = KMeans(n_clusters=l, random_state=CONFIG["random_seed"], n_init=20)
        cluster_labels = kmeans.fit_predict(X_major)
        cluster_centers = kmeans.cluster_centers_

        # 构建初始训练集
        X_train_init = np.vstack([cluster_centers, X_minor])
        y_train_init = np.hstack([np.zeros(l), y_minor])
        X_init_tr, X_init_val, y_init_tr, y_init_val = train_test_split(
            X_train_init, y_train_init, test_size=0.2, random_state=CONFIG["random_seed"], stratify=y_train_init
        )

        # 训练MLP
        model = AdaptiveMLP(input_dim, device)
        model, auc_score = train_mlp(model, X_init_tr, y_init_tr, X_init_val, y_init_val, device)

        # 更新最优k
        if first_k:
            best_score = auc_score
            best_k = k
            first_k = False
        else:
            if auc_score > best_score:
                best_score = auc_score
                best_k = k

    # 步骤2：迭代更新聚类中心
    l_opt = int(best_k * n_minor)
    l_opt = max(1, min(l_opt, m_major))
    kmeans = KMeans(n_clusters=l_opt, random_state=CONFIG["random_seed"], n_init=20)
    cluster_labels = kmeans.fit_predict(X_major)
    cluster_centers = kmeans.cluster_centers_

    current_centers = cluster_centers.copy()
    no_improve_count = 0
    best_auc = 0.0
    best_centers = current_centers.copy()

    for iter_idx in range(CONFIG["max_iterations"]):
        # 构建当前训练集
        X_train_current = np.vstack([current_centers, X_minor])
        y_train_current = np.hstack([np.zeros(l_opt), y_minor])

        # 训练MLP
        model = AdaptiveMLP(input_dim, device)
        model, current_auc = train_mlp(model, X_train_current, y_train_current, None, None, device)

        # 收敛判断
        if current_auc > best_auc:
            best_auc = current_auc
            best_centers = current_centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= CONFIG["converge_threshold"]:
                break

        # 预测多数类样本概率
        major_probs = predict_mlp(model, X_major, device)

        # 高斯分布采样更新聚类中心
        new_centers = []
        for cluster_id in range(l_opt):
            cluster_sample_idx = cluster_labels == cluster_id
            cluster_samples = X_major[cluster_sample_idx]
            cluster_probs = major_probs[cluster_sample_idx]

            if len(cluster_samples) == 0:
                new_centers.append(current_centers[cluster_id])
                continue

            # 选择top-p%样本
            n_selected = max(1, int(len(cluster_samples) * CONFIG["p"]))
            top_idx = np.argsort(cluster_probs)[-n_selected:]
            selected_samples = cluster_samples[top_idx]

            # 筛选μ±σ范围内样本
            mu = np.mean(selected_samples, axis=0)
            sigma = np.std(selected_samples, axis=0) + 1e-8
            lower_bound = mu - sigma
            upper_bound = mu + sigma
            in_range_mask = np.all((selected_samples >= lower_bound) & (selected_samples <= upper_bound), axis=1)
            in_range_samples = selected_samples[in_range_mask]

            # 兜底逻辑
            if len(in_range_samples) == 0:
                in_range_samples = selected_samples[np.argmax(cluster_probs[top_idx]):top_idx[np.argmax(cluster_probs[top_idx])] + 1]

            new_center = np.mean(in_range_samples, axis=0)
            new_centers.append(new_center)

        current_centers = np.array(new_centers)

    # 生成最终欠采样训练集
    X_train_ciu = np.vstack([best_centers, X_minor])
    y_train_ciu = np.hstack([np.zeros(l_opt), y_minor])
    return X_train_ciu, y_train_ciu, best_k




def main():
    # 示例：使用Promise数据集进行软件缺陷预测
    # 初始化数据处理器
    #  D:\projects\pythonProject\uci_heart_disease_clean.csv

    data_processor = DataProcessor(data_path ,dataset_name="")
    # 加载数据
    data = data_processor.load_data()
    # 特征选择
    #data_processor.select_features()
    # 准备训练集和测试集
    data_processor.prepare_data()

    # 获取处理后的数据
    X_train, X_test = data_processor.X_train, data_processor.X_test
    y_train, y_test = data_processor.y_train, data_processor.y_test

    # 初始化归一化流模型
    flow_model = NormalizingFlow(dim=X_train.shape[1])

    # 初始化分布分析器
    analyzer = DistributionAnalyzer(flow_model)

    # 训练归一化流模型
    analyzer.train_flow_model(X_train)


    # 原数据集
    ed = analyzer.calculate_distribution_differences(X_train, X_test, method="energy_distance")
    jsd = analyzer.calculate_distribution_differences(X_train, X_test, method="jsd")
    mmd = analyzer.calculate_distribution_differences(X_train, X_test, method="mmd")
    weighted_jsd = analyzer.calculate_distribution_differences(X_train, X_test, method="weighted_jsd")
    print(f"\n优化前能量距离：{ed:.4f} MMD: {mmd:.4f}  JSD: {jsd:.4f}  加权JSD：{weighted_jsd:.4f}")
    metrics_before = analyzer.analyze_impact(X_train, X_test, y_train, y_test)
    print("优化前模型性能指标:")
    for metric, value in metrics_before.items():
        print(f"{metric}: {value:.4f}")

    # SMOTE优化
    smote = SMOTE(k_neighbors=5,random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    ed = analyzer.calculate_distribution_differences(X_res, X_test, method="energy_distance")
    jsd = analyzer.calculate_distribution_differences(X_res, X_test, method="jsd")
    mmd = analyzer.calculate_distribution_differences(X_res, X_test, method="mmd")
    weighted_jsd = analyzer.calculate_distribution_differences(X_res, X_test, method="weighted_jsd")
    print(f"\nSMOTE后能量距离：{ed:.4f} MMD: {mmd:.4f}  JSD: {jsd:.4f} 加权JSD：{weighted_jsd:.4f}")
    metrics_aftersmote = analyzer.analyze_impact(X_res, X_test, y_res, y_test)
    print("\nSMOTE优化后模型性能指标:")
    for metric, value in metrics_aftersmote.items():
           print(f"{metric}: {value:.4f}")

    #聚类迭代
    X_major_train = X_train[y_train == 0]
    y_major_train = y_train[y_train == 0]
    X_minor_train = X_train[y_train == 1]
    y_minor_train = y_train[y_train == 1]
    X_train_ciu, y_train_ciu, best_k = clustering_iterative_undersampling(
        X_major_train, X_minor_train, y_minor_train
    )
    ed = analyzer.calculate_distribution_differences(X_train_ciu, X_test, method="energy_distance")
    jsd = analyzer.calculate_distribution_differences(X_train_ciu, X_test, method="jsd")
    mmd = analyzer.calculate_distribution_differences(X_train_ciu, X_test, method="mmd")
    weighted_jsd = analyzer.calculate_distribution_differences(X_train_ciu, X_test, method="weighted_jsd")
    print(f"\n聚类迭代后能量距离：{ed:.4f} MMD: {mmd:.4f}  JSD: {jsd:.4f} 加权JSD：{weighted_jsd:.4f}")
    metrics_aftersmote = analyzer.analyze_impact(X_train_ciu, X_test, y_train_ciu, y_test)
    print("\n聚类迭代后模型性能指标:")
    for metric, value in metrics_aftersmote.items():
        print(f"{metric}: {value:.4f}")


    #WM-cave
    X_train_wmcave, y_train_wmcave = wm_cvae_data_augmentation(X_train,y_train)
    metrics_aftersmote = analyzer.analyze_impact(X_train_wmcave, X_test, y_train_wmcave, y_test)
    print("\nWMCAVE优化后模型性能指标:")
    for metric, value in metrics_aftersmote.items():
        print(f"{metric}: {value:.4f}")



    X_pool = X_res
    y_pool = y_res

    # SMOTE-ENN
    enn = EditedNearestNeighbours(
        n_neighbors=3,  # ENN的近邻数（参数名是n_neighbors，不是enn_neighbors）
        kind_sel='all'  # 筛选规则：all表示近邻全为异类才剔除，默认即可
    )
    smote_enn = SMOTEENN(
        smote=smote,  # 传入自定义SMOTE实例
        enn=enn,  # 传入自定义ENN实例
        sampling_strategy='auto',
        random_state=42
    )
    X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
    ed = analyzer.calculate_distribution_differences(X_resampled, X_test, method="energy_distance")
    jsd = analyzer.calculate_distribution_differences(X_resampled, X_test, method="jsd")
    mmd = analyzer.calculate_distribution_differences(X_resampled, X_test, method="mmd")
    weighted_jsd = analyzer.calculate_distribution_differences(X_resampled, X_test, method="weighted_jsd")
    print(f"\nSMOTE-ENN后能量距离：{ed:.4f} MMD: {mmd:.4f}  JSD: {jsd:.4f} 加权JSD：{weighted_jsd:.4f}")
    metrics_aftersmote = analyzer.analyze_impact(X_resampled, X_test, y_resampled, y_test)
    print("\nSMOTE-ENN优化后模型性能指标:")
    for metric, value in metrics_aftersmote.items():
        print(f"{metric}: {value:.4f}")

    X_selected, y_selected = analyzer.optimize_training_set_selection(
        X_pool, y_pool, X_test, num_samples=4 * len(X_test), method="closest_distribution",
        distance_metric="euclidean"
    )
    ed = analyzer.calculate_distribution_differences(X_selected, X_test, method="energy_distance")
    jsd = analyzer.calculate_distribution_differences(X_selected, X_test, method="jsd")
    mmd = analyzer.calculate_distribution_differences(X_selected, X_test, method="mmd")
    weighted_jsd = analyzer.calculate_distribution_differences(X_selected, X_test, method="weighted_jsd")
    print(f"\n欧氏距离筛选后能量距离：{ed:.4f} MMD: {mmd:.4f}  JSD: {jsd:.4f} 加权JSD：{weighted_jsd:.4f}")
    # 使用优化后的训练集重新评估模型性能
    metrics_after = analyzer.analyze_impact(X_selected, X_test, y_selected, y_test)
    print("优化后模型性能指标:")
    for metric, value in metrics_after.items():
        print(f"{metric}: {value:.4f}")


    X_selected, y_selected = analyzer.optimize_training_set_selection(
        X_pool, y_pool, X_test, num_samples=4 * len(X_test), method="closest_distribution",
        distance_metric="weighted_jsd"
    )
    ed = analyzer.calculate_distribution_differences(X_selected, X_test, method="energy_distance")
    jsd = analyzer.calculate_distribution_differences(X_selected, X_test, method="jsd")
    mmd = analyzer.calculate_distribution_differences(X_selected, X_test, method="mmd")
    weighted_jsd = analyzer.calculate_distribution_differences(X_selected, X_test, method="weighted_jsd")
    print(f"\n加权JS散度筛选后能量距离：{ed:.4f} MMD: {mmd:.4f}  JSD: {jsd:.4f} 加权JSD：{weighted_jsd:.4f}")
    # 使用优化后的训练集重新评估模型性能
    metrics_after = analyzer.analyze_impact(X_selected, X_test, y_selected, y_test)
    print("优化后模型性能指标:")
    for metric, value in metrics_after.items():
        print(f"{metric}: {value:.4f}")

    visualizer = Visualizer()
    # 假设我们有特征名称
    feature_names = data.columns.tolist()[:-1]  # 最后一列是目标变量
    # 绘制分布比较
    visualizer.plot_distribution_comparison(X_selected, X_test, feature_names)
    # 绘制性能比较
    visualizer.plot_performance_comparison(metrics_before, metrics_after)


# jedit-3.2: 0.0104,0.2022,0.0375,0.0161,0.0219,0.0337,0.0144,0.0030,0.0033,0.0059,0.0108,0.0556,0.0611,0.0233,0.2757, 0.1081,0.0464,0.0172,0.0291,0.0244
# jedit-4.0: 0.1598,0.0942,0.1175,0.0266,0.0162,0.0557,0.0025,0,0.0010,0.0060,0.0054,0.0688,0.0005,0.0281,0.0182,0.0889,0.0209,0,0.0831,0.2067
# jedit-4.1: 0.0284, 0.0437, 0.0617, 0.0011, 0.0444, 0.1086, 0.0292, 0.0111, 0.1940, 0.0181, 0.0819, 0.0007, 0.1340, 0.0481, 0.0182, 0.0289, 0.0096, 0.0635, 0.0322, 0.0426
# camel-1.2: 0.0017,0.0209,0.0903,0.0030,0.0257,0.0298,0.1009,0.0842,0.0023,0.0448,0.0575,0.0151,0.0382,0.2349,0.0915,0.0468,0.0157,0.0798,0.0002,0.0167
# camel-1.4: 0.0701,0.0866,0.0394,0.0000,0.0587,0.1048,0.0108,0.0483,0.0418,0.0757,0.0911,0.0705,0.0005,0.0548,0.0611,0.1323,0.0119,0.0001,0.0077,0.0257
# ant-1.3:   0.0885,0.0439,0.0146,0.1504,0.0158,0.0212,0.0321,0.0624,0.0009,0.0339,0.0253,0.0143,0.0817,0.1105,0.0374,0.1312,0.0241,0.0000,0.0109,0.1008
# ant-1.4:   0.0387,0.0465,0.0849,0.0207,0.0160,0.0001,0.0524,0.1479,0.0214,0.0483,0.0013,0.1689,0.0923,0.0164,0.1504,0.0470,0.0000,0.0000,0.0014,0.0455
# log4j-1.0： 0.0017,0.0313,0.0378,0.1345,0.0735,0.0731,0.0822,0.0064,0.036,0.0593,0.0011,0.0649,0.1112,0.0258,0.1092,0.0739,0.0333,0.0022,0.0123,0.0303
# log4j-1.1： 0.0018,0.0084,0.0486,0.0199,0.0127,0.0169,0.0003,0.0706,0.0317,0.0282,0.0011,0.041,0.0006,0.0746,0.2247,0.0152,0.0011,0.2797,0.0485,0.0745
# log4j-1.2： 0.0349,0.0002,0.0524,0.0007,0.0802,0.0283,0.0919,0.0,0.0971,0.0232,0.0633,0.0437,0.0565,0.0312,0.0732,0.0593,0.009,0.0646,0.0382,0.1522
# uci_heart_disease: 0.0196,0.1176,0.018,0.0009,0.3763,0.0603,0.0198,0.1782,0.0483,0.0001,0.0008,0.1355,0.0245
# wisconsin_breast_cancer： 0.058,0.0259,0.0047,0.0463,0.0108,0.002,0.0211,0.0912,0.0015,0.0202,0.0033,0.0766,0.0412,0.0022,0.0466,
# 0.0229,0.0365,0.0007,0.0028,0.0517,0.0123,0.0016,0.0425,0.0218,0.094,0.0324,0.1061,0.1005,0.0119,0.0107
# uci_pima_Indians_Diabetes:0.1558,0.0606,0.4834,0.0598,0.0052,0.125,0.0283,0.0818
# balance: 0.346,0.0588,0.1493,0.4459
# bupa: 0.1668,0.0320,0.3818,0.0569,0.2279,0.1347
# glass: 0.3322,0.1141,0.1871,0.0277,0.0006,0.0761,0.1984,0.059,0.0048
# hayes-roth: 0.0834,0.1927,0.4328,0.2153,0.0758
# statlog-germen:0.0145,0.0132,0.0740,0.0162,0.0000,0.0154,0.0002,0.0779,0.0424,0.0759,0.0003,0.0187,0.0021,0.0002,0.0069,0.0001,0.0045,0.0071,0.0094,0.0165,0.0838,0.0135,0.0974,0.0009,0.0327,0.0000,0.0265,0.0136,0.0237,0.0242,0.0201,0.0186,0.0060,0.1272,0.0697,0.0409,0.0058


# r"D:\projects\pythonProject\source_data\PROMISE-backup-master\PROMISE-backup-master\bug-data\ant\ant-1.3.csv"
data_path = r"D:\projects\pythonProject\processed_data\imbalance_data\glass.csv"
optimal_weights = np.array(
    [0.3322,0.1141,0.1871,0.0277,0.0006,0.0761,0.1984,0.059,0.0048])

if __name__ == "__main__":
    main()