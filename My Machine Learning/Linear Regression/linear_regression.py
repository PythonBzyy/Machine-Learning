import numpy as np
from utils.features import prepare_for_training

class LinearRegression:
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
        1.对数据进行预处理操作
        2.先得到所有的特征个数
        3.初始化参数矩阵
        """
        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data=True)

        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        # 特征数量
        num_features = self.data.shape[1]
        # 初始化参数矩阵
        self.theta = np.zeros((num_features, 1))

    # 训练函数
    def train(self, alpha, num_iterations=500):  # num_iterations迭代次数
        """训练模块，执行梯度下降"""
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iterations):
        """梯度下降迭代模块"""
        cost_history = []  # 损失值
        for _ in range(num_iterations):
            self.gradient_step(alpha)   # 完成一次更新
            cost_history.append(self.cost_function(self.data, self.labels))
        return cost_history

    def gradient_step(self, alpha):
        """梯度下降参数更新(矩阵运算)，执行一次"""
        num_examples = self.data.shape[0]
        prediction = LinearRegression.hypothesis(self.data, self.theta)
        delta = prediction - self.labels    # labels真实值
        theta = self.theta
        theta = theta - alpha * (1/num_examples) * (np.dot(delta.T, self.data)).T   # theta更新
        self.theta = theta

    def cost_function(self, data, labels):
        """损失函数（测试，训练）"""
        num_example = data.shape[0]  # data.shape[0]样本个数
        delta = LinearRegression.hypothesis(self.data, self.theta) - labels  # delta预测值
        cost = (1/2) * np.dot(delta.T, delta) / num_example  # 均方误差
        return cost[0][0]

    @staticmethod
    def hypothesis(data, theta):
        """计算预测值"""
        predictions = np.dot(data, theta)
        return predictions

    def get_cost(self, data, labels):
        data_processed = prepare_for_training(data,
         self.polynomial_degree,
         self.sinusoid_degree,
         self.normalize_data
         )[0]
        return self.cost_function(data_processed, labels)

    def predict(self, data):
        """用训练的参数模型，去预测得到回归值结果"""
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data
                                              )[0]

        predictions = LinearRegression.hypothesis(data_processed, self.theta)
        return predictions