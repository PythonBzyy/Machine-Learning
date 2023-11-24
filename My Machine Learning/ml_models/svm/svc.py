import numpy as np
import random
import matplotlib.pyplot as plt
from utils import kernels, utils


class SVC:

    def __init__(self, max_iter=100, C=1.0, tol=1e-4, kernel=None, degree=3, gamma=0.1):
        """
        
        :param max_iter: 最大迭代次数
        :param C: 正则化系数
        :param tol: 提前中止训练时的误差上限
        :param kernel: 核函数
        :param degree: poly核参数
        :param gamma: rbf核参数
        """
        self.b = None
        self.alpha = None
        self.E = None
        self.max_iter = max_iter
        self.C = C
        self.tol = tol
        if kernel is None:
            self.kernel = kernels.kernel_rbf(gamma)
        elif kernel == 'poly':
            self.kernel = kernels.kernel_poly(degree)
        elif kernel == 'linear':
            self.kernel = kernels.kernel_linear()
        else:
            self.kernel = kernels.kernel_rbf(gamma)

        self.support_vectors = None  # 记录支持向量
        self.support_vector_x = []
        self.support_vector_y = []
        self.support_vector_alpha = []

    def init_params(self, features, labels):
        """
        初始化参数
        """
        self.X = features
        self.y = labels
        self.n_samples, self.n_features = features.shape
        self.w = None
        self.b = .0
        self.alpha = np.zeros(self.n_samples)
        self.E = [self._E(i) for i in range(self.n_samples)]

    def _f(self, i):
        """
        计算 f(x)
        :param i:
        :return:
        """
        x = np.array([self.X[i]])
        if len(self.support_vectors) == 0:
            if x.ndim <= 1:
                return 0
            return np.zeros((x.shape[:-1]))
        else:
            if x.ndim <= 1:
                wx = 0
            else:
                wx = np.zeros((x.shape[:-1]))
            for j in range(len(self.support_vectors)):
                wx += self.kernel([self.X[i]], [self.support_vector_x[j]]) * \
                      self.support_vector_alpha[j] * self.support_vector_y[j]
            return wx + self.b

        # r = self.b
        # for j in range(self.n_features):
        #     r += self.alpha[j] * self.y[j] * self.kernel(self.X[i], self.X[j])
        # return r

    def _E(self, i):
        """
        计算E，E为g(x)对输入x的预测值和y的差
        """
        return self._f(i) - self.y[i]

    def _kkt_condition(self, i):
        """
        判断是否满足KKT条件
        :param i:
        :return:
        """
        y_f = self._f(i) * self.y[i]
        if self.alpha[i] == 0:
            return y_f >= 1 - self.tol
        elif 0 < self.alpha[i] < self.C:
            return y_f == 1
        else:
            return y_f <= 1 + self.tol

    def _select_j(self, best_i):
        """
        选择最优 j
        """
        list_j_valid = [i for i in range(self.n_samples) if self.alpha[i] > 0 and i != best_i]
        best_j = -1
        # 优先选择使得|E_i-E_j|最大的j
        if len(list_j_valid) > 0:
            max_E = 0
            for j in list_j_valid:
                cur_E = np.abs(self.E[best_i] - self.E[j])
                if cur_E > max_E:
                    best_j = j
                    max_E = cur_E
        else:
            # random
            l = list(range(len(self.alpha)))
            seq = l[: best_i] + l[best_i + 1:]
            best_j = random.choice(seq)
        return best_j

    def _cut(self, alpha, L, H):
        if alpha > H:
            return H
        elif alpha < L:
            return L
        else:
            return alpha

    def fit(self, features, labels, show_train_process=False):
        labels[labels == 0] = -1
        self.init_params(features, labels)  # 初始化参数
        for _ in range(self.max_iter):
            flag_kkt = True
            for i in range(self.n_samples):
                alpha_i_old = self.alpha[i].copy()
                E_i_old = self.E[i].copy()
                if not self._kkt_condition(i):
                    flag_kkt = False
                    # 1. 得到最优的 alpha_2
                    j = self._select_j(i)

                    alpha_j_old = self.alpha[j].copy()
                    E_j_old = self.E[j].copy()

                    # 2. 得到 alpha_2_new_unc, 剪辑得到 alpha_2_new
                    if self.y[i] == self.y[j]:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    else:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])

                    # eta = K_11 + K_22 -2K_12
                    eta = self.kernel([self.X[i]], [self.X[i]]) + \
                          self.kernel([self.X[j]], [self.X[j]]) - 2 * \
                          self.kernel([self.X[i]], [self.X[j]])
                    if eta < 1e-3:  # 如果 X_i 和 X_j 很接近
                        continue

                    alpha_j_new_unc = alpha_j_old + self.y[j] * (E_i_old - E_j_old) / eta  # alpha_2迭代
                    alpha_j_new = self._cut(alpha_j_new_unc, L, H)  # 剪辑
                    # 如果变化不够大则跳过
                    if np.abs(alpha_j_new - alpha_j_old) < 1e-5:
                        continue

                    # 3. 得到 alpha_1_new
                    alpha_i_new = alpha_i_old + self.y[i] * self.y[j] * (alpha_j_old - alpha_j_new)

                    # 4. 更新 alpha_1, alpha_2
                    self.alpha[i] = alpha_i_new
                    self.alpha[j] = alpha_j_new

                    # 5. 更新 b
                    b_i_new = self.y[i] - self._f(i) + self.b
                    b_j_new = self.y[j] - self._f(j) + self.b
                    if 0 < alpha_i_new < self.C:
                        self.b = b_i_new
                    elif 0 < alpha_j_new < self.C:
                        self.b = b_j_new
                    else:
                        self.b = (b_i_new + b_j_new) / 2.0

                    # 6. 更新 E
                    self.E[i] = self._E(i)
                    self.E[j] = self._E(j)

                    # 7. 更新支持向量
                    self.support_vectors = np.where(self.alpha > 1e-3)[0]
                    self.support_vector_x = [features[i, :] for i in self.support_vectors]
                    self.support_vector_y = [labels[i] for i in self.support_vectors]
                    self.support_vector_alpha = [self.alpha[i] for i in self.support_vectors]

            # 如果所有的点都满足KKT条件，则中止
            if flag_kkt:
                break

        # 显示最终结果
        if show_train_process:
            utils.plot_decision_function(features, labels, self, self.support_vectors)
            utils.plt.pause(0.1)
            utils.plt.show()

    def _weight(self):
        self.w = [self._f(i) for i in range(self.n_samples)]
        return self.w

    def get_params(self):
        """
        输出模型参数
        :return:
        """
        return self.w, self.b

    def predict_proba(self, x):
        """
        软分类
        :param x:
        :return:
        """
        r = self.b
        for j in range(self.n_features):
            r += self.alpha[j] * self.y[j] * self.kernel(x, [self.X[j]])
        return utils.sigmoid(r)

    def predict(self, x):
        """
        硬分类
        :param x:
        :return:
        """
        proba = self.predict_proba(x)
        return (proba >= 0.5).astype(int)
