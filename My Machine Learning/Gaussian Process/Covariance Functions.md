> Begin: 2024-04-08
>
> References: Automatic Model Construction with Gaussian Processes

# Expressing Structure with Kernels

## A few basic kernels

- Squared-exponential (SE, 平方指数核): $k(x, x') = \sigma_f^2 \exp (- \frac{(x - x')^2}{2 \ell^2})$
- Periodic (Per, 周期核): $k(x, x') = \sigma_f^2 \exp (-\frac{2}{\ell^2} \sin^2 (\pi \frac{x - x'}{p}))$​
- Linear (Lin, 线性核): $k(x, x') = \sigma_f^2 (x-c)(x'-c)$

<img src="C:\Users\12716\AppData\Roaming\Typora\typora-user-images\image-20240408003026631.png" alt="image-20240408003026631" style="zoom: 50%;" />







**平稳（Stationary）与非平稳（Non-stationary）** SE 和 Per 核是平稳的，这意味着它们的值仅取决于差值 $x - x'$。这意味着即使我们将所有 $x$ 值移动相同的量，观察到特定数据集的概率仍保持不变。相比之下，线性核 (Lin) 是非平稳的，这意味着如果在核参数保持固定的情况下移动数据，则相应的 GP 模型将产生不同的预测。





## Combining kernels

<img src="C:\Users\12716\AppData\Roaming\Typora\typora-user-images\image-20240408003555559.png" alt="image-20240408003555559" style="zoom:50%;" />

### Combining properties through multiplication

- **多项式回归**：通过将 $T$ 个线性核相乘，我们获得了 $T$ 次多项式的先验。图 2.2 的第一列显示了一个二次核。
- **局部周期函数**：在单变量数据中，将核乘以 SE 提供了一种**将全局结构转换为局部结构**的方法。例如，**Per对应于精确周期结构，而Per×SE对应于局部周期结构**，如图2.2第二列所示。
- **振幅增长函数**：乘以一个线性核，意味着被建模的函数的边际标准差从核参数 $c$ 给出的位置**线性增长**。图 2.2 的第三列和第四列显示了两个例子。

我们可以用这种方法将任意数量的核相乘，生成具有若干高级性质的核。例如，内核 $\mathrm{SE \times Lin \times Per }$​ 指定了局部周期性的线性增长幅度函数的先验。



### Building multi-dimensional models

对具有多个输入的函数建模的一种灵活方法是将在每个单独输入上定义的核相乘。例如，不同维度上SE核的乘积，每个核都有不同的长度尺度参数，称为 $\text{SE-ARD}$​ 核:



ARD代表自动相关性确定，之所以这样命名是因为估计长度尺度参数 $\ell_1, \ell_2, \dots, \ell_D$ 隐含地决定了每个维度的“相关性”。**具有较大长度尺度的输入维度意味着在被建模的函数中沿着这些维度的变化相对较小。**



## Modeling sums of functions

<img src="C:\Users\12716\AppData\Roaming\Typora\typora-user-images\image-20240408004804130.png" alt="image-20240408004804130" style="zoom:50%;" />



### Modeling noise

加性噪声可以建模为添加到信号中的未知的、快速变化的函数。通过添加本地内核（例如具有短长度的 SE），可以将该结构合并到 GP 模型中，如图 2.4 的第四列所示。当 SE 内核的长度变为零时，其极限是“白噪声”(WN) 内核。从带有 WN 核的 GP 中提取的函数值是从高斯随机变量中独立提取的。





### Posterior variance of additive component



为了说明加性内核如何产生可解释的模型，我们建立了混凝土强度的加性模型，作为七种不同成分（水泥、矿渣、粉煤灰、水、增塑剂、粗骨料和细骨料）用量的函数，和混凝土的年龄（Yeh，1998）。我们的简单模型是 8 个不同的一维函数的总和，每个函数仅取决于以下量之一：
$$
f(\mathbf X) = f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8 + \mathrm{noise}
$$
其中，$\mathrm{noise} \sim \mathcal N(0, \sigma_n^2)$ ，每个函数 $f_1, f_2, \dots, f_8$ 都使用带有 SE 内核的 GP 建模。这八个 SE 核加上一个白噪声核，形成一个单一的 GP 模型，其核有 9 个加性分量。

通过最大化数据的边际似然来学习核参数后，我们可以可视化模型每个组件的预测分布。



在这里，我们导出 GP 的所有加性分量的后验方差和协方差。这些公式允许人们绘制如图 2.7 所示的图。

<img src="C:\Users\12716\AppData\Roaming\Typora\typora-user-images\image-20240408010541812.png" alt="image-20240408010541812" style="zoom:50%;" />

首先，我们写下独立于 GP 先验得出的两个函数的联合先验分布及其总和。我们区分 $\boldsymbol f(\mathbf X)$（训练位置 $[\boldsymbol x_1, \boldsymbol x_2, \dots, \boldsymbol x_N]^{\top} := \mathbf X$ 处的函数值）和 $\boldsymbol f(\mathbf X^*)$（某些查询位置集 $[\boldsymbol x_1^*, \boldsymbol x_2^*, \dots, \boldsymbol x_M^*]^{\top} := \mathbf X^*$ 处的函数值）。形式上，如果 $\boldsymbol f_1$ 和 $\boldsymbol f_2$ 是先验独立的，并且 $\boldsymbol f_1 \sim \mathcal{GP}(\boldsymbol \mu_1, \mathbf K_1)$ 和 $\boldsymbol f_2 \sim \mathcal{GP} (\boldsymbol \mu_2, \mathbf K_2)$ ，那么
$$
\begin{bmatrix}
	\boldsymbol f_1(\mathbf X) \\
	\boldsymbol f_1(\mathbf X^*) \\
	\boldsymbol f_2(\mathbf X) \\
	\boldsymbol f_2(\mathbf X^*) \\
	\boldsymbol f_1(\mathbf X) + \boldsymbol f_2(\mathbf X) \\
	\boldsymbol f_1(\mathbf X^*) + \boldsymbol f_2(\mathbf X^*)
\end{bmatrix} \sim \mathcal N 
\left(
	\begin{bmatrix}
		\boldsymbol \mu_1 \\
		\boldsymbol \mu_1^* \\
		\boldsymbol \mu_2 \\
		\boldsymbol \mu_2^* \\
		\boldsymbol \mu_1 + \boldsymbol \mu_2 \\
		\boldsymbol \mu_1^* + \boldsymbol \mu_2^*
	\end{bmatrix} , 
	\begin{bmatrix}
		\mathbf K_1 & \mathbf K_1^* & 0 & 0 & \mathbf K_1 & \mathbf K_1^* \\
		{\mathbf K_1^*}^{\top} & {\mathbf K_1^{**}} & 0 & 0 & \mathbf K_1^* & \mathbf K_1^{**} \\
		0 & 0 & \mathbf K_2 & \mathbf K_2^* & \mathbf K_2 & \mathbf K_2^* \\
		0 & 0 & {\mathbf K_2^*}^{\top} & \mathbf K_2^{**} & \mathbf K_2^* & \mathbf K_2^{**} \\
		\mathbf K_1 & {\mathbf K_1^*}^{\top} & \mathbf K_2 & {\mathbf K_2^*}^{\top} & \mathbf K_1 + \mathbf K_2 & \mathbf K_1^* + \mathbf K_2^* \\
		{\mathbf K_1^*}^{\top} & \mathbf K_1^{**} & {\mathbf K_2^*}^{\top} & \mathbf K_2^{**} & {\mathbf K_1^*}^{\top} + {\mathbf K_2^*}^{\top} & \mathbf K_1^{**} + \mathbf K_2^{**} 
	\end{bmatrix}

\right)
$$
其中我们表示 Gram 矩阵，其第 i, j 个条目由 $\mathcal K(\boldsymbol x_i, \boldsymbol x_j)$ 给出
$$
\begin{aligned}
	\mathbf K_i &= \mathcal K_i(\mathbf X, \mathbf X) \\
	\mathbf K_i^* &= \mathcal K_i(\mathbf X, \mathbf X^*) \\
	\mathbf K_i^{**} &= \mathcal K_i(\mathbf X^*, \mathbf X^*)
\end{aligned}
$$
高斯条件式 [A.2]() 的公式可用于给出 GP 分布函数的条件分布，该函数以其与另一个 GP 分布函数的和为条件：
$$
\begin{aligned}
	\boldsymbol f_1(\mathbf X^*) | \boldsymbol f_1(\mathbf X) + \boldsymbol f_2(\mathbf X) \sim \mathcal N\bigg( & \boldsymbol \mu_1^* + {\mathbf K_1^*}^{\top}(\mathbf K_1 + \mathbf K_2)^{-1} \big[\boldsymbol f_1(\mathbf X) + \boldsymbol f_2(\mathbf X) -\boldsymbol \mu_1 - \boldsymbol \mu_2 \big], \\
    & \mathbf K_1^{**} - {\mathbf K_1^*}^{\top} (\mathbf K_1 + \mathbf K_2)^{-1} \mathbf K_1^* \bigg)
\end{aligned}
$$
这些公式表达了模型关于信号不同分量的后验不确定性，并对其他分量的可能配置进行了积分。为了将这些公式扩展到两个以上函数的和，可以将术语 $\mathbf K_1 + \mathbf K_2$ 简单地替换为 $\sum_i \mathbf K_i$​ 。



还可以计算任意两个函数的高度之间的后验协方差，以它们的总和为条件：
$$
\mathrm{Cov} \big[\boldsymbol f_1(\mathbf X^*), \boldsymbol f_2(\mathbf X^*) | \boldsymbol f(\mathbf X) \big] = - {\mathbf K_1^*}^{\top} (\mathbf K_1 + \mathbf K_2)^{-1} \mathbf K_2^*
$$
如果该量为负，则意味着该位置的两个函数中哪一个较高或较低存在模糊性。例如，图 2.8 显示了具体模型的所有非零分量之间的后验相关性。

<img src="C:\Users\12716\AppData\Roaming\Typora\typora-user-images\image-20240408112219863.png" alt="image-20240408112219863" style="zoom: 67%;" />

> 图 2.8：方程（2.12）中不同一维函数高度之间的后验相关性，其总和模拟了混凝土强度。红色表示高相关性，青色表示无相关性，蓝色表示负相关性。对角线图显示同一函数的不同评估之间的后验相关性。
>
> 相关性在与图 2.7 相同的输入范围内进行评估。未显示与 $f_6$ 和 $f_7$​​（精细）的相关性，因为它们的估计方差为零。
>
> 
>
> 这张图比较好理解，每个方格都表示某两个维度特征之间的协方差矩阵，即两个特征在不同取值下的后验相关性。



## Changepoints

**变化点内核** 给出了组合内核如何产生更结构化先验的示例，它可以表达不同类型结构之间的变化。变化点内核可以通过 sigmoidal 函数的加法和乘法来定义，例如 $\sigma(x) = 1 / 1 + \exp(−x)$​：
$$
\mathrm{CP}(k_1, k_2)(x, x') = \sigma(x)k_1(x, x')\sigma(x') + (1 - \sigma(x)) k_2(x, x') (1 - \sigma(x'))
$$
也可以写作：
$$
\mathrm{CP}(k_1, k_2) = k_1 \times \boldsymbol \sigma + k_2 \times \bar{\boldsymbol \sigma}
$$
其中，$\boldsymbol \sigma = \sigma(x)\sigma(x'), \bar{\boldsymbol \sigma} = (1 - \sigma(x))(1 - \sigma(x'))$ 。

这个复合内核表达了从一个内核到另一个内核的变化。 sigmoid 的参数决定了这种变化发生的位置和速度。图 2.9 显示了一些示例。

<img src="C:\Users\12716\AppData\Roaming\Typora\typora-user-images\image-20240408124847244.png" alt="image-20240408124847244" style="zoom: 80%;" />

==我们还可以构建一个函数模型，其结构仅在某个区间 **变化窗口** 内变化，方法是将 $\sigma(x)$ 替换为两个 sigmoid（一个递增，一个递减）的乘积。==











# Learning the Kernel

## Empirical Bayes for the kernel parameters

假设正在使用带有 RBF 核的 GP 进行一维回归，由于数据存在观测噪声，因此内核具有以下形式：
$$
\begin{aligned}
	\mathcal K(x_p, x_q) = \sigma_f^2 \exp\{- \frac{1}{2\ell^2}(x_p - x_q)^2\} + \sigma_y^2 \delta_{pq}
\end{aligned}
$$

- $\ell$ 函数变化的水平尺度；
- $\sigma_f^2$ 控制函数的垂直尺度；
- $\sigma_y^2$ 是噪声方差；

为了估计内核参数 $\boldsymbol \theta$（有时称为超参数），我们可以对离散值网格使用穷举搜索，以验证损失作为目标，但这可能会非常慢。（这是非概率方法（例如 SVM）用来调整内核的方法。）这里用**经验贝叶斯**方法，允许使用梯度方法和优化方法。

最大化**边际似然**：
$$
\begin{aligned}
	p(\boldsymbol y | \mathbf X, \boldsymbol \theta) &= \int p(\boldsymbol y | \boldsymbol f, \mathbf X)\, p(\boldsymbol f | \mathbf X, \boldsymbol \theta)\, \mathrm{d}\boldsymbol f \\
	p(\boldsymbol f | \mathbf X) &= \mathcal N(\boldsymbol f | \boldsymbol 0, \mathbf K) \\
	p(\boldsymbol y | \boldsymbol f) &= \prod_{i=1}^N \mathcal N(y_i | f_i, \sigma_y^2)
\end{aligned}
$$
那么 log marginal likelihood：
$$
\begin{aligned}
	\log p(\boldsymbol y | \mathbf X, \boldsymbol \theta) &= \log \mathcal N(\boldsymbol y | \boldsymbol 0, \mathbf K_\sigma) \\
	&= -\frac{1}{2}\boldsymbol y^{\top} \mathbf K_\sigma^{-1}\boldsymbol y -\frac{1}{2} \log |\mathbf K_\sigma| - \frac{N}{2} \log(2\pi)
\end{aligned}
$$
其中 $\mathbf K_\sigma = \mathbf K + \sigma_y^2 \mathbf I$ 对于 $\boldsymbol \theta$ 的依赖是隐含的。

- 第一项是数据拟合项；
- 第二项是模型复杂项；
- 第三项为常数

为了理解前两项之间的权衡，考虑 1d 中的 SE 内核，因为我们改变长度尺度 $\ell$ 并保持 $\sigma_y^2$ 固定。令 $J(\ell) = - \log p(\boldsymbol y | \mathbf X, \ell)$。对于短长度尺度，拟合效果会很好，因此 $\boldsymbol y^{\top} \mathbf K_{\sigma}^{-1} \boldsymbol y$ 会很小。然而，模型复杂度会很高：$\mathbf K$ 几乎是对角线，因为大多数点不会被视为“靠近”任何其他点，因此 $\log |\mathbf K_{\sigma}|$ 会很大。对于长尺度，拟合度会很差，但模型复杂度会很低：$\mathbf K$ 几乎全是 1，因此 $\log |\mathbf K_{\sigma}|$​​ 会很小。

<img src="C:\Users\12716\AppData\Roaming\Typora\typora-user-images\image-20240420153842705.png" alt="image-20240420153842705" style="zoom:50%;" />

数据拟合随着长度尺度单调减小，因为模型变得越来越不灵活。负复杂度惩罚随着长度尺度的增加而增加，因为模型随着长度尺度的增加而变得不那么复杂。边际似然本身在接近 1 的值处达到峰值。对于稍长于 1 的长度尺度，边际似然迅速下降，这是由于模型解释数据的能力较差，对比图2.5(c)。与图 2.5(b) 相比，对于较小的长度尺度，边际似然下降得更慢，对应于确实容纳数据的模型，但在远离基础函数的区域浪费了预测质量。

<img src="C:\Users\12716\AppData\Roaming\Typora\typora-user-images\image-20240420153948304.png" alt="image-20240420153948304" style="zoom:50%;" />

为了通过最大化边际似然来设置超参数，我们寻求边际似然梯度，即边际似然相对于超参数 $\boldsymbol \theta$ 的偏导数：
$$
\begin{aligned}
	\frac{\partial}{\partial \theta_j} \log p(\boldsymbol y | \mathbf X, \boldsymbol \theta) &= \frac{1}{2} \boldsymbol y^{\top} \mathbf K_{\sigma}^{-1} \frac{\partial \mathbf K_{\sigma}}{\partial \theta_j} \mathbf K_{\sigma}^{-1} \boldsymbol y - \frac{1}{2} \tr (\mathbf K_{\sigma}^{-1}\frac{\partial \mathbf K_{\sigma}}{\partial \theta_j}) \\
	&= \frac{1}{2} \tr((\boldsymbol \alpha \boldsymbol \alpha^{\top} - \mathbf K_{\sigma}^{-1}) \frac{\partial \mathbf K_{\sigma}}{\partial \theta_j})
\end{aligned}
$$
其中 $\boldsymbol \alpha = \mathbf K_{\sigma}^{-1} \boldsymbol y$ 。计算 $\mathbf K_{\sigma}^{-1}$ 需要 $O(N^3)$ 时间，然后每个超参数需要 $O(N^2)$ 时间来计算梯度。

$\frac{\partial \mathbf K_{\sigma}}{\partial \theta_j}$ 的形式取决于内核的形式以及我们对哪个参数进行导数。通常我们对超参数有约束，例如 $\sigma_y^2 \geq 0$。在这种情况下，我们可以定义 $θ = \log(\sigma^2_y )$，然后使用链式法则。















# Appendix Ⅰ: Gaussian Conditionals

If: 
$$
\boldsymbol y = \left[ \begin{array}{c}
					\boldsymbol y_A \\
					\boldsymbol y_B
				\end{array} \right] \sim \mathcal N \left(
                \left[ \begin{array}{c}
                	\boldsymbol \mu_A \\
                	\boldsymbol \mu_B
                \end{array} \right], 
                \left[ \begin{array}{c c}
                	\boldsymbol \Sigma_{AA} & \boldsymbol \Sigma_{AB} \\
                	\boldsymbol \Sigma_{BA} & \boldsymbol \Sigma_{BB}
                \end{array} \right]\right)
$$
then
$$
\boldsymbol y_A | \boldsymbol y_B \sim \mathcal N\big(\boldsymbol \mu_A + \boldsymbol\Sigma_{AB} \boldsymbol \Sigma_{BB}^{-1}(\boldsymbol y_B - \boldsymbol \mu_B), \boldsymbol \Sigma_{AA} - \boldsymbol \Sigma_{AB} \boldsymbol \Sigma_{BB}^{-1} \boldsymbol \Sigma_{BA}\big)
$$
