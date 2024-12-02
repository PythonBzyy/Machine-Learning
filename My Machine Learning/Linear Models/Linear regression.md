# Linear Regression







## Probabilistic view

术语”线性回归“通常指以下形式的模型：
$$
p(y | \boldsymbol x, \boldsymbol \theta) = \mathcal N(y | \boldsymbol w^{\top} \boldsymbol x + w_0, \sigma^2)
$$
其中，$\boldsymbol \theta = (\boldsymbol w, w_0, \sigma^2)$ 表示模型的所有参数（统计学中常用 $\boldsymbol \beta, \beta_0$ 表示 $\boldsymbol w, w$）。

对应机器学习中常见的模型表达形式：
$$
y = \boldsymbol w^{\top} \boldsymbol x + \varepsilon, \quad \varepsilon \sim \mathcal N(0, \sigma^2)
$$
这里的权重参数 $\boldsymbol w$ 将偏置项 $w_0$​ 包括。



当 $\boldsymbol y \in \mathbb R^{J}$ 具有多个维度（多重线性回归，multivariate linear regression）：
$$
p(\boldsymbol y | \boldsymbol x, \mathbf W) = \prod_{j=1}^{J} \mathcal N(y_j | \boldsymbol w_j^{\top} \boldsymbol x, \sigma_j^2)
$$
更一般地，当直线无法很好拟合时，可以对输入特征应用一个非线性变换 $\boldsymbol \phi(\boldsymbol x)$ 
$$
p(y | \boldsymbol x, \boldsymbol \theta) = \mathcal N(y | \boldsymbol w^{\top} \boldsymbol \phi(\boldsymbol x), \sigma^2)
$$

> TODO
>
> 非线性变化，特征提取 $\boldsymbol \phi(\cdot)$





### Least squares estimation