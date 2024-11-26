# Gaussian Process Regression

## Introduction

我们有一个包含 $n$ 个观测值的数据集 $\mathcal D$ ，$\mathcal D = \{(\boldsymbol x_i, y_i) | i = 1, \dots n \}$ 。给定这些数据集，我们希望对训练集中未见过的新输入 $\boldsymbol x_*$ 进行预测。手头的问题是归纳的 ***inductive*** 。我们需要从有限的数据 $\mathcal D$ 转移到一个函数 $f$ 可以对所有可能的输入值进行预测。我们必须对底层函数的特征做出假设，否则，任何与训练数据一致的函数都同样有效。

两种常见的方法：

- 限制我们考虑的函数类，例如只考虑输入的线性函数；
- 为每个可能的函数提供先验概率 ***prior***，其中更高的概率被赋予我们认为更有可能的函数，例如因为它们比其他函数更平滑
- 第一种方法有一个明显的问题，即我们必须决定所考虑的函数类的丰富程度；如果我们使用基于某一类函数（例如线性函数）的模型，而目标函数不是由该类很好地建模的，那么预测就会很差。人们可能会试图增加函数类的灵活性，但这会带来**过拟合**的危险，即我们可以很好地拟合训练数据，但在进行测试预测时表现不佳。
- 第二种方法存在一个严重的问题，即肯定存在一个不可数的**无限个可能函数集**，我们如何在有限的时间内用这个集合进行计算？这时候我们选择**高斯过程**。高斯过程是高斯概率分布的泛化。概率分布描述的是标量或向量（对于多元分布）的随机变量，而随机过程则控制函数的性质。可以粗略地将函数视为一个很长的向量，向量中的每个元素都指定了特定输入 $\boldsymbol x$ 处的函数值 $f(\boldsymbol x)$​ 。如何用计算处理这些无限维对象的问题具有最令人愉快的解决方案：如果你只询问有限个点的函数属性，那么高斯过程中的推理将给出与忽略无限多个其他点相同的答案，就好像你将它们全部考虑在内一样！这些答案与你可能有的任何其他有限查询的答案一致。高斯过程框架的主要吸引力之一恰恰在于**它将复杂而一致的观点与计算可处理性结合在一起。**



![image-20241119221731124](C:\Users\12716\AppData\Roaming\Typora\typora-user-images\image-20241119221731124.png)



---



## Weight-space View

训练集：$\mathcal D =\{(\boldsymbol x_i, y_i) | i = 1, \dots, n \} \longrightarrow \mathcal D = \{\mathbf X, \boldsymbol y \}$ 。
其中，$\boldsymbol x$ 表示 $d$ 维输入向量（协变量），$y$ 表示输出（因变量）。

> [Bayesian linear regression]()

具有高斯噪声的标准的线性回归模型的贝叶斯分析：
$$
f(\boldsymbol x) = \boldsymbol x^{\top} \boldsymbol w, \quad y = f(\boldsymbol x) + \varepsilon
$$
假设观测值 $y$ 与函数值 $f(\boldsymbol x)$ 之间的差异在于附加的噪声，并且进一步假设噪声遵循独立同分布的高斯分布，均值为0，方差为 $\sigma^2_n$ ，
$$
\varepsilon \sim \mathcal N(0, \sigma_n^2)
$$


















## References

[A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/)

