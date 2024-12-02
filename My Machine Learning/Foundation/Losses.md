## Cross-entropy Loss

### Cross-entropy

交叉熵的定义为：

分布 $q$ 相对于分布 $p$ 的交叉熵在给定集合上的定义如下：
$$
H(p, q) = - \mathbb{E}_{p}[\log q]
$$
其中，$\mathbb E[\cdot]$ 是关于分布的期望算子。

该定义可以使用 [Kullback-Leibler 散度](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) (KL 散度，$D_{\text{KL}}(p \parallel q)$) 来表述，即 $p$ 相对于 $q$ 的散度（也被称为 $p$ 关于 $q$ 的相对熵）。
$$
H(p, q) = H(p) + D_{\text{KL}}(p \parallel q)
$$
对于离散概率分布 $p$ 和 $q$ 在同样的支持下 $\mathcal X$ ，交叉熵为
$$
H(p, q) = - \sum_{x \in \mathcal X} p(x)\, \log q(x)
$$


### CrossEntropyLoss in PyTorch

https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss

在 pytorch 中的 `CrossEntropyLoss` 包含了 两部分，Softmax 和交叉熵计算。

该标准计算输入 logits 和目标之间的交叉熵损失。当训练C类的分类问题时，它很有用。如果提供，可选参数权 `weight` 应该是一维张量 为每个类别分配权重。 当您的训练集不平衡时，这特别有用。

该标准期望的目标应包含：

- 类别索引在 $[0, C)$ 中，其中 $C$ 是类别数量；未归一化（如 `reduction` set to `none`）的损失可以表示为：
    $$
    \ell(x, y)=L=\{l_1, \dots, l_N\}^{\top} \\
    l_n = - w_{y_n} \log \frac{\exp(x_{n, y_n})}{\sum_{c=1}^C \exp(x_{n, c})} \cdot 1\{y_n \neq \text{ignore\_index}\}
    $$
    其中，$x$ 是输入，$y$ 是目标，$w$ 是权重，$C$ 是类别数量，$N$ 是小批量维度。如果 `reduction` 不是 `none` ，那么
    $$
    \ell(x, y) = 
    \begin{cases}
    	\sum_{n=1}^N \frac{l_n}{\sum_{n=1}^N w_{y_n} \cdot 1\{y_n \neq \text{ignore\_index}\}}, & \text{if reduction='mean'}; \\
    	\sum_{n=1}^N l_n, & \text{if reduction='sum'}.
    \end{cases}
    $$
    请注意，这种情况相当于应用 [`LogSoftmax`](https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html#torch.nn.LogSoftmax) 在输入上，后跟 [`NLLLoss`](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss) 。





### BCELoss in PyTorch

Binary Cross Entropy loss，二元交叉熵损失。

未归一化的 BCE-loss 可以描述为：
$$
\ell(x, y) = L = \{l_1, \dots, l_N\}^{\top}, \\
l_n = - w_n [y_n \cdot \log x_n + (1 - y_n) \cdot \log(1 - x_n)]
$$
其中 $N$ 是批量大小。如果 `reduction` 不是 `none` ，那么
$$
\ell(x, y) = 
\begin{cases}
	\mathrm{mean}(L), & \text{if reduction = 'mean'}; \\
	\mathrm{sum}(L), & \text{if reduction = 'sum'}.
\end{cases}
$$
这用于测量重建的误差，例如自动编码器。请注意，目标 $y$ 应该是 0 到 1 之间的数字。