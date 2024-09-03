# Kalman Filter

> Begin: 2024-05-03
>
> Logs: 
>
> - version 1 begin.
> - 

## Linear Dynamical Systems (LDSs)

线性高斯状态空间模型 (LG-SSM)，也称为线性动态系统 (LDS)。这是 SSM 的一个特例，其中**转移函数和观测函数都是线性的，过程噪声和观测噪声都是高斯分布的**。

### Conditional independence properties

<img src="C:\Users\12716\AppData\Roaming\Typora\typora-user-images\image-20240503123208038.png" alt="image-20240503123208038" style="zoom:67%;" />

编码的隐藏状态是马尔可夫的假设，并且观察结果以隐藏状态为独立同分布。剩下的就是指定每个节点的条件概率分布的形式。



### Inference for Linear-Gaussian SSMs

当 SSM 中的推断，其中所有分布都是线性高斯分布，这称为 **线性高斯状态空间模型 (linear Gaussian state space model: LG-SSM)** 或 **线性动态系统 (linear dynamical system: LDS)** 。简单来说，它们具有以下形式：
$$
\begin{aligned}
	p(\boldsymbol z_t | \boldsymbol z_{t-1}, \boldsymbol u_t) &= \mathcal N(\boldsymbol z_t | \mathbf F_t \boldsymbol z_{t-1} + \mathbf B_t \boldsymbol u_t + \boldsymbol b_t, \mathbf Q_t) \\
	p(\boldsymbol y_t | \boldsymbol z_t, \boldsymbol u_t) &= \mathcal N(\boldsymbol y_t | \mathbf H_t \boldsymbol z_t + \mathbf D_t \boldsymbol u_t + \boldsymbol d_t, \mathbf R_t)
\end{aligned}
$$
其中，$\boldsymbol z_t \in \mathbb R^{N_z}$ 是隐状态，$\boldsymbol y_t \in \mathbb R^{N_y}$ 是观测，$\boldsymbol u_t \in \mathbb R^{N_u}$ 是输入。（允许参数随时间变化，可扩展）我们经常假设**过程噪声和观测噪声的均值 (即偏置 bias 或偏差 offset )** 为0，于是 $\boldsymbol b_t = \boldsymbol 0, \boldsymbol d_t = 0$ 。此外，经常没有输入，于是 $\mathbf B_t = \mathbf D_t = \boldsymbol 0$ 。因此模型简化为：
$$
\begin{aligned}
	\text{Transition probability:}&\quad p(\boldsymbol z_t | \boldsymbol z_{t-1}, \boldsymbol u_t) = \mathcal N(\boldsymbol z_t | \mathbf F_t \boldsymbol z_{t-1}, \mathbf Q_t) \\
	\text{Measurement probability:}&\quad p(\boldsymbol y_t | \boldsymbol z_t, \boldsymbol u_t) = \mathcal N(\boldsymbol y_t | \mathbf H_t \boldsymbol z_t, \mathbf R_t)
\end{aligned}
$$

> **NOTE**
>
> LG-SSM 只是高斯贝叶斯网络的特例，整个联合分布 $p(\boldsymbol y_{1:T}, \boldsymbol z_{1:T} | \boldsymbol u_{1:T})$ 是一个具有 $N_y \times N_z \times T$ 维度的大型的**多元高斯分布**。然而，它具有特殊的结构，使其在计算上易于使用，特别是**卡尔曼滤波器 (Kalman filter)**和**卡尔曼平滑器 (Kalman smoother)** ，它们可以在  $O(TN_z^3)$ 时间内执行**精确的滤波和平滑**。



---



## The Kalman Filter

卡尔曼滤波器 (KF) 是一种针对线性高斯状态空间模型进行精确贝叶斯滤波的算法。时间 $t$ 处的信念状态表示为 $p(\boldsymbol z_t | \boldsymbol y_{1:t}) = \mathcal N(\boldsymbol z_t | \hat{\boldsymbol \mu}_t, \hat{\boldsymbol \Sigma}_t)$​​ 。由于一切都是高斯分布，我们可以以封闭形式执行预测和更新步骤。

滤波问题可以定义为，给定截止时间 $t$ 的所有观测值 $\boldsymbol y_{1:t}$ ，求 $p(\boldsymbol z_t | \boldsymbol y_{1:t})$ 。在KF中，问题细分为：

1. **Prediction**
    $$
    \begin{aligned}
    	\underbrace{\color{royalblue}p(\boldsymbol z_t | \boldsymbol y_{1:t-1})}_{\color{royalblue}\text{prediction:}\ \mathcal N(\boldsymbol z_t | \bar{\boldsymbol \mu}_t, \bar{\boldsymbol \Sigma}_t)} = \int p(\boldsymbol z_t | \boldsymbol z_{t-1})\, \underbrace{\color{red}p(\boldsymbol z_{t-1} | \boldsymbol y_{1:t-1})}_{\color{red}\text{update:}\ \mathcal N(\boldsymbol z_{t-1} | \hat{\boldsymbol \mu}_{t-1}, \hat{\boldsymbol \Sigma}_{t-1})}\, \mathrm{d}\boldsymbol z_{t-1}
    \end{aligned}
    $$

2. **Update**
    $$
    \begin{aligned}
    	\underbrace{\color{red}p(\boldsymbol z_t | \boldsymbol y_{1:t})}_{\color{red}\text{update:}\ \mathcal N(\boldsymbol z_t | \hat{\boldsymbol \mu}_t, \hat{\boldsymbol \Sigma}_t)} = \frac{p(\boldsymbol y_t | \boldsymbol z_t)\, \overbrace{\color{royalblue}p(\boldsymbol z_t | \boldsymbol y_{1:t-1})}^{\color{royalblue}\text{prediction:}\ \mathcal N(\boldsymbol z_t | \bar{\boldsymbol \mu}_t, \bar{\boldsymbol \Sigma}_t)}}{\displaystyle\int p(\boldsymbol y_t | \boldsymbol z_t)\, p(\boldsymbol z_t | \boldsymbol y_{1:t-1})\, \mathrm{d} \boldsymbol z_t}
    \end{aligned}
    $$
    

prediction和update推导：
$$
\begin{aligned}
	{\color{royalblue}\textbf{prediction:}\ p(\boldsymbol z_t | \boldsymbol y_{1:t-1})} &= \int p(\boldsymbol z_t, \boldsymbol z_{t-1} | \boldsymbol y_{1:t-1})\, \mathrm{d}\boldsymbol z_{t-1} \\
	&= \int p(\boldsymbol z_t | \boldsymbol z_{t-1}, \cancel{\boldsymbol y_{1:t-1}})\, p(\boldsymbol z_{t-1} | \boldsymbol y_{1:t-1})\, \mathrm{d}\boldsymbol z_{t-1} \\
	&= \int p(\boldsymbol z_t | \boldsymbol z_{t-1})\, {\color{red}p(\boldsymbol z_{t-1} | \boldsymbol y_{1:t-1})}\, \mathrm{d}\boldsymbol z_{t-1}
\end{aligned}
$$
于是将预测问题转换成了滤波问题。
$$
\begin{aligned}
	{\color{red}\textbf{update:}\ p(\boldsymbol z_t | \boldsymbol y_{1:t})} &= p(\boldsymbol y_{1:t}, \boldsymbol z_t) / p(\boldsymbol y_{1:t}) \\
	&\propto p(\boldsymbol y_t, \boldsymbol y_{1:t-1}, \boldsymbol z_t) \\
	&= p(\boldsymbol y_t | \cancel{\boldsymbol y_{1:t-1}}, \boldsymbol z_t)\, p(\boldsymbol y_{1:t-1}, \boldsymbol z_t) \\
	&= p(\boldsymbol y_t | \boldsymbol z_t)\, p(\boldsymbol z_t | \boldsymbol y_{1:t-1})\, {p(\boldsymbol y_{1:t-1})} \\
	&\propto p(\boldsymbol y_t | \boldsymbol z_t)\, {\color{royalblue}p(\boldsymbol z_t | \boldsymbol y_{1:t-1})}
\end{aligned}
$$
出现了预测问题。于是得到时间 $t$ 上的递推：
$$
\begin{array}{c|c}
    \hline
    	t=1 & {\color{red}p(\boldsymbol z_1 | \boldsymbol y_1)} \\
    \hline
    	t=2 & {\color{royalblue}p(\boldsymbol z_2 | \boldsymbol y_1)} \\
    	& {\color{red}p(\boldsymbol z_2 | \boldsymbol y_{1:2})} \\    	
    \hline
    	t=3 & {\color{royalblue}p(\boldsymbol z_3 | \boldsymbol y_{1:2})} \\
    	& {\color{red}p(\boldsymbol z_3 | \boldsymbol y_{1:3})} \\
    \hline
    	\dots & \dots \\
    \hline
    	t & {\color{royalblue}p(\boldsymbol z_t | \boldsymbol y_{1:t-1})} \\
    	& {\color{red}p(\boldsymbol z_t | \boldsymbol y_{1:t})} \\
    \hline
    	\dots & \dots \\
    \hline
\end{array}
$$


### Predict step

隐藏状态的一步预测，也称为**时间更新步骤 (time update step)** ：
$$
\begin{aligned}
	p(\boldsymbol z_t | \boldsymbol y_{1:t-1}, \boldsymbol u_{1:t}) &= \mathcal N(\boldsymbol z_t | {\color{royalblue}\bar{\boldsymbol \mu}_t}, {\color{royalblue}\bar{\boldsymbol \Sigma}_t}) \\
	{\color{royalblue}\bar{\boldsymbol \mu}_t} &= \mathbf F_t {\color{red}\hat{\boldsymbol \mu}_{t-1}} {\color{gray}+ \mathbf B_t \boldsymbol u_t + \boldsymbol b_t} \\
	{\color{royalblue}\bar{\boldsymbol \Sigma}_t} &= \mathbf F_t {\color{red}\hat{\boldsymbol \Sigma}_{t-1}} \mathbf F_t^{\top} + \mathbf Q_t
\end{aligned}
$$

#### 详细推导









### Update step

更新步骤（也称为测量更新步骤）可以使用贝叶斯规则计算：
$$
\begin{aligned}
	p(\boldsymbol z_t | \boldsymbol y_{1:t}, \boldsymbol u_{1:t}) &= \mathcal N(\boldsymbol z_t | {\color{red}\hat{\boldsymbol \mu}_t}, {\color{red}\hat{\boldsymbol \Sigma}_t}) \\
	{\color{red}\hat{\boldsymbol \mu}_t} &= {\color{royalblue}\bar{\boldsymbol \mu}_t} + \underbrace{{\color{royalblue}\bar{\boldsymbol \Sigma}_t} \mathbf H_t (\mathbf H_t {\color{royalblue}\bar{\boldsymbol \Sigma}_t} \mathbf H_t^{\top} + \mathbf R_t)^{-1}}_{\color{darkorange}\mathbf K_t,\ \text{Kalman gain matrix}} (\boldsymbol y_t - \underbrace{\mathbf H_t {\color{royalblue}\bar{\boldsymbol \mu}_t} {\color{gray}+ \mathbf D_t \boldsymbol u_t + \boldsymbol d_t}}_{\hat{\boldsymbol y}_t}) \\
	{\color{red}\hat{\boldsymbol \Sigma}_t} &= (\mathbf I - \underbrace{{\color{royalblue}\bar{\boldsymbol \Sigma}_t} \mathbf H_t (\mathbf H_t {\color{royalblue}\bar{\boldsymbol \Sigma}_t} \mathbf H_t^{\top} + \mathbf R_t)^{-1}}_{\color{darkorange}\mathbf K_t,\ \text{Kalman gain matrix}} \mathbf H_t) {\color{royalblue}\bar{\boldsymbol \Sigma}_t}
\end{aligned}
$$
其中，${\color{darkorange}\mathbf K_t} = {\color{royalblue}\bar{\boldsymbol \Sigma}_t} \mathbf H_t (\mathbf H_t {\color{royalblue}\bar{\boldsymbol \Sigma}_t} \mathbf H_t^{\top} + \mathbf R_t)^{-1}$ 是 **卡尔曼增益矩阵 (Kalman gain matrix)** 。$\hat{\boldsymbol y}_t$ 是预期观测值，因此 $\boldsymbol e_t = \boldsymbol y_t - \hat{\boldsymbol y}_t$ 是 **残差误差 (residual error)** ，也称为 **创新项 (innovation term)** 。

将上式直观表示：
$$
\begin{aligned}
	{\color{red}\hat{\boldsymbol \mu}_t} &= {\color{royalblue}\bar{\boldsymbol \mu}_t} + {\color{darkorange}\mathbf K_t} \boldsymbol e_t \\
	{\color{red}\hat{\boldsymbol \Sigma}_t} &= (\mathbf I - {\color{darkorange}\mathbf K_t} \mathbf H_t) {\color{royalblue}\bar{\boldsymbol \Sigma}_t}
\end{aligned}
$$

- 潜在均值 ${\color{red}\hat{\boldsymbol \mu}_t}$ 的更新是预测潜在均值 ${\color{royalblue}\bar{\boldsymbol \mu}_t}$ 加上校正因子，即 $\mathbf K_t$ 乘误差信号 $\boldsymbol e_t$ ；
- 如果 $\mathbf H_t = \mathbf I$ ，则 $\mathbf K_t = {\color{royalblue}\bar{\boldsymbol \Sigma}_t} (\mathbf H_t {\color{royalblue}\bar{\boldsymbol \Sigma}_t} \mathbf H_t^{\top} + \mathbf R_t)^{-1}$ ，==TODO==





#### 详细推导

