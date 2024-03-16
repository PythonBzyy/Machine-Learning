### 一.简介
支持向量机(svm)的想法与前面介绍的感知机模型类似，找一个超平面将正负样本分开，但svm的想法要更深入了一步，它要求正负样本中离超平面最近的点的距离要尽可能的大，所以svm模型建模可以分为两个子问题：  

（1）分的对：怎么能让超平面将正负样本分的开；  
（2）分的好：怎么能让距离超平面最近的点的距离尽可能的大。  

**对于第一个子问题**：将样本分开，与感知机模型一样，我们也可以定义模型目标函数为：  

$$
f(x)=sign(w^Tx+b)
$$
所以对每对样本$(x,y)$，只要满足$y\cdot (w^Tx+b)>0$，即表示模型将样本正确分开了  

**对于第二个子问题**：怎么能让离超平面最近的点的距离尽可能的大，对于这个问题，又可以拆解为两个小问题：  

（1）怎么度量距离？
（2）距离超平面最近的点如何定义？

距离的度量很简单，可以使用高中时代就知道的点到面的距离公式：  
$$
d=\frac{|w^Tx+b|}{||w||}
$$

距离超平面最近的点，我们可以强制定义它为满足$|w^Tx+b|=1$的点（注意，正负样本都要满足），为什么可以这样定义呢？我们可以反过来看，一个训练好的模型可以满足：（1）要使得正负样本距离超平面最近的点的距离都尽可能大，那么这个距离必然要相等，（2）参数$w,b$可以等比例的变化，而不会影响到模型自身，所以$|w^Tx+b|=1$自然也可以满足，所以这时最近的点的距离可以表示为：

$$
d^*=\frac{1}{||w||}
$$

同时第一个子问题的条件要调整为$y\cdot(w^Tx+b)\geq1$，而$\max d^*$可以等价的表示为$\min \frac{1}{2}w^Tw$，所以svm模型的求解可以表述为如下优化问题：  

$$
\min_{w,b} \frac{1}{2}w^Tw \\
s.t.y_i(w^Tx_i+b)\geq 1,i=1,2,...,N
$$




### 二.原优化问题的对偶问题

对于上面优化问题的求解往往转化为对其对偶问题的求解，首先，构造其拉格朗日函数：  

$$
L(w,b,\alpha)=\frac{1}{2}w^Tw+\sum_{i=1}^N \alpha_i(1-y_i(w^Tx_i+b)),\alpha=[\alpha_1,\alpha_2,...,\alpha_N]
$$

这时，原优化问题（设为$P$）就等价于：  

$$
\min_{w,b}\max_{\alpha}L(w,b,\alpha)\\
s.t.\alpha_i\geq 0,i=1,2,...,N
$$

这里简单说明一下为什么等价，首先看里面$\max$那一层
$$\max_{\alpha}L(w,b,\alpha)\\
s.t.\alpha_i\geq 0,i=1,2,...,N$$  

对每个样本都有约束条件$1-y_i(w^Tx_i+b)$，如果满足约束，即$\leq 0$，必有$\alpha_i(1-y_i(w^Tx_i+b))=0$，如果不满足，必有$\alpha_i(1-y_i(w^Tx_i+b))\rightarrow 正无穷$，所以，（1）如果所有样本均满足约束条件(即$w,b$在可行域内时)，原问题与上面的$\min\max$问题等价，（2）如果有任意一个样本不满足约束，这时上面$\max$问题的函数取值为正无穷，外层再对其求$\min$会约束其只能在可行域内求最小值，所以两问题是等价的，简单手绘演示一下（两个问题的最优解都是红点标记）：  
![avatar](./source/06_原问题与其min-max问题.png) 

假设对于问题$P$我们求得了最优解$w^*,b^*,\alpha^*$，则必有$L(w^*,b^*,\alpha^*)=L(w^*,b^*,0)$，所以有：  

$$
\sum_{i=1}^N\alpha_i^*(1-y_i({w^*}^Tx_i+b^*))=0(条件1)
$$

而最优解自然也满足原始的约束条件，即：  

$$
1-y_i({w^*}^Tx_i+b)\leq0,i=1,2,...,N(条件2)\\
\alpha_i^*\geq0,i=1,2,...,N(条件3)\\
$$

由条件1，2，3，我们可以得出更强地约束条件：  

$$
\alpha_i^*(1-y_i({w^*}^Tx_i+b^*))=0,i=1,2,...,N(条件4)
$$

证明也很简单，由条件2,3可以知道，$\forall i,\alpha_i^*(1-y_i({w^*}^Tx_i+b^*))\leq0$都成立，要使条件1成立，则只能$\alpha_i^*(1-y_i({w^*}^Tx_i+b^*))=0,i=1,2,...,N$。


进一步的，可以推导出这样的关系：  

$$
\forall \alpha_i^*>0\Rightarrow 1-y_i({w^*}^Tx_i+b^*)=0(关系1)\\
\forall 1-y_i({w^*}^Tx_i+b^*)<0\Rightarrow \alpha_i^*=0(关系2)
$$

所以条件4有个很形象的称呼：**互补松弛条件**，而对于满足关系1的样本，也有个称呼，叫**支持向量**   

好的，我们继续看svm的对偶问题（设为$Q$）的定义：  
$$
\max_{\alpha}\min_{w,b}L(w,b,\alpha)\\
s.t.\alpha_i\geq 0,i=1,2,...,N
$$

很幸运，svm的对偶问题$\max\min$与原问题$\min\max$等价（等价是指两个问题的最优值、最优解$w,b,\alpha$均相等，**具体证明需要用到原问题为凸以及slater条件，可以参看《凸优化》**），先看里层的$\min_{w,b} L(w,b,\alpha)，$由于$L(w,b,\alpha)$是关于$w,b$的凸函数，所以对偶问题的最优解必然满足：$L(w,b,\alpha)$关于$w,b$的偏导为0，即：  

$$
w=\sum_{i=1}^N\alpha_iy_ix_i(条件5)\\
0=\sum_{i=1}^N\alpha_iy_i(条件6)
$$

消去$w,b$，可得对偶问题关于$\alpha$的表达式：  

$$
\max_{\alpha} \sum_{i=1}^N\alpha_i-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jx_i^Tx_j\\
s.t.\sum_{i=1}^N\alpha_iy_i=0,\\
\alpha_i\geq0,i=1,2,...,N
$$

显然，等价于如下优化问题（设为$Q^*$）：  

$$
\min_{\alpha} \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jx_i^Tx_j-\sum_{i=1}^N\alpha_i\\
s.t.\sum_{i=1}^N\alpha_iy_i=0,\\
\alpha_i\geq0,i=1,2,...,N
$$


该问题是关于$\alpha$的凸二次规划(QP)问题，可以通过一些优化计算包(比如cvxopt)直接求解最优的$\alpha^*$，再由条件5，可知：

$$
w^*=\sum_{i=1}^N\alpha_i^*y_ix_i
$$


而关于$b^*$，我们可以巧妙求解：找一个样本点$(x_i,y_i)$，它满足对应的$\alpha_i^*>0$（即支持向量），利用关系1，可知$1-y_i({w^*}^Tx_i+b^*)=0$，所以：$b^*=y_i-{w^*}^Tx_i$   

这里，条件2,3,4,5,6即是**KKT条件**，而且对于该优化问题，**KKT条件**还是最优解的充分条件，即满足KKT条件的解就是最优解。





### 三.SMO求解对偶问题最优解
关于对偶问题($Q^*$)可以使用软件包暴力求解，而且一定能得到最优解，但它的复杂度有点高：（1）变量数与样本数相同，每个变量$\alpha_i$对应样本$(x_i,y_i)$；（2）约束条件数也与样本数相同；而序列最小最优化化(sequential minimal optimization,SMO)算法是求解SVM对偶问题的一种启发式算法，它的思路是：**每次只选择一个变量优化，而固定住其他变量**，比如选择$\alpha_1$进行优化，而固定住$\alpha_i,i=2,3,...,N$，但由于我们的问题中有一个约束：$\sum_i^N\alpha_iy_i=0$，需要另外选择一个$\alpha_2$来配合$\alpha_1$做改变，当两者中任何一个变量确定后，另外一个也就随之确定了，比如确定$\alpha_2$后：  

$$
\alpha_1=-y_i\sum_{i=2}^N\alpha_iy_i(关系3)
$$

**选择两个变量后，如果优化？**  
我们在选择好两个变量后，如何进行优化呢？比如选择的$\alpha_1,\alpha_2$，由于剩余的$\alpha_3,\alpha_4,...,\alpha_N$都视作常量，在$Q^*$中可以忽略，重新整理一下此时的$Q^*$：  

$$
\min_{\alpha_1,\alpha_2}\frac{1}{2}\alpha_1^2 x_1^Tx_1+\frac{1}{2}\alpha_2^2x_2^Tx_2+\alpha_1\alpha_2y_1y_2x_1^Tx_2+\frac{1}{2}\alpha_1y_1x_1^T\sum_{i=3}^N\alpha_iy_ix_i+\frac{1}{2}\alpha_2y_2x_2^T\sum_{i=3}^N\alpha_iy_ix_i-\alpha_1-\alpha_2\\
s.t.\alpha_1y_1+\alpha_2y_2=-\sum_{i=3}^N\alpha_iy_i=\eta\\
\alpha_1\geq0,\alpha_2\geq0
$$

这里求解其实就很简单了，将关系3带入，消掉$\alpha_1$后可以发现，优化的目标函数其实是关于$\alpha_2$的二次函数（且开口朝上）：  

$$
\min_{\alpha_2}\frac{1}{2}(x_1-x_2)^T(x_1-x_2)\alpha_2^2+(-y_2\eta x_1^Tx_1+y_1\eta x_1^Tx_2+\frac{1}{2}y_2x_2^T\gamma-\frac{1}{2}y_2x_1^T\gamma-1+y_1y_2)\alpha_2\\
s.t.\alpha_2\geq0,y_1(\eta-\alpha_2y_2)\geq0
$$

这里，$\eta=-\sum_{i=3}^N\alpha_iy_i,\gamma=\sum_{i=3}^N\alpha_iy_ix_i$  

所以该问题无约束的最优解为：  

$$
\alpha_2^{unc}=-\frac{-y_2\eta x_1^Tx_1+y_1\eta x_1^Tx_2+\frac{1}{2}y_2x_2^T\gamma-\frac{1}{2}y_2x_1^T\gamma-1+y_1y_2}{(x_1-x_2)^T(x_1-x_2)}(公式1)
$$




接下来，我们对上面的表达式做一些优化，大家注意每次迭代时，$\gamma,\eta$都有大量的重复计算（每次仅修改了$\alpha$的两个变量，剩余部分其实无需重复计算），而且对于$\alpha_1,\alpha_2$的更新也没有有效利用它上一阶段的取值（记作$\alpha_1^{old},\alpha_2^{old}$）：  

我们记：  
$$
g(x)=\sum_{i=1}^N\alpha_iy_ix_i^Tx+b
$$
记：  
$$
E_i=g(x_i)-y_i
$$
这里$g(x)$表示模型对$x$的预测值，$E_i$表示预测值与真实值之差，于是我们有：  

$$
x_1^T\gamma=g(x_1)-\alpha_1^{old}y_1x_1^Tx_1-\alpha_2^{old}y_2x_2^Tx_1-b^{old}\\
x_2^T\gamma=g(x_2)-\alpha_1^{old}y_1x_1^Tx_2-\alpha_2^{old}y_2x_2^Tx_2-b^{old}
$$

另外：  

$$
\eta=\alpha_1^{old}y_1+\alpha_2^{old}y_2
$$
带入公式1，可得：  

$$
\alpha_2^{unc}=\alpha_2^{old}+\frac{y_2(E_1^{old}-E_2^{old})}{\beta}
$$

这里$\beta=(x_1-x_2)^T(x_1-x_2)$，到这一步，可以发现计算量大大降低，因为$E_1^{old},E_2^{old}$可先缓存到内存中，但别忘了$\alpha_2$还有约束条件$\alpha_2\geq0,y_1(\eta-\alpha_2y_2)\geq0$，所以需要进一步对它的最优解分情况讨论：  

当$y_1y_2=1$时，
$$
\alpha_2^{new}=\left\{\begin{matrix}
0 & \alpha_2^{unc}<0\\ 
\alpha_2^{unc} & 0\leq\alpha_2^{unc}\leq \alpha_1^{old}+\alpha_2^{old}\\ 
\alpha_1^{old}+\alpha_2^{old} & \alpha_2^{unc}>\alpha_1^{old}+\alpha_2^{old}
\end{matrix}\right.
$$

当$y_1y_2=-1$时，  

$$
\alpha_2^{new}=\left\{\begin{matrix}
\alpha_2^{unc} & \alpha_2^{unc}\geq max\{0,\alpha_2^{old}-\alpha_1^{old}\}\\ 
max\{0,\alpha_2^{old}-\alpha_1^{old}\} & \alpha_2^{unc}< max\{0, \alpha_2^{old}-\alpha_1^{old}\}
\end{matrix}\right.
$$

到这儿，我们可以发现，SMO算法可以极大的方便$Q^*$的求解，而且是以解析解方式，得到$\alpha_2^{new}$后，由于$\alpha_1^{new}y_1+\alpha_2^{new}y_2=\alpha_1^{old}y_1+\alpha_2^{old}y_2$，可得到$\alpha_1^{new}$的更新公式：  
$$
\alpha_1^{new}=\alpha_1^{old}+y_1y_2(\alpha_2^{old}-\alpha_2^{new})
$$

最后，得到$w$的更新公式：  

$$
w^{new}=w^{old}+(\alpha_1^{new}-\alpha_1^{old})y_1x_1+(\alpha_2^{new}-\alpha_2^{old})y_2x_2
$$




**对$b$和$E$的更新**

而对于$b$的更新同样借助于$\alpha_1,\alpha_2$更新，在更新后，倾向于$\alpha_1^{new}>0,\alpha_2^{new}>0$，还记得前面的互补松弛条件吧（条件4），即对于$\alpha_i>0$的情况，必然要有$1-y_i(w^Tx_i+b)=0$成立，即$w^Tx_i+b=y_i$，所以对$(x_1,y_1),(x_2,y_2)$有如下关系：  

$$
{w^{new}}^Tx_1+b=y_1(关系4)\\
{w^{new}}^Tx_2+b=y_2(关系5)\\
$$
对关系4和关系5可以分别计算出$b_1^{new}=y_1-{w^{new}}^Tx_1,b_2^{new}=y_2-{w^{new}}^Tx_2$，对$b$的更新，可以取两者的均值：  

$$
b^{new}=\frac{b_1^{new}+b_2^{new}}{2}
$$

接下来，对于$E_1,E_2$的更新就很自然了：  

$$
E_1^{new}={w^{new}}^Tx_1+b^{new}-y_1\\
E_2^{new}={w^{new}}^Tx_2+b^{new}-y_2
$$
那接下来还有一个问题，那就是$\alpha_1,\alpha_2$如何选择的问题  

**如何选择两个优化变量？**  

这可以启发式选择，分为两步：第一步是如何选择$\alpha_1$，第二步是在选定$\alpha_1$时，如何选择一个不错的$\alpha_2$：  

**$\alpha_1$的选择**   

选择$\alpha_1$同感知机模型类似，选择一个不满足KKT条件的点$(x_i,y_i)$，即不满足如下两种情况之一的点：  

$$
\left\{\begin{matrix}
\alpha_i=0\Leftrightarrow  y_i(w^Tx_i+b)\geq1\\ 
\alpha_i>0\Leftrightarrow  y_i(w^Tx_i+b)=1
\end{matrix}\right.
$$

**$\alpha_2$的选择**  

对$\alpha_2$的选择倾向于选择使其变化尽可能大的点，由前面的更新公式可知是使$\mid E_1^{old}-E_2^{old}\mid$最大的点，所以选择的两个点$(x_1,y_1)$和$(x_2,y_2)$会更倾向于选择异类的点





### 四.软间隔

那怕仅含有一个异常点，对硬间隔支持向量机的训练影响就很大，我们希望它能具有一定的包容能力，容忍哪些放错的点，但又不能容忍过度，我们可以引入变量$\xi$和一个超参$C$来进行控制，原始的优化问题更新为如下：  

$$
\min_{w,b,\xi} \frac{1}{2}w^Tw + C\sum_{i=1}^N\xi_i\\
s.t.y_i(w^Tx_i+b)\geq 1-\xi_i,i=1,2,...,N\\
\xi_i\geq0,i=1,2,...,N
$$

这里$C$若越大，包容能力就越小，当取值很大时，就等价于硬间隔支持向量机，而$\xi$使得支持向量的间隔可以调整，不必像硬间隔那样，严格等于1

#### Lagrange函数
关于原问题的Lagrange函数：  

$$
L(w,b,\xi,\alpha,\mu)=\frac{1}{2}w^Tw+C\sum_{i=1}^N\xi_i+\sum_{i=1}^N\alpha_i(1-\xi_i-y_i(w^Tx_i+b))-\sum_{i=1}^N\mu_i\xi_i\\
s.t.\mu_i\geq 0,\alpha_i\geq0,i=1,2,...,N
$$

#### 对偶问题

对偶问题的求解过程与硬间隔类似：  

$$
\min_{\alpha} \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jx_i^Tx_j-\sum_{i=1}^N\alpha_i\\
s.t.\sum_{i=1}^N\alpha_iy_i=0,\\
0\leq\alpha_i\leq C,i=1,2,...,N
$$
可以发现与硬间隔的不同是$\alpha$加了一个上界的约束$C$ 

#### KKT条件

这里就直接写KKT条件看原优化变量与拉格朗日乘子之间的关系：  

$$
\frac{\partial L}{\partial w}=0\Rightarrow w^*=\sum_{i=1}^N\alpha_i^*y_ix_i(关系1)\\
\frac{\partial L}{\partial b}=0\Rightarrow \alpha_i^*y_i=0(关系2)\\
\frac{\partial L}{\partial \xi}=0\Rightarrow C-\alpha_i^*-\mu_i^*=0(关系3)\\
\alpha_i^*(1-\xi_i^*-y_i({w^*}^Tx_i+b^*))=0(关系4)\\
\mu_i^*\xi_i^*=0(关系5)\\
y_i({w^*}^Tx_i+b^*)-1-\xi_i^*\geq0(关系6)\\
\xi_i^*\geq0(关系7)\\
\alpha_i^*\geq0(关系8)\\
\mu_i^*\geq0(关系9)\\
$$

#### $w^*,b^*$的求解

由KKT条件中的关系1，我们可以知道：  

$$
w^*=\sum_{i=1}^N\alpha_i^*y_ix_i
$$

对于$b^*$的求解，我们可以取某点，其$0<\alpha_k^*<C$，由关系3,4,5可以推得到：${w^*}^Tx_k+b^*=y_k$，所以：  

$$
b^*=y_k-{w^*}^Tx_k
$$

#### SMO求$\alpha^*$  

好了，最终模型得求解落到了对$\alpha^*$得求解上，求解过程与硬间隔一样，无非就是就是对$\alpha$多加了一个约束：$\alpha_i^*<=C$，具体而言需要对$\alpha_2^{new}$的求解进行更新：  


当$y_1\neq y_2$时：  

$$
L=max(0,\alpha_2^{old}-\alpha_1^{old})\\
H=min(C,C+\alpha_2^{old}-\alpha_1^{old})
$$

当$y_1=y_2$时：  

$$
L=max(0,\alpha_2^{old}+\alpha_1^{old}-C)\\
H=min(C,\alpha_2^{old}+\alpha_1^{old})
$$

更新公式：  

$$
\alpha_2^{new}=\left\{\begin{matrix}
H & \alpha_2^{unc}> H\\ 
\alpha_2^{unc} &  L \leq \alpha_2^{unc} \leq H\\
L & \alpha_2^{unc}<L
\end{matrix}\right.
$$




### Kernel Trick

核技巧简单来说分为两步：  
（1）将低维非线性可分数据$x$，通过一个非线性映射函数$\phi$，映射到一个新空间（高维度甚至是无限维空间）；  
（2）对新空间的数据$\phi(x)$训练线性分类器  

比如如下的情况：  

原始数据需要使用一个椭圆才能分隔开，但对原始数据施加一个非线性变换$\phi:(x_1,x_2)->(x_1^2,x_2^2)$变换后，在新空间中就可以线性分隔了



#### 利用核技巧后的SVM
所以，如果对原始数据施加一个映射，此时软间隔SVM的对偶问题为：  

$$
\min_{\alpha} \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j\phi(x_i)^T\phi(x_j)-\sum_{i=1}^N\alpha_i\\
s.t.\sum_{i=1}^N\alpha_iy_i=0,\\
0\leq\alpha_i\leq C,i=1,2,...,N
$$
求解得最优$\alpha_i^*$后，SVM模型为：  

$$
f(x)=sign(\sum_{i=1}^N\alpha_iy_i\phi(x_i)^T\phi(x)+b^*)
$$

### 核函数   
观察一下上面公式，我们的目的其实是求解$\phi(x_i)^T\phi(x_j)$，有没有一种函数让$(x_i,x_j)$只在原始空间做计算就达到$\phi(x_i)^T\phi(x_j)$的效果呢？有的，那就是核函数，即：  

$$
K(x_i,x_j)=\phi(x_i)^T\phi(x_j)
$$

#### 怎样的函数才能做核函数？
要成为核函数必须满足如下两点条件：  

（1）对称性：$K(x_i,x_j)=K(x_j,x_i)$  

（2）正定性：对任意的$x_i,i=1,2,..,m$，$K(x,z)$对应的Gramm矩阵：  

$$
K=[K(x_i,x_j)]_{m\times m}
$$
是半正定矩阵，这里的$x_i\in$可行域，并不要求一定要属于样本集  

#### 常见的核函数有哪些？

目前用的比较多的核函数有如下一些：  

（1）多项式核函数：  

$$
K(x,z)=(x^Tz+1)^p
$$

（2）高斯核函数：  

$$
K(x,z)=exp(-\frac{\mid\mid x-z\mid\mid^2}{2\sigma^2})
$$

显然，线性可分SVM中使用的是$K(x,z)=x^Tz$也是核函数

#### 利用核函数后的SVM

利用核函数后，软间隔SVM的对偶问题为：  

$$
\min_{\alpha} \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jK(x_i,x_j)-\sum_{i=1}^N\alpha_i\\
s.t.\sum_{i=1}^N\alpha_iy_i=0,\\
0\leq\alpha_i\leq C,i=1,2,...,N
$$
求解得最优$\alpha_i^*$后，SVM模型为：  

$$
f(x)=sign(\sum_{i=1}^N\alpha_iy_iK(x,x_i)+b^*)
$$
