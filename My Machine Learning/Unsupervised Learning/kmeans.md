# K-means

### 一.算法原理

k均值聚类是有“目标”的，假设给定样本$D=\{x_1,x_2,...,x_m\}$，针对聚类所得簇划分$C=\{C_1,C_2,...,C_k\}$最小化如下的平方误差函数：   

$$
C^*=arg\min_{C}\sum_{i=1}^k\sum_{x\in C_i}\left|\left|x-\mu_i\right|\right|_2^2
$$

其中，$\mu_i=\frac{1}{|C_i|}\sum_{x\in C_i}x$是簇$C_i$的均值向量，将$m$个样本分到$k$个簇共有$\frac{1}{m!}\sum_{i=1}^m(-1)^iC_m^i(m-i)^k$种可能，显然这是一个NP-hard问题，我们所熟知的k均值算法其实是对这个问题的贪心搜索



### 二.算法流程
输入：样本集$D=\{x_1,x_2,...,x_m\}$；聚类次数$k$；终止误差$tol$  

过程   

​	（1）从$D$中随机选择$k$个样本作为初始均值向量${\mu_1,\mu_2,...,\mu_k}$   

​	（2）重复如下过程，直到终止条件   

​		（2.1）令$C_i=\{\},i=1,2,...,k$；  

​		（2.2）对$j=1,2,...,m$；   

​			（2.2.1）计算样本$x_j$与各均值向量$\mu_i,i=1,2,...,k$：$d_{ji}=||x_j-\mu_i||_2$；  

​			（2.2.2）根据距离最近的均值向量确定$x_j$的簇标记：$\lambda_j=arg\min_{i\in \{1,2,...,k\}}d_{ji}$；   

​			（2.2.3）将样本$x_j$划入相应的簇$C_{\lambda_j}=C_{\lambda_j}\bigcup\{x_j\}$；  

​		（2.3）令$\epsilon=0$，对$i=1,2,...,k$  

​			（2.3.1）计算新的均值向量：$\mu'_i=\frac{1}{C_i}\sum_{x\in C_i}x$，令$\epsilon=\epsilon+||u_i-u'_i||_2$

​			（2.3.2）并更新$u_i=u'_i$  

​		（2.4）如果$\epsilon<tol$，则终止循环  

​	输出：$C=\{C_1,C_2,...,C_k\}$	





# 层次聚类

### 算法流程
输入：样本集$D=\{x_1,x_2,...,x_m\}$;聚类簇距离度量函数$d$;聚类簇数$k$；  

过程：

​	（1） 对$j=1,2,...,m$,

​			 $G_j=\{x_j\}$  

​	（2） 对$i=1,2,..,m$, 对$j=1,2,...,m$
$$
M(i,j)=d(G_i,G_j)\\
M(j,i)=M(i,j)
$$

​	（3） 设置当前聚类簇个数$q=m$，当$q>k$时，循环执行以下过程  

​		（3.1）找出距离最近的两个聚类簇$G_{i^*}$和$G_{j^*}$（$i^*<j^*$）；   

​		（3.2）合并$G_{i^*}$和$G_{j^*}$：$G_{i^*}=G_{i^*}\bigcup G_{j^*}$；   

​		（3.3） 对$j=j^*+1,j^*+2,...,q$  

​			将聚类簇$G_j$重新编号为$G_{j-1}$

​		（3.4）删除距离矩阵$M$的第$j^*$行与第$j^*$列（注意，矩阵的行列都要减1）  

​		（3.5）对$j=1,2,...,q-1$
$$
M(i^*,j)=d(G_{i^*},G_j)\\
M(j,i^*)=M(i^*,j)
$$
​		（3.6）$q=q-1$  

输出$G=\{G_1,G_2,...,G_k\}$





# 谱聚类

### 基本原理

谱聚类将每个样本看作空间中的一个点，点与点之间的距离越近则权重越大，而谱聚类同样离不开“同类相近，异类相斥”的核心思想，所以需要量化“同类”的权重，使其尽可能的大，量化“异类”的权重，是其尽可能的小，所以谱聚类的核心内容两个：  

（1）如何表示点与点之间的相似度权重，这里通常可以使用RBF函数，对于任意两点$x_i,x_j$，它们之间的权重可以表示为$w_{ij}=exp\left(-\frac{\left|\left|x_i-x_j\right|\right|_2^2}{2\sigma^2}\right)$  

（2）如何对同类以及异类进行量化：   

​	（2.1）同类的权重可以简单由该类包含的样本来决定，对于类别样本点id的集合$A$，定义为$|A|:=A的大小$；   

​	（2.2）异类之间的权重可以定义为，$A$集合与$B$任意两点之间的权重和$W(A,B)=\sum_{i\in A,j\in B}w_{ij}$

离我们的优化目标还差一步了，那就是只需要一个单目标来表示同类权重尽可能大，异类权重尽可能小，将其相除即可，即最终的目标函数为：   

$$
    L(A_1,A_2,...,A_k)=\sum_{i=1}^k\frac{W(A_i,\bar{A_i})}{|A_i|}
$$

其中，$k$为类别数，即我们定义的超参数，$\bar{A_i}$为$A_i$的补集，显然聚类任务要求$A_1,A_2,...,A_k$之间互斥且完备 。



### 优化目标推导
我们的优化目标是从$A_1,A_2,...,A_k$的不同组合中选择使$L(A_1,A_2,...,A_k)$最小的，这显然是一个NP-Hard问题，借鉴降维的思想，我们假设这$k$聚类由$k$个指示向量来表示:$h_1,h2,...,h_k$,其中每个向量$h_j$是$n$维向量（$n$是样本量），并令：   

$$
h_{ij}=\left\{\begin{matrix}
0 & i\notin A_j\\ 
\frac{1}{\sqrt{|A_j|}} & i\in A_j
\end{matrix}\right. j=1,2,..,k;i=1,2,...,n
$$

所以，我们聚类指示向量之间是单位正交化的$h_i^Th_i=1,h_i^Th_j=0$，所以上面的组合问题就转换为了求指示向量的问题，让我们推导一下  

$$
\begin{equation}
\begin{split}
\frac{W(A_i,\bar{A_i})}{|A_i|}&=\frac{1}{2}\left(\frac{W(A_i,\bar{A_i})}{|A_i|}+\frac{W(\bar{A_i},A_i)}{|A_i|}\right)\\
&=\frac{1}{2}(\sum_{m\in A_i,n\notin A_i}\frac{w_{mn}}{|A_i|}+\sum_{m\notin A_i,n\in A_i}\frac{w_{mn}}{|A_i|})\\
&=\frac{1}{2}\sum_{m,n}w_{mn}(h_{mi}-h_{ni})^2\\
&=h_i^TLh_i
\end{split}
\end{equation}
$$

其中，$L$即是拉普拉斯矩阵，它由两部分构成:   

$$
L=D-W
$$

这里，$D=diag(d_1,d_2,...,d_n),d_i=\sum_{j=1}^nw_{ij}$，而$W_{ij}=w_{ij}$  

所以，整体的损失函数，可以表示为：  

$$
\begin{equation}
\begin{split}
L(A_1,A_2,...,A_k)&=\sum_{i=1}^k h_i^TLh_i\\
&=tr(H^TLH)\\
s.t.H^TH=I
\end{split}
\end{equation}
$$

所以，$H\in R^{n\times k}$就是对$L$对特征分解后，由最小的$k$个特征值对应的特征向量组成，当然实际求解出的$H$未必能满足我们的期望：  

$$
h_{ij}=\left\{\begin{matrix}
0 & i\notin A_j\\ 
\frac{1}{\sqrt{|A_j|}} & i\in A_j
\end{matrix}\right. j=1,2,..,k;i=1,2,...,n
$$

所以，通常还需要对其进行一次聚类，比如K-means。