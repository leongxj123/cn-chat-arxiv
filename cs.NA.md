# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the Convergence Rate of the Stochastic Gradient Descent (SGD) and application to a modified policy gradient for the Multi Armed Bandit](https://arxiv.org/abs/2402.06388) | 该论文证明了当学习速率按照逆时间衰减规则时，随机梯度下降（SGD）的收敛速度，并应用于修改的带有L2正则化的策略梯度多臂赌博机（MAB）的收敛性分析。 |
| [^2] | [Randomized Kaczmarz with geometrically smoothed momentum.](http://arxiv.org/abs/2401.09415) | 本文研究了向随机Kaczmarz算法中添加几何平滑动量的效果，并证明了关于最小二乘损失矩阵奇异向量方向上的期望误差。数值示例证明了结果的实用性，并提出了几个问题。 |

# 详细

[^1]: 关于随机梯度下降（SGD）的收敛速度及其在修改的多臂赌博机上的策略梯度应用

    On the Convergence Rate of the Stochastic Gradient Descent (SGD) and application to a modified policy gradient for the Multi Armed Bandit

    [https://arxiv.org/abs/2402.06388](https://arxiv.org/abs/2402.06388)

    该论文证明了当学习速率按照逆时间衰减规则时，随机梯度下降（SGD）的收敛速度，并应用于修改的带有L2正则化的策略梯度多臂赌博机（MAB）的收敛性分析。

    

    我们提出了一个自包含的证明，证明了当学习速率遵循逆时间衰减规则时，随机梯度下降（SGD）的收敛速度；接下来，我们将这些结果应用于带有L2正则化的修改的策略梯度多臂赌博机（MAB）的收敛性分析。

    We present a self-contained proof of the convergence rate of the Stochastic Gradient Descent (SGD) when the learning rate follows an inverse time decays schedule; we next apply the results to the convergence of a modified form of policy gradient Multi-Armed Bandit (MAB) with $L2$ regularization.
    
[^2]: 具有几何平滑动量的随机Kaczmarz方法

    Randomized Kaczmarz with geometrically smoothed momentum. (arXiv:2401.09415v1 [math.NA])

    [http://arxiv.org/abs/2401.09415](http://arxiv.org/abs/2401.09415)

    本文研究了向随机Kaczmarz算法中添加几何平滑动量的效果，并证明了关于最小二乘损失矩阵奇异向量方向上的期望误差。数值示例证明了结果的实用性，并提出了几个问题。

    

    本文研究了向随机Kaczmarz算法中添加几何平滑动量的效果，该算法是线性最小二乘损失函数上随机梯度下降的实例。我们证明了关于定义最小二乘损失的矩阵的奇异向量方向上期望误差的结果。我们给出了几个数值示例来说明我们结果的实用性，并提出了几个问题。

    This paper studies the effect of adding geometrically smoothed momentum to the randomized Kaczmarz algorithm, which is an instance of stochastic gradient descent on a linear least squares loss function. We prove a result about the expected error in the direction of singular vectors of the matrix defining the least squares loss. We present several numerical examples illustrating the utility of our result and pose several questions.
    

