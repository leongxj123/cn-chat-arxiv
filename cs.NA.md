# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CoLoRA: Continuous low-rank adaptation for reduced implicit neural modeling of parameterized partial differential equations](https://arxiv.org/abs/2402.14646) | CoLoRA通过连续低秩自适应提供了一种快速预测参数化偏微分方程解演变的简化神经网络建模方法 |

# 详细

[^1]: CoLoRA:用于参数化偏微分方程简化隐式神经建模的连续低秩自适应

    CoLoRA: Continuous low-rank adaptation for reduced implicit neural modeling of parameterized partial differential equations

    [https://arxiv.org/abs/2402.14646](https://arxiv.org/abs/2402.14646)

    CoLoRA通过连续低秩自适应提供了一种快速预测参数化偏微分方程解演变的简化神经网络建模方法

    

    该工作介绍了一种基于连续低秩自适应（CoLoRA）的简化模型，它预先训练神经网络适用于给定的偏微分方程，然后在时间上连续地调整低秩权重，以快速预测新物理参数和新初始条件下解场的演变。自适应可以是纯粹数据驱动的，也可以通过一个方程驱动的变分方法，提供Galerkin最优的逼近。由于CoLoRA在时间上局部逼近解场，权重的秩可以保持较小，这意味着只需要离线训练几条轨迹，因此CoLoRA非常适用于数据稀缺的情况。与传统方法相比，CoLoRA的预测速度快上几个数量级，其准确度和参数效率也比其他神经网络方法更高。

    arXiv:2402.14646v1 Announce Type: new  Abstract: This work introduces reduced models based on Continuous Low Rank Adaptation (CoLoRA) that pre-train neural networks for a given partial differential equation and then continuously adapt low-rank weights in time to rapidly predict the evolution of solution fields at new physics parameters and new initial conditions. The adaptation can be either purely data-driven or via an equation-driven variational approach that provides Galerkin-optimal approximations. Because CoLoRA approximates solution fields locally in time, the rank of the weights can be kept small, which means that only few training trajectories are required offline so that CoLoRA is well suited for data-scarce regimes. Predictions with CoLoRA are orders of magnitude faster than with classical methods and their accuracy and parameter efficiency is higher compared to other neural network approaches.
    

