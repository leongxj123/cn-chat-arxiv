# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Layer-wise Feedback Propagation.](http://arxiv.org/abs/2308.12053) | 本文提出了一种名为“层级反馈传播（LFP）”的新型神经网络预测器训练方法，通过利用可解释性细化与层级相关性传播（LRP）相结合，根据每个连接对任务的贡献分配奖励，该方法克服了传统梯度下降方法存在的问题。对于各种模型和数据集，LFP取得了与梯度下降相当的性能。 |

# 详细

[^1]: 层级反馈传播

    Layer-wise Feedback Propagation. (arXiv:2308.12053v1 [cs.LG])

    [http://arxiv.org/abs/2308.12053](http://arxiv.org/abs/2308.12053)

    本文提出了一种名为“层级反馈传播（LFP）”的新型神经网络预测器训练方法，通过利用可解释性细化与层级相关性传播（LRP）相结合，根据每个连接对任务的贡献分配奖励，该方法克服了传统梯度下降方法存在的问题。对于各种模型和数据集，LFP取得了与梯度下降相当的性能。

    

    本文提出了一种称为“层级反馈传播（LFP）”的新型神经网络预测器训练方法，该方法利用可解释性，具体而言是层级相关性传播（LRP），根据每个连接对解决给定任务的贡献独立分配奖励。这与传统的梯度下降方法不同，梯度下降方法是朝向估计的损失最小值更新参数。LFP在模型中传播奖励信号，而无需梯度计算。它增强接收到正反馈的结构，同时降低接收到负反馈的结构的影响。我们从理论和实证的角度证明了LFP的收敛性，并展示了它在各种模型和数据集上实现与梯度下降相当的性能。值得注意的是，LFP克服了梯度方法的某些局限性，例如对有意义的导数的依赖。我们进一步研究了LFP如何解决梯度方法相关问题的限制。

    In this paper, we present Layer-wise Feedback Propagation (LFP), a novel training approach for neural-network-like predictors that utilizes explainability, specifically Layer-wise Relevance Propagation(LRP), to assign rewards to individual connections based on their respective contributions to solving a given task. This differs from traditional gradient descent, which updates parameters towards anestimated loss minimum. LFP distributes a reward signal throughout the model without the need for gradient computations. It then strengthens structures that receive positive feedback while reducingthe influence of structures that receive negative feedback. We establish the convergence of LFP theoretically and empirically, and demonstrate its effectiveness in achieving comparable performance to gradient descent on various models and datasets. Notably, LFP overcomes certain limitations associated with gradient-based methods, such as reliance on meaningful derivatives. We further investigate how 
    

