# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fast gradient-free activation maximization for neurons in spiking neural networks.](http://arxiv.org/abs/2401.10748) | 本论文提出了一个快速无梯度激活最大化的方法，用于探索神经网络中神经元的特化。在一个人工脉冲神经网络上成功测试了这个方法，并提供了一个有效的设计框架。 |

# 详细

[^1]: 针对脉冲神经网络中神经元的快速无梯度激活最大化

    Fast gradient-free activation maximization for neurons in spiking neural networks. (arXiv:2401.10748v1 [cs.NE])

    [http://arxiv.org/abs/2401.10748](http://arxiv.org/abs/2401.10748)

    本论文提出了一个快速无梯度激活最大化的方法，用于探索神经网络中神经元的特化。在一个人工脉冲神经网络上成功测试了这个方法，并提供了一个有效的设计框架。

    

    神经网络（NNs），无论是生物还是人工的，都是由神经元构成的复杂系统，每个神经元都有自己的专业化。揭示这些专业化对于理解NNs的内部工作机制至关重要。对于一个生物系统，其对刺激的神经响应不是已知的（更不用说是可微分的函数），唯一的方式是建立一个反馈循环，将其暴露于刺激之中，其性质可以迭代地变化，以寻求最大响应的方向。为了在一个生物网络上测试这样的循环，首先需要学会快速和高效地运行它，以在最少的迭代次数内达到最有效的刺激（最大化某些神经元的激活）。我们提出了一个具有有效设计的框架，成功地在人工脉冲神经网络（SNN，模拟生物大脑NNs行为的模型）上测试了它。我们用于激活最大化（AM）的优化方法是基于快梯度方法的。

    Neural networks (NNs), both living and artificial, work due to being complex systems of neurons, each having its own specialization. Revealing these specializations is important for understanding NNs inner working mechanisms. The only way to do this for a living system, the neural response of which to a stimulus is not a known (let alone differentiable) function is to build a feedback loop of exposing it to stimuli, the properties of which can be iteratively varied aiming in the direction of maximal response. To test such a loop on a living network, one should first learn how to run it quickly and efficiently, reaching most effective stimuli (ones that maximize certain neurons activation) in least possible number of iterations. We present a framework with an effective design of such a loop, successfully testing it on an artificial spiking neural network (SNN, a model that mimics the behaviour of NNs in living brains). Our optimization method used for activation maximization (AM) was ba
    

