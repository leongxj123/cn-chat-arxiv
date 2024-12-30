# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Energy stable neural network for gradient flow equations.](http://arxiv.org/abs/2309.10002) | 本文提出了能量稳定网络(EStable-Net)用于解决梯度流方程，该网络能够降低离散能量并生成高准确性和稳定性的预测。 |

# 详细

[^1]: 梯度流方程的能量稳定神经网络

    Energy stable neural network for gradient flow equations. (arXiv:2309.10002v1 [cs.LG])

    [http://arxiv.org/abs/2309.10002](http://arxiv.org/abs/2309.10002)

    本文提出了能量稳定网络(EStable-Net)用于解决梯度流方程，该网络能够降低离散能量并生成高准确性和稳定性的预测。

    

    本文提出了一种用于求解梯度流方程的能量稳定网络（EStable-Net）。我们的神经网络EStable-Net的解更新方案受到了梯度流方程基于辅助变量的等价形式的启发。EStable-Net能够在神经网络中降低离散能量，与梯度流方程的演化过程的性质保持一致。神经网络EStable-Net的架构包括几个能量衰减模块，每个模块的输出可以解释为梯度流方程演化过程的中间状态。这种设计提供了一个稳定、高效且可解释的网络结构。数值实验结果表明，我们的网络能够生成高准确性和稳定性的预测。

    In this paper, we propose an energy stable network (EStable-Net) for solving gradient flow equations. The solution update scheme in our neural network EStable-Net is inspired by a proposed auxiliary variable based equivalent form of the gradient flow equation. EStable-Net enables decreasing of a discrete energy along the neural network, which is consistent with the property in the evolution process of the gradient flow equation. The architecture of the neural network EStable-Net consists of a few energy decay blocks, and the output of each block can be interpreted as an intermediate state of the evolution process of the gradient flow equation. This design provides a stable, efficient and interpretable network structure. Numerical experimental results demonstrate that our network is able to generate high accuracy and stable predictions.
    

