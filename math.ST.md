# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A path-norm toolkit for modern networks: consequences, promises and challenges.](http://arxiv.org/abs/2310.01225) | 本文介绍了适用于现代神经网络的路径范数工具包，可以包括具有偏差、跳跃连接和最大池化的通用DAG ReLU网络。这个工具包恢复或超越了已知的路径范数界限，并挑战了基于路径范数的一些具体承诺。 |

# 详细

[^1]: 一种适用于现代网络的路径范数工具包：影响、前景和挑战

    A path-norm toolkit for modern networks: consequences, promises and challenges. (arXiv:2310.01225v1 [stat.ML])

    [http://arxiv.org/abs/2310.01225](http://arxiv.org/abs/2310.01225)

    本文介绍了适用于现代神经网络的路径范数工具包，可以包括具有偏差、跳跃连接和最大池化的通用DAG ReLU网络。这个工具包恢复或超越了已知的路径范数界限，并挑战了基于路径范数的一些具体承诺。

    

    本文介绍了第一个完全能够包括具有偏差、跳跃连接和最大池化的通用DAG ReLU网络的路径范数工具包。这个工具包不仅适用于最广泛的基于路径范数的现代神经网络，还可以恢复或超越已知的此类范数的最尖锐界限。这些扩展的路径范数还享有路径范数的常规优点：计算简便、对网络的对称性具有不变性，在前馈网络上比操作符范数的乘积（另一种常用的复杂度度量）具有更好的锐度。工具包的多功能性和易于实施使我们能够通过数值评估在ImageNet上对ResNet的最尖锐界限来挑战基于路径范数的具体承诺。

    This work introduces the first toolkit around path-norms that is fully able to encompass general DAG ReLU networks with biases, skip connections and max pooling. This toolkit notably allows us to establish generalization bounds for real modern neural networks that are not only the most widely applicable path-norm based ones, but also recover or beat the sharpest known bounds of this type. These extended path-norms further enjoy the usual benefits of path-norms: ease of computation, invariance under the symmetries of the network, and improved sharpness on feedforward networks compared to the product of operators' norms, another complexity measure most commonly used.  The versatility of the toolkit and its ease of implementation allow us to challenge the concrete promises of path-norm-based generalization bounds, by numerically evaluating the sharpest known bounds for ResNets on ImageNet.
    

