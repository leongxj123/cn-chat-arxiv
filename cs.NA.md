# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Stochastic gradient descent for streaming linear and rectified linear systems with Massart noise](https://arxiv.org/abs/2403.01204) | 我们提出了一种针对具有Massart噪声的线性和ReLU回归问题的随机梯度下降方法，具有新颖的近乎线性收敛保证，首次在流式设置中为鲁棒ReLU回归提供了收敛保证，并展示了其相比于以前的方法有改进的收敛速率。 |

# 详细

[^1]: 具有Massart噪声的流式线性和修正线性系统的随机梯度下降

    Stochastic gradient descent for streaming linear and rectified linear systems with Massart noise

    [https://arxiv.org/abs/2403.01204](https://arxiv.org/abs/2403.01204)

    我们提出了一种针对具有Massart噪声的线性和ReLU回归问题的随机梯度下降方法，具有新颖的近乎线性收敛保证，首次在流式设置中为鲁棒ReLU回归提供了收敛保证，并展示了其相比于以前的方法有改进的收敛速率。

    

    我们提出了SGD-exp，一种用于线性和ReLU回归的随机梯度下降方法，在Massart噪声（对抗性半随机破坏模型）下，完全流式设置下。我们展示了SGD-exp对真实参数的近乎线性收敛保证，最高可达50%的Massart破坏率，在对称无忧破坏情况下，任意破坏率也有保证。这是流式设置中鲁棒ReLU回归的第一个收敛保证结果，它显示了相比于以前的鲁棒方法对于L1线性回归具有改进的收敛速率，这是由于选择了指数衰减步长，这在实践中已被证明是有效的。我们的分析基于离散随机过程的漂移分析，这本身也可能是有趣的。

    arXiv:2403.01204v1 Announce Type: new  Abstract: We propose SGD-exp, a stochastic gradient descent approach for linear and ReLU regressions under Massart noise (adversarial semi-random corruption model) for the fully streaming setting. We show novel nearly linear convergence guarantees of SGD-exp to the true parameter with up to $50\%$ Massart corruption rate, and with any corruption rate in the case of symmetric oblivious corruptions. This is the first convergence guarantee result for robust ReLU regression in the streaming setting, and it shows the improved convergence rate over previous robust methods for $L_1$ linear regression due to a choice of an exponentially decaying step size, known for its efficiency in practice. Our analysis is based on the drift analysis of a discrete stochastic process, which could also be interesting on its own.
    

