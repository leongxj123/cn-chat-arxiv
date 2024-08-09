# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Drift Control of High-Dimensional RBM: A Computational Method Based on Neural Networks.](http://arxiv.org/abs/2309.11651) | 该论文提出了一种基于神经网络的计算方法，用于漂移控制高维RBMs。通过深度神经网络技术，该方法在测试问题上达到了较高的准确性。 |

# 详细

[^1]: 高维RBM的漂移控制：基于神经网络的计算方法

    Drift Control of High-Dimensional RBM: A Computational Method Based on Neural Networks. (arXiv:2309.11651v1 [eess.SY])

    [http://arxiv.org/abs/2309.11651](http://arxiv.org/abs/2309.11651)

    该论文提出了一种基于神经网络的计算方法，用于漂移控制高维RBMs。通过深度神经网络技术，该方法在测试问题上达到了较高的准确性。

    

    受排队理论应用的启发，我们考虑了一个状态空间为d维正半轴的随机控制问题。控制过程Z按照一个反射布朗运动演化，其协方差矩阵是外生指定的，反射方向是从正半轴边界表面反射。系统管理员根据Z的历史选择每个时间点t上的漂移向量θ(t)，而时间点t上的成本率取决于Z(t)和θ(t)。在我们的初始问题表述中，目标是在无限规划时间范围内最小化期望贴现成本，之后我们处理相应的人均控制问题。借鉴韩海亮等人（国家科学院学报，2018, 8505-8510）的早期工作，我们开发并展示了一种基于深度神经网络技术的基于模拟的计算方法。到目前为止，我们研究的测试问题中，我们的方法的精度在一个小数范围内准确。

    Motivated by applications in queueing theory, we consider a stochastic control problem whose state space is the $d$-dimensional positive orthant. The controlled process $Z$ evolves as a reflected Brownian motion whose covariance matrix is exogenously specified, as are its directions of reflection from the orthant's boundary surfaces. A system manager chooses a drift vector $\theta(t)$ at each time $t$ based on the history of $Z$, and the cost rate at time $t$ depends on both $Z(t)$ and $\theta(t)$. In our initial problem formulation, the objective is to minimize expected discounted cost over an infinite planning horizon, after which we treat the corresponding ergodic control problem. Extending earlier work by Han et al. (Proceedings of the National Academy of Sciences, 2018, 8505-8510), we develop and illustrate a simulation-based computational method that relies heavily on deep neural network technology. For test problems studied thus far, our method is accurate to within a fraction o
    

