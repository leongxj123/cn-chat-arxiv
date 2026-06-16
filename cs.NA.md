# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-Grade Deep Learning for Partial Differential Equations with Applications to the Burgers Equation.](http://arxiv.org/abs/2309.07401) | 本文提出了一种多级深度学习方法，用于解决非线性偏微分方程。该方法通过将DNN的学习任务分解为多个堆叠的神经网络，以解决随着网络层数增加而导致的非凸优化问题的复杂度增加的挑战。 |

# 详细

[^1]: 多级深度学习在解决偏微分方程中的应用，以Burgers方程为例

    Multi-Grade Deep Learning for Partial Differential Equations with Applications to the Burgers Equation. (arXiv:2309.07401v1 [math.NA])

    [http://arxiv.org/abs/2309.07401](http://arxiv.org/abs/2309.07401)

    本文提出了一种多级深度学习方法，用于解决非线性偏微分方程。该方法通过将DNN的学习任务分解为多个堆叠的神经网络，以解决随着网络层数增加而导致的非凸优化问题的复杂度增加的挑战。

    

    本文提出了一种多级深度学习方法，用于解决非线性偏微分方程（PDEs）。深度神经网络在解决PDEs方面表现出超强的性能，除了在自然语言处理、计算机视觉和机器人等领域的卓越成功。然而，训练一个非常深的网络往往是一项具有挑战性的任务。随着DNN的层数增加，解决由PDEs的DNN求解结果导致的大规模非凸优化问题变得越来越困难，这可能导致预测准确性的降低而不是增加。为了克服这一挑战，我们提出了一种两阶段多级深度学习（TS-MGDL）方法，将学习DNN的任务分解为一系列堆叠在彼此上方的神经网络。这种方法可以减轻解决具有大量参数的非凸优化问题的复杂度，并学习残差。

    We develop in this paper a multi-grade deep learning method for solving nonlinear partial differential equations (PDEs). Deep neural networks (DNNs) have received super performance in solving PDEs in addition to their outstanding success in areas such as natural language processing, computer vision, and robotics. However, training a very deep network is often a challenging task. As the number of layers of a DNN increases, solving a large-scale non-convex optimization problem that results in the DNN solution of PDEs becomes more and more difficult, which may lead to a decrease rather than an increase in predictive accuracy. To overcome this challenge, we propose a two-stage multi-grade deep learning (TS-MGDL) method that breaks down the task of learning a DNN into several neural networks stacked on top of each other in a staircase-like manner. This approach allows us to mitigate the complexity of solving the non-convex optimization problem with large number of parameters and learn resid
    

