# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Input Convex LSTM: A Convex Approach for Fast Lyapunov-Based Model Predictive Control.](http://arxiv.org/abs/2311.07202) | 本研究提出了一种基于输入凸LSTM的基于Lyapunov的模型预测控制方法，通过减少收敛时间和缓解梯度消失/爆炸问题来改善MPC的性能。 |
| [^2] | [System Identification for Continuous-time Linear Dynamical Systems.](http://arxiv.org/abs/2308.11933) | 本文解决了在连续时间下观测不规则采样的情况下，Kalman滤波器的系统识别问题。通过引入连续时间Ito随机微分方程来推广Kalman滤波器的学习，并提供一个新颖的两滤波器的后验计算方法，通过贝叶斯派生获得的解析形式的后验计算方法可以高效地估计SDE的参数。 |
| [^3] | [Efficient Interaction-Aware Interval Analysis of Neural Network Feedback Loops.](http://arxiv.org/abs/2307.14938) | 本文提出了一种计算效率高的神经网络控制系统区间可达性分析框架，通过引入包含函数和构建嵌入系统来捕捉系统和神经网络控制器之间的相互作用。 |
| [^4] | [A Convex Hull Cheapest Insertion Heuristic for the Non-Euclidean TSP.](http://arxiv.org/abs/2302.06582) | 本文提出了一种适用于非欧几里德旅行商问题的凸包最便宜插入启发式解法，通过使用多维缩放将非欧几里德空间的点近似到欧几里德空间，生成了初始化算法的凸包。在评估中发现，该算法在大多数情况下优于最邻近算法。 |

# 详细

[^1]: 输入凸LSTM：一种快速基于Lyapunov模型预测控制的凸方法

    Input Convex LSTM: A Convex Approach for Fast Lyapunov-Based Model Predictive Control. (arXiv:2311.07202v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2311.07202](http://arxiv.org/abs/2311.07202)

    本研究提出了一种基于输入凸LSTM的基于Lyapunov的模型预测控制方法，通过减少收敛时间和缓解梯度消失/爆炸问题来改善MPC的性能。

    

    利用输入凸神经网络（ICNN），基于ICNN的模型预测控制（MPC）通过在MPC框架中保持凸性成功实现全局最优解。然而，当前的ICNN架构存在梯度消失/爆炸问题，限制了它们作为复杂任务的深度神经网络的能力。此外，当前基于神经网络的MPC，包括传统的基于神经网络的MPC和基于ICNN的MPC，与基于第一原理模型的MPC相比面临较慢的收敛速度。在本研究中，我们利用ICNN的原理提出了一种新的基于输入凸LSTM的基于Lyapunov的MPC，旨在减少收敛时间、缓解梯度消失/爆炸问题并确保闭环稳定性。通过对非线性化学反应器的模拟研究，我们观察到了梯度消失/爆炸问题的缓解和收敛时间的减少，收敛时间平均降低了一定的百分之。

    Leveraging Input Convex Neural Networks (ICNNs), ICNN-based Model Predictive Control (MPC) successfully attains globally optimal solutions by upholding convexity within the MPC framework. However, current ICNN architectures encounter the issue of vanishing/exploding gradients, which limits their ability to serve as deep neural networks for complex tasks. Additionally, the current neural network-based MPC, including conventional neural network-based MPC and ICNN-based MPC, faces slower convergence speed when compared to MPC based on first-principles models. In this study, we leverage the principles of ICNNs to propose a novel Input Convex LSTM for Lyapunov-based MPC, with the specific goal of reducing convergence time and mitigating the vanishing/exploding gradient problem while ensuring closed-loop stability. From a simulation study of a nonlinear chemical reactor, we observed a mitigation of vanishing/exploding gradient problem and a reduction in convergence time, with a percentage de
    
[^2]: 连续时间线性动态系统的系统识别

    System Identification for Continuous-time Linear Dynamical Systems. (arXiv:2308.11933v1 [cs.LG])

    [http://arxiv.org/abs/2308.11933](http://arxiv.org/abs/2308.11933)

    本文解决了在连续时间下观测不规则采样的情况下，Kalman滤波器的系统识别问题。通过引入连续时间Ito随机微分方程来推广Kalman滤波器的学习，并提供一个新颖的两滤波器的后验计算方法，通过贝叶斯派生获得的解析形式的后验计算方法可以高效地估计SDE的参数。

    

    Kalman滤波器的系统识别问题在学习动态系统的基础参数时，通常假设观测值在等间隔的时间点采样。然而，在许多应用中，这个假设是有限制和不切实际的。本文针对连续离散滤波器的系统识别问题，通过求解连续时间Ito随机微分方程（SDE）来推广Kalman滤波器的学习。我们引入了一个新颖的两滤波器，具有贝叶斯派生的解析形式的后验，这样可以得到不需要预先计算的正向传递的解析更新。利用这种解析的高效计算后验的方法，我们提供了一种EM过程，用于估计SDE的参数，自然地纳入了不规则采样的测量。

    The problem of system identification for the Kalman filter, relying on the expectation-maximization (EM) procedure to learn the underlying parameters of a dynamical system, has largely been studied assuming that observations are sampled at equally-spaced time points. However, in many applications this is a restrictive and unrealistic assumption. This paper addresses system identification for the continuous-discrete filter, with the aim of generalizing learning for the Kalman filter by relying on a solution to a continuous-time It\^o stochastic differential equation (SDE) for the latent state and covariance dynamics. We introduce a novel two-filter, analytical form for the posterior with a Bayesian derivation, which yields analytical updates which do not require the forward-pass to be pre-computed. Using this analytical and efficient computation of the posterior, we provide an EM procedure which estimates the parameters of the SDE, naturally incorporating irregularly sampled measurement
    
[^3]: 高效互动感知神经网络反馈环的区间分析

    Efficient Interaction-Aware Interval Analysis of Neural Network Feedback Loops. (arXiv:2307.14938v1 [eess.SY])

    [http://arxiv.org/abs/2307.14938](http://arxiv.org/abs/2307.14938)

    本文提出了一种计算效率高的神经网络控制系统区间可达性分析框架，通过引入包含函数和构建嵌入系统来捕捉系统和神经网络控制器之间的相互作用。

    

    本文提出了一种计算效率高的神经网络控制系统区间可达性分析框架。我们的方法基于神经网络控制器和开环系统的包含函数。我们观察到，许多最先进的神经网络验证器可以为神经网络生成包含函数。我们介绍并分析了一种基于函数雅可比边界的开环动力学包含函数的新类别，特别适用于捕捉系统和神经网络控制器之间的相互作用。接下来，对于任意动力系统，我们使用包含函数构建一个状态数是原系统两倍的嵌入系统。我们证明嵌入系统的单个轨迹可以提供可达集的超矩形近似。然后，我们提出了两种构建神经网络控制动力系统的闭环嵌入系统的方法，考虑系统之间的互动。

    In this paper, we propose a computationally efficient framework for interval reachability of neural network controlled systems. Our approach builds upon inclusion functions for the neural network controller and the open-loop system. We observe that many state-of-the-art neural network verifiers can produce inclusion functions for neural networks. We introduce and analyze a new class of inclusion functions for the open-loop dynamics based on bounds of the function Jacobian that is particularly suitable for capturing the interactions between systems and neural network controllers. Next, for any dynamical system, we use inclusion functions to construct an embedding system with twice the number of states as the original system. We show that a single trajectory of this embedding system provides hyper-rectangular over-approximations of reachable sets. We then propose two approaches for constructing a closed-loop embedding system for a neural network controlled dynamical system that accounts 
    
[^4]: 非欧几里德旅行商问题的凸包最便宜插入启发式解法

    A Convex Hull Cheapest Insertion Heuristic for the Non-Euclidean TSP. (arXiv:2302.06582v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2302.06582](http://arxiv.org/abs/2302.06582)

    本文提出了一种适用于非欧几里德旅行商问题的凸包最便宜插入启发式解法，通过使用多维缩放将非欧几里德空间的点近似到欧几里德空间，生成了初始化算法的凸包。在评估中发现，该算法在大多数情况下优于最邻近算法。

    

    众所周知，凸包最便宜插入启发式算法可以在欧几里德空间中产生良好的旅行商问题解决方案，但还未在非欧几里德情况下进行扩展。为了解决非欧几里德空间中处理障碍物的困难，提出的改进方法使用多维缩放将这些点首先近似到欧几里德空间，从而可以生成初始化算法的凸包。通过修改TSPLIB基准数据集，向其中添加不可通过的分割器来产生非欧几里德空间，评估了所提出的算法。在所研究的案例中，该算法表现出优于常用的最邻近算法的性能，达到96%的情况。

    The convex hull cheapest insertion heuristic is known to generate good solutions to the Traveling Salesperson Problem in Euclidean spaces, but it has not been extended to the non-Euclidean case. To address the difficulty of dealing with obstacles in the non-Euclidean space, the proposed adaptation uses multidimensional scaling to first approximate these points in a Euclidean space, thereby enabling the generation of the convex hull that initializes the algorithm. To evaluate the proposed algorithm, the TSPLIB benchmark data-set is modified by adding impassable separators that produce non-Euclidean spaces. The algorithm is demonstrated to outperform the commonly used Nearest Neighbor algorithm in 96% of the cases studied.
    

