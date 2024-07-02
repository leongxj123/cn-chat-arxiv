# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Update Monte Carlo tree search (UMCTS) algorithm for heuristic global search of sizing optimization problems for truss structures.](http://arxiv.org/abs/2309.06045) | 本文提出了一种基于启发式全局搜索的算法（UMCTS）用于桁架结构尺寸优化问题，通过结合更新过程和蒙特卡洛树搜索（MCTS）以及使用上界置信度（UCB）来获得合适的设计方案。 |
| [^2] | [Affine Invariant Ensemble Transform Methods to Improve Predictive Uncertainty in ReLU Networks.](http://arxiv.org/abs/2309.04742) | 本文提出了一种仿射不变集成变换方法，可以改善在ReLU网络中的预测不确定性。通过基于集合卡尔曼滤波的贝叶斯推断，我们提出了两个相互作用的粒子系统，并证明了它们的收敛性。同时，我们还探讨了这些方法用于量化预测不确定性的有效性。 |
| [^3] | [Inverse Evolution Layers: Physics-informed Regularizers for Deep Neural Networks.](http://arxiv.org/abs/2307.07344) | 本文提出了一种新颖的方法，通过一种新的正则化方法将基于偏微分方程（PDE）的进化模型与神经网络集成在一起。这些层可以实现特定的正则化目标，并赋予神经网络输出与进化模型对应的特性。此外，逆进化层的构建和实现相对简单，可以轻松地为不同的物理进化和神经网络设计。 |

# 详细

[^1]: 基于启发式全局搜索的桁架结构尺寸优化问题的改进蒙特卡洛树搜索（UMCTS）算法

    Update Monte Carlo tree search (UMCTS) algorithm for heuristic global search of sizing optimization problems for truss structures. (arXiv:2309.06045v1 [cs.AI])

    [http://arxiv.org/abs/2309.06045](http://arxiv.org/abs/2309.06045)

    本文提出了一种基于启发式全局搜索的算法（UMCTS）用于桁架结构尺寸优化问题，通过结合更新过程和蒙特卡洛树搜索（MCTS）以及使用上界置信度（UCB）来获得合适的设计方案。

    

    桁架结构尺寸优化是一个复杂的计算问题，强化学习（RL）适用于处理无梯度计算的多模态问题。本研究提出了一种新的高效优化算法——更新蒙特卡洛树搜索（UMCTS），用于获得合适的桁架结构设计。UMCTS是一种基于RL的方法，将新颖的更新过程和蒙特卡洛树搜索（MCTS）与上界置信度（UCB）相结合。更新过程意味着在每一轮中，每个构件的最佳截面积通过搜索树确定，其初始状态是上一轮的最终状态。在UMCTS算法中，引入了加速选择成员面积和迭代次数的加速器，以减少计算时间。此外，对于每个状态，平均奖励被最佳奖励的模拟过程中收集来的奖励替代，确定最优解。本文提出了一种优化算法，通过结合更新过程和MCTS，以及使用UCB来优化桁架结构设计。

    Sizing optimization of truss structures is a complex computational problem, and the reinforcement learning (RL) is suitable for dealing with multimodal problems without gradient computations. In this paper, a new efficient optimization algorithm called update Monte Carlo tree search (UMCTS) is developed to obtain the appropriate design for truss structures. UMCTS is an RL-based method that combines the novel update process and Monte Carlo tree search (MCTS) with the upper confidence bound (UCB). Update process means that in each round, the optimal cross-sectional area of each member is determined by search tree, and its initial state is the final state in the previous round. In the UMCTS algorithm, an accelerator for the number of selections for member area and iteration number is introduced to reduce the computation time. Moreover, for each state, the average reward is replaced by the best reward collected on the simulation process to determine the optimal solution. The proposed optim
    
[^2]: 改进ReLU网络的预测不确定性的仿射不变集成变换方法

    Affine Invariant Ensemble Transform Methods to Improve Predictive Uncertainty in ReLU Networks. (arXiv:2309.04742v1 [stat.ML])

    [http://arxiv.org/abs/2309.04742](http://arxiv.org/abs/2309.04742)

    本文提出了一种仿射不变集成变换方法，可以改善在ReLU网络中的预测不确定性。通过基于集合卡尔曼滤波的贝叶斯推断，我们提出了两个相互作用的粒子系统，并证明了它们的收敛性。同时，我们还探讨了这些方法用于量化预测不确定性的有效性。

    

    我们考虑使用合适的集合卡尔曼滤波的扩展进行逻辑回归的贝叶斯推断问题。我们提出了两个相互作用的粒子系统，从近似后验分布中采样，并证明当粒子数量趋于无穷时，这些相互作用粒子系统收敛到均场极限的量化收敛速率。此外，我们应用这些技术并考察它们作为贝叶斯近似方法在ReLU网络中量化预测不确定性的有效性。

    We consider the problem of performing Bayesian inference for logistic regression using appropriate extensions of the ensemble Kalman filter. Two interacting particle systems are proposed that sample from an approximate posterior and prove quantitative convergence rates of these interacting particle systems to their mean-field limit as the number of particles tends to infinity. Furthermore, we apply these techniques and examine their effectiveness as methods of Bayesian approximation for quantifying predictive uncertainty in ReLU networks.
    
[^3]: 逆进化层:物理信息化正则化器用于深度神经网络

    Inverse Evolution Layers: Physics-informed Regularizers for Deep Neural Networks. (arXiv:2307.07344v1 [cs.LG])

    [http://arxiv.org/abs/2307.07344](http://arxiv.org/abs/2307.07344)

    本文提出了一种新颖的方法，通过一种新的正则化方法将基于偏微分方程（PDE）的进化模型与神经网络集成在一起。这些层可以实现特定的正则化目标，并赋予神经网络输出与进化模型对应的特性。此外，逆进化层的构建和实现相对简单，可以轻松地为不同的物理进化和神经网络设计。

    

    本文提出了一种新颖的方法，通过一种新的正则化方法将基于偏微分方程（PDE）的进化模型与神经网络集成在一起。具体而言，我们提出了基于进化方程的逆进化层（IELs）。这些层可以实现特定的正则化目标，并赋予神经网络输出与进化模型对应的特性。此外，逆进化层的构建和实现相对简单，可以轻松地为不同的物理进化和神经网络设计。此外，这些层的设计过程可以为神经网络提供直观和数学可解释性，从而增强了方法的透明度和解释性。为了证明我们方法的有效性、效率和简单性，我们提出了一个将语义分割模型赋予热扩散模型平滑性属性的示例。

    This paper proposes a novel approach to integrating partial differential equation (PDE)-based evolution models into neural networks through a new type of regularization. Specifically, we propose inverse evolution layers (IELs) based on evolution equations. These layers can achieve specific regularization objectives and endow neural networks' outputs with corresponding properties of the evolution models. Moreover, IELs are straightforward to construct and implement, and can be easily designed for various physical evolutions and neural networks. Additionally, the design process for these layers can provide neural networks with intuitive and mathematical interpretability, thus enhancing the transparency and explainability of the approach. To demonstrate the effectiveness, efficiency, and simplicity of our approach, we present an example of endowing semantic segmentation models with the smoothness property based on the heat diffusion model. To achieve this goal, we design heat-diffusion IE
    

