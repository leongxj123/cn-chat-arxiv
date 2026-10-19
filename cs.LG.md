# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep Conditional Generative Learning: Model and Error Analysis](https://rss.arxiv.org/abs/2402.01460) | 提出了一种基于ODE的深度生成方法，通过条件Follmer流来学习条件分布，通过离散化和深度神经网络实现高效转化。同时，通过Wasserstein距离的非渐近收敛速率，提供了第一个端到端误差分析，数值实验证明其在不同场景下的优越性。 |
| [^2] | [UAV Pathfinding in Dynamic Obstacle Avoidance with Multi-agent Reinforcement Learning.](http://arxiv.org/abs/2310.16659) | 本文提出了一种基于多智能体强化学习的集中式训练和分布式执行方法，以在线解决动态障碍物避障问题。实验结果验证了该方法的有效性。 |
| [^3] | [Accelerated gradient methods for nonconvex optimization: Escape trajectories from strict saddle points and convergence to local minima.](http://arxiv.org/abs/2307.07030) | 本文研究了一类加速梯度方法在非凸优化问题上的行为，包括逃离鞍点和收敛到局部极小值点的分析。研究在渐进和非渐进情况下，提出了一类新的Nesterov类型的加速方法，并回答了Nesterov加速梯度方法是否避免了严格鞍点的问题。 |

# 详细

[^1]: 深度条件生成学习：模型与误差分析

    Deep Conditional Generative Learning: Model and Error Analysis

    [https://rss.arxiv.org/abs/2402.01460](https://rss.arxiv.org/abs/2402.01460)

    提出了一种基于ODE的深度生成方法，通过条件Follmer流来学习条件分布，通过离散化和深度神经网络实现高效转化。同时，通过Wasserstein距离的非渐近收敛速率，提供了第一个端到端误差分析，数值实验证明其在不同场景下的优越性。

    

    我们介绍了一种基于普通微分方程（ODE）的深度生成方法，用于学习条件分布，称为条件Follmer流。从标准高斯分布开始，所提出的流能够以高效的方式将其转化为目标条件分布，在时间1处达到稳定。为了有效实现，我们使用欧拉方法对流进行离散化，使用深度神经网络非参数化估计速度场。此外，我们导出了学习样本的分布与目标分布之间的Wasserstein距离的非渐近收敛速率，在条件分布学习中提供了第一个全面的端到端误差分析。我们的数值实验展示了它在一系列情况下的有效性，从标准的非参数化条件密度估计问题到涉及图像数据的更复杂的挑战，说明它优于各种现有方法。

    We introduce an Ordinary Differential Equation (ODE) based deep generative method for learning a conditional distribution, named the Conditional Follmer Flow. Starting from a standard Gaussian distribution, the proposed flow could efficiently transform it into the target conditional distribution at time 1. For effective implementation, we discretize the flow with Euler's method where we estimate the velocity field nonparametrically using a deep neural network. Furthermore, we derive a non-asymptotic convergence rate in the Wasserstein distance between the distribution of the learned samples and the target distribution, providing the first comprehensive end-to-end error analysis for conditional distribution learning via ODE flow. Our numerical experiments showcase its effectiveness across a range of scenarios, from standard nonparametric conditional density estimation problems to more intricate challenges involving image data, illustrating its superiority over various existing condition
    
[^2]: 无人机在多智能体强化学习中的动态避障路径规划

    UAV Pathfinding in Dynamic Obstacle Avoidance with Multi-agent Reinforcement Learning. (arXiv:2310.16659v1 [cs.RO])

    [http://arxiv.org/abs/2310.16659](http://arxiv.org/abs/2310.16659)

    本文提出了一种基于多智能体强化学习的集中式训练和分布式执行方法，以在线解决动态障碍物避障问题。实验结果验证了该方法的有效性。

    

    多智能体强化学习方法在动态和不确定的场景中在线规划智能体的可行且安全的路径具有重要意义。本文提出了一种基于多智能体强化学习的集中式训练和分布式执行方法，以在线解决动态障碍物避障问题。在该方法中，每个智能体仅与中央规划者或其邻居进行通信，以在线规划可行且安全的路径。我们基于模型预测控制的思想改进了我们的方法，以提高智能体的训练效率和采样利用率。在模拟、室内和室外环境中的实验结果验证了我们方法的有效性。

    Multi-agent reinforcement learning based methods are significant for online planning of feasible and safe paths for agents in dynamic and uncertain scenarios. Although some methods like fully centralized and fully decentralized methods achieve a certain measure of success, they also encounter problems such as dimension explosion and poor convergence, respectively. In this paper, we propose a novel centralized training with decentralized execution method based on multi-agent reinforcement learning to solve the dynamic obstacle avoidance problem online. In this approach, each agent communicates only with the central planner or only with its neighbors, respectively, to plan feasible and safe paths online. We improve our methods based on the idea of model predictive control to increase the training efficiency and sample utilization of agents. The experimental results in both simulation, indoor, and outdoor environments validate the effectiveness of our method. The video is available at htt
    
[^3]: 加速梯度方法用于非凸优化：逃逸轨迹和收敛到局部极小值点

    Accelerated gradient methods for nonconvex optimization: Escape trajectories from strict saddle points and convergence to local minima. (arXiv:2307.07030v1 [math.OC])

    [http://arxiv.org/abs/2307.07030](http://arxiv.org/abs/2307.07030)

    本文研究了一类加速梯度方法在非凸优化问题上的行为，包括逃离鞍点和收敛到局部极小值点的分析。研究在渐进和非渐进情况下，提出了一类新的Nesterov类型的加速方法，并回答了Nesterov加速梯度方法是否避免了严格鞍点的问题。

    

    本文研究了一类广义的加速梯度方法在光滑非凸函数上的行为。通过对Polyak的重球方法和Nesterov加速梯度方法进行改进，以实现对非凸函数局部极小值的收敛，本文提出了一类Nesterov类型的加速方法，并通过渐进分析和非渐进分析对这些方法进行了严格研究，包括逃离鞍点和收敛到局部极小值点。在渐进情况下，本文回答了一个开放问题，即带有可变动量参数的Nesterov加速梯度方法（NAG）是否几乎必定避免了严格鞍点。本文还提出了两种渐进收敛和发散的度量方式，并对几种常用的标准加速方法（如NAG和Ne）进行了评估。

    This paper considers the problem of understanding the behavior of a general class of accelerated gradient methods on smooth nonconvex functions. Motivated by some recent works that have proposed effective algorithms, based on Polyak's heavy ball method and the Nesterov accelerated gradient method, to achieve convergence to a local minimum of nonconvex functions, this work proposes a broad class of Nesterov-type accelerated methods and puts forth a rigorous study of these methods encompassing the escape from saddle-points and convergence to local minima through a both asymptotic and a non-asymptotic analysis. In the asymptotic regime, this paper answers an open question of whether Nesterov's accelerated gradient method (NAG) with variable momentum parameter avoids strict saddle points almost surely. This work also develops two metrics of asymptotic rate of convergence and divergence, and evaluates these two metrics for several popular standard accelerated methods such as the NAG, and Ne
    

