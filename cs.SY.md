# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Active Inverse Learning in Stackelberg Trajectory Games.](http://arxiv.org/abs/2308.08017) | 这项研究提出了一种在Stackelberg博弈中的主动逆向学习方法，通过活跃地最大化跟随者在不同假设下的轨迹差异来加速领导者的推断过程。 |
| [^2] | [Meta-Learning Operators to Optimality from Multi-Task Non-IID Data.](http://arxiv.org/abs/2308.04428) | 本文提出了从多任务非独立同分布数据中恢复线性操作符的方法，并发现现有的各向同性无关的元学习方法会对表示更新造成偏差，限制了表示学习的样本复杂性。为此，引入了去偏差和特征白化的适应方法。 |

# 详细

[^1]: Stackelberg轨迹博弈中的主动逆向学习

    Active Inverse Learning in Stackelberg Trajectory Games. (arXiv:2308.08017v1 [cs.GT])

    [http://arxiv.org/abs/2308.08017](http://arxiv.org/abs/2308.08017)

    这项研究提出了一种在Stackelberg博弈中的主动逆向学习方法，通过活跃地最大化跟随者在不同假设下的轨迹差异来加速领导者的推断过程。

    

    博弈论的逆向学习是从玩家的行为中推断出他们的目标的问题。我们在一个Stackelberg博弈中，通过每个玩家的动态系统轨迹来定义一个逆向学习问题，其中包括一个领导者和一个跟随者。我们提出了一种主动逆向学习方法，让领导者推断出一个有限候选集中描述跟随者目标函数的假设。与现有方法使用被动观察到的轨迹不同，所提出的方法主动地最大化不同假设下跟随者轨迹的差异，加速领导者的推断过程。我们在一个递进的重复轨迹博弈中展示了所提出的方法。与均匀随机输入相比，所提供的方法加速了概率收敛到条件于跟随者轨迹的不同假设上的收敛速度。

    Game-theoretic inverse learning is the problem of inferring the players' objectives from their actions. We formulate an inverse learning problem in a Stackelberg game between a leader and a follower, where each player's action is the trajectory of a dynamical system. We propose an active inverse learning method for the leader to infer which hypothesis among a finite set of candidates describes the follower's objective function. Instead of using passively observed trajectories like existing methods, the proposed method actively maximizes the differences in the follower's trajectories under different hypotheses to accelerate the leader's inference. We demonstrate the proposed method in a receding-horizon repeated trajectory game. Compared with uniformly random inputs, the leader inputs provided by the proposed method accelerate the convergence of the probability of different hypotheses conditioned on the follower's trajectory by orders of magnitude.
    
[^2]: 从多任务非独立同分布数据中元学习操作符到最优性

    Meta-Learning Operators to Optimality from Multi-Task Non-IID Data. (arXiv:2308.04428v1 [stat.ML])

    [http://arxiv.org/abs/2308.04428](http://arxiv.org/abs/2308.04428)

    本文提出了从多任务非独立同分布数据中恢复线性操作符的方法，并发现现有的各向同性无关的元学习方法会对表示更新造成偏差，限制了表示学习的样本复杂性。为此，引入了去偏差和特征白化的适应方法。

    

    机器学习中最近取得进展的一个强大概念是从异构来源或任务的数据中提取共同特征。直观地说，将所有数据用于学习共同的表示函数，既有助于计算效率，又有助于统计泛化，因为它可以减少要在给定任务上进行微调的参数数量。为了在理论上做出这些优点的根源，我们提出了从噪声向量测量$y = Mx + w$中回复线性操作符$M$的一般模型。其中，协变量$x$既可以是非独立同分布的，也可以是非各向同性的。我们证明了现有的各向同性无关的元学习方法会对表示更新造成偏差，这导致噪声项的缩放不再有利于源任务数量。这反过来会导致表示学习的样本复杂性受到单任务数据规模的限制。我们引入了一种方法，称为去偏差和特征白化。

    A powerful concept behind much of the recent progress in machine learning is the extraction of common features across data from heterogeneous sources or tasks. Intuitively, using all of one's data to learn a common representation function benefits both computational effort and statistical generalization by leaving a smaller number of parameters to fine-tune on a given task. Toward theoretically grounding these merits, we propose a general setting of recovering linear operators $M$ from noisy vector measurements $y = Mx + w$, where the covariates $x$ may be both non-i.i.d. and non-isotropic. We demonstrate that existing isotropy-agnostic meta-learning approaches incur biases on the representation update, which causes the scaling of the noise terms to lose favorable dependence on the number of source tasks. This in turn can cause the sample complexity of representation learning to be bottlenecked by the single-task data size. We introduce an adaptation, $\texttt{De-bias & Feature-Whiten}
    

