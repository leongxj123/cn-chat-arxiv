# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Socially Integrated Navigation: A Social Acting Robot with Deep Reinforcement Learning](https://arxiv.org/abs/2403.09793) | 提出了一种新颖的社会整合导航方法，通过与人的互动使机器人的社交行为自适应，并从其他基于DRL的导航方法中区分出具有明确预定义社交行为的社会意识方法和缺乏社交行为的社会碰撞回避。 |
| [^2] | [Graph-Structured Kernel Design for Power Flow Learning using Gaussian Processes.](http://arxiv.org/abs/2308.07867) | 本文提出了一种图结构核设计，用于使用高斯过程进行功率流学习，通过顶点度核和网络扫描主动学习方案，实现了更高效的学习和样本复杂度降低。 |
| [^3] | [Meta-Learning Operators to Optimality from Multi-Task Non-IID Data.](http://arxiv.org/abs/2308.04428) | 本文提出了从多任务非独立同分布数据中恢复线性操作符的方法，并发现现有的各向同性无关的元学习方法会对表示更新造成偏差，限制了表示学习的样本复杂性。为此，引入了去偏差和特征白化的适应方法。 |

# 详细

[^1]: 社会整合导航：具有深度强化学习的社交行动机器人

    Socially Integrated Navigation: A Social Acting Robot with Deep Reinforcement Learning

    [https://arxiv.org/abs/2403.09793](https://arxiv.org/abs/2403.09793)

    提出了一种新颖的社会整合导航方法，通过与人的互动使机器人的社交行为自适应，并从其他基于DRL的导航方法中区分出具有明确预定义社交行为的社会意识方法和缺乏社交行为的社会碰撞回避。

    

    移动机器人正在广泛应用于各种拥挤场景，并成为我们社会的一部分。一个具有个体人类考虑的社会可接受的导航行为对于可扩展的应用和人类接受至关重要。最近使用深度强化学习（DRL）方法来学习机器人的导航策略，并对机器人与人类之间的复杂交互进行建模。我们建议根据机器人展示的社交行为将现有基于DRL的导航方法分为具有缺乏社交行为的社会碰撞回避和具有明确预定义社交行为的社会意识方法。此外，我们提出了一种新颖的社会整合导航方法，其中机器人的社交行为是自适应的，并且是通过与人类的互动而产生的。我们的方法的构式源自社会学定义，

    arXiv:2403.09793v1 Announce Type: cross  Abstract: Mobile robots are being used on a large scale in various crowded situations and become part of our society. The socially acceptable navigation behavior of a mobile robot with individual human consideration is an essential requirement for scalable applications and human acceptance. Deep Reinforcement Learning (DRL) approaches are recently used to learn a robot's navigation policy and to model the complex interactions between robots and humans. We propose to divide existing DRL-based navigation approaches based on the robot's exhibited social behavior and distinguish between social collision avoidance with a lack of social behavior and socially aware approaches with explicit predefined social behavior. In addition, we propose a novel socially integrated navigation approach where the robot's social behavior is adaptive and emerges from the interaction with humans. The formulation of our approach is derived from a sociological definition, 
    
[^2]: 使用高斯过程进行功率流学习的图结构核设计

    Graph-Structured Kernel Design for Power Flow Learning using Gaussian Processes. (arXiv:2308.07867v1 [eess.SY])

    [http://arxiv.org/abs/2308.07867](http://arxiv.org/abs/2308.07867)

    本文提出了一种图结构核设计，用于使用高斯过程进行功率流学习，通过顶点度核和网络扫描主动学习方案，实现了更高效的学习和样本复杂度降低。

    

    本文提出了一种基于物理启发的图结构核设计，用于使用高斯过程进行功率流学习。该核被命名为顶点度核（VDK），它依赖于基于网络图或拓扑的电压注入关系的潜在分解。值得注意的是，VDK设计避免了需要解决核搜索的优化问题。为了提高效率，我们还探索了一种图缩减方法，以获得具有较少项的VDK表示。此外，我们提出了一种新颖的网络扫描主动学习方案，它智能地选择顺序训练输入，加速VDK的学习。利用VDK的可加性结构，主动学习算法对GP的预测方差进行了块下降类型的过程，作为信息增益的代理。仿真结果表明，所提出的VDK-GP与中等规模500个节点和大规模1354个节点的完整GP相比，实现了超过两倍的样本复杂度降低。

    This paper presents a physics-inspired graph-structured kernel designed for power flow learning using Gaussian Process (GP). The kernel, named the vertex-degree kernel (VDK), relies on latent decomposition of voltage-injection relationship based on the network graph or topology. Notably, VDK design avoids the need to solve optimization problems for kernel search. To enhance efficiency, we also explore a graph-reduction approach to obtain a VDK representation with lesser terms. Additionally, we propose a novel network-swipe active learning scheme, which intelligently selects sequential training inputs to accelerate the learning of VDK. Leveraging the additive structure of VDK, the active learning algorithm performs a block-descent type procedure on GP's predictive variance, serving as a proxy for information gain. Simulations demonstrate that the proposed VDK-GP achieves more than two fold sample complexity reduction, compared to full GP on medium scale 500-Bus and large scale 1354-Bus 
    
[^3]: 从多任务非独立同分布数据中元学习操作符到最优性

    Meta-Learning Operators to Optimality from Multi-Task Non-IID Data. (arXiv:2308.04428v1 [stat.ML])

    [http://arxiv.org/abs/2308.04428](http://arxiv.org/abs/2308.04428)

    本文提出了从多任务非独立同分布数据中恢复线性操作符的方法，并发现现有的各向同性无关的元学习方法会对表示更新造成偏差，限制了表示学习的样本复杂性。为此，引入了去偏差和特征白化的适应方法。

    

    机器学习中最近取得进展的一个强大概念是从异构来源或任务的数据中提取共同特征。直观地说，将所有数据用于学习共同的表示函数，既有助于计算效率，又有助于统计泛化，因为它可以减少要在给定任务上进行微调的参数数量。为了在理论上做出这些优点的根源，我们提出了从噪声向量测量$y = Mx + w$中回复线性操作符$M$的一般模型。其中，协变量$x$既可以是非独立同分布的，也可以是非各向同性的。我们证明了现有的各向同性无关的元学习方法会对表示更新造成偏差，这导致噪声项的缩放不再有利于源任务数量。这反过来会导致表示学习的样本复杂性受到单任务数据规模的限制。我们引入了一种方法，称为去偏差和特征白化。

    A powerful concept behind much of the recent progress in machine learning is the extraction of common features across data from heterogeneous sources or tasks. Intuitively, using all of one's data to learn a common representation function benefits both computational effort and statistical generalization by leaving a smaller number of parameters to fine-tune on a given task. Toward theoretically grounding these merits, we propose a general setting of recovering linear operators $M$ from noisy vector measurements $y = Mx + w$, where the covariates $x$ may be both non-i.i.d. and non-isotropic. We demonstrate that existing isotropy-agnostic meta-learning approaches incur biases on the representation update, which causes the scaling of the noise terms to lose favorable dependence on the number of source tasks. This in turn can cause the sample complexity of representation learning to be bottlenecked by the single-task data size. We introduce an adaptation, $\texttt{De-bias & Feature-Whiten}
    

