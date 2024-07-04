# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Collaborative Multi-Agent Heterogeneous Multi-Armed Bandits.](http://arxiv.org/abs/2305.18784) | 本研究研究了一个新的合作多智能体老虎机设置，并发展了去中心化算法以减少代理之间的集体遗憾，在数学分析中证明了该算法实现了近乎最优性能。 |
| [^2] | [Visual Analytics of Multivariate Networks with Representation Learning and Composite Variable Construction.](http://arxiv.org/abs/2303.09590) | 本文提出了一种用于研究多变量网络的视觉分析工作流程，其中包括神经网络学习阶段、降维和优化阶段以及用户交互式可视化接口进行解释。关键的组合变量构建步骤将非线性特征重塑为线性特征，以方便检查和理解。案例研究表明该工作流程具有有效性和可理解性。 |

# 详细

[^1]: 合作多智能体异构多臂老虎机翻译论文

    Collaborative Multi-Agent Heterogeneous Multi-Armed Bandits. (arXiv:2305.18784v1 [cs.LG])

    [http://arxiv.org/abs/2305.18784](http://arxiv.org/abs/2305.18784)

    本研究研究了一个新的合作多智能体老虎机设置，并发展了去中心化算法以减少代理之间的集体遗憾，在数学分析中证明了该算法实现了近乎最优性能。

    

    最近合作多智能体老虎机的研究吸引了很多关注。因此，我们开始研究一个新的合作设置，其中$N$个智能体中的每个智能体正在学习$M$个具有随机性的多臂老虎机，以减少他们的集体累计遗憾。我们开发了去中心化算法，促进了代理之间的合作，并针对两种情况进行了性能表征。通过推导每个代理的累积遗憾和集体遗憾的上限，我们对这些算法的性能进行了表征。我们还证明了这种情况下集体遗憾的下限，证明了所提出算法的近乎最优性能。

    The study of collaborative multi-agent bandits has attracted significant attention recently. In light of this, we initiate the study of a new collaborative setting, consisting of $N$ agents such that each agent is learning one of $M$ stochastic multi-armed bandits to minimize their group cumulative regret. We develop decentralized algorithms which facilitate collaboration between the agents under two scenarios. We characterize the performance of these algorithms by deriving the per agent cumulative regret and group regret upper bounds. We also prove lower bounds for the group regret in this setting, which demonstrates the near-optimal behavior of the proposed algorithms.
    
[^2]: 用表示学习和组合变量构建多变量网络的视觉分析

    Visual Analytics of Multivariate Networks with Representation Learning and Composite Variable Construction. (arXiv:2303.09590v1 [cs.SI])

    [http://arxiv.org/abs/2303.09590](http://arxiv.org/abs/2303.09590)

    本文提出了一种用于研究多变量网络的视觉分析工作流程，其中包括神经网络学习阶段、降维和优化阶段以及用户交互式可视化接口进行解释。关键的组合变量构建步骤将非线性特征重塑为线性特征，以方便检查和理解。案例研究表明该工作流程具有有效性和可理解性。

    

    多变量网络在真实世界的数据驱动应用中经常被发现。发掘和理解多变量网络中的关系并不是一项简单的任务。本文提出了一种用于研究多变量网络以提取网络不同结构和语义特征之间关联的视觉分析工作流程（例如，什么是在社交网络密度方面与不同属性的组合关系）。该工作流程包括基于神经网络的学习阶段，根据所选输入和输出属性来对数据进行分类，降维和优化阶段以产生一个简化的结果集合以便检查，最后通过用户交互式可视化接口进行解释阶段的操作。我们设计的一个关键部分是组合变量构建步骤，该步骤将由神经网络获得的非线性特征重塑为直观解释的线性特征。我们通过对大型组织员工之间的电子邮件通信数据集进行案例研究，证明了工作流程的有效性和可理解性。

    Multivariate networks are commonly found in real-world data-driven applications. Uncovering and understanding the relations of interest in multivariate networks is not a trivial task. This paper presents a visual analytics workflow for studying multivariate networks to extract associations between different structural and semantic characteristics of the networks (e.g., what are the combinations of attributes largely relating to the density of a social network?). The workflow consists of a neural-network-based learning phase to classify the data based on the chosen input and output attributes, a dimensionality reduction and optimization phase to produce a simplified set of results for examination, and finally an interpreting phase conducted by the user through an interactive visualization interface. A key part of our design is a composite variable construction step that remodels nonlinear features obtained by neural networks into linear features that are intuitive to interpret. We demon
    

