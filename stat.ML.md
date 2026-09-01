# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation](https://arxiv.org/abs/2402.04355) | PQMass是一种使用概率质量估计来评估生成模型质量的全面方法，能够直接处理高维数据，不依赖于假设或训练其他模型。 |
| [^2] | [PhyloGFN: Phylogenetic inference with generative flow networks.](http://arxiv.org/abs/2310.08774) | PhyloGFN是一种基于生成流网络的系统发育推断方法，通过采样复杂的组合结构，能够产生多样且高质量的进化假设，并在边缘似然估计方面具有竞争力。 |
| [^3] | [Delta-AI: Local objectives for amortized inference in sparse graphical models.](http://arxiv.org/abs/2310.02423) | Delta-AI算法提出了一种基于稀疏图模型的摊还推理方法，通过局部信用分配和离策略训练加快了训练速度。 |
| [^4] | [Potential Energy Advantage of Quantum Economy.](http://arxiv.org/abs/2308.08025) | 量子计算在能源效率方面具有优势，并且能够在盈利和能源效率上超越经典计算。这使得量子计算成为计算行业更可持续的选择。 |
| [^5] | [Deep graph kernel point processes.](http://arxiv.org/abs/2306.11313) | 本文提出了一种基于潜在图拓扑的图点过程方法，并开发了一种新颖的深度图核来描述事件之间的触发和抑制效应，该方法在合成和实际数据集上具有优越性。 |

# 详细

[^1]: PQMass: 使用概率质量估计的生成模型质量的概率评估

    PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation

    [https://arxiv.org/abs/2402.04355](https://arxiv.org/abs/2402.04355)

    PQMass是一种使用概率质量估计来评估生成模型质量的全面方法，能够直接处理高维数据，不依赖于假设或训练其他模型。

    

    我们提出了一种全面的基于样本的方法来评估生成模型的质量。所提出的方法能够估计两个样本集合来自同一分布的概率，为评估单个生成模型的性能或比较在同一数据集上训练的多个竞争模型提供了一个统计上严格的方法。该比较可以通过将空间划分为非重叠的区域并比较每个区域中的数据样本数量来进行。该方法仅需要生成模型和测试数据的样本。它能够直接处理高维数据，无需降维。显著的是，该方法不依赖于关于真实分布密度的假设，并且不依赖于训练或拟合任何辅助模型。相反，它着重于近似计算密度的积分（概率质量）。

    We propose a comprehensive sample-based method for assessing the quality of generative models. The proposed approach enables the estimation of the probability that two sets of samples are drawn from the same distribution, providing a statistically rigorous method for assessing the performance of a single generative model or the comparison of multiple competing models trained on the same dataset. This comparison can be conducted by dividing the space into non-overlapping regions and comparing the number of data samples in each region. The method only requires samples from the generative model and the test data. It is capable of functioning directly on high-dimensional data, obviating the need for dimensionality reduction. Significantly, the proposed method does not depend on assumptions regarding the density of the true distribution, and it does not rely on training or fitting any auxiliary models. Instead, it focuses on approximating the integral of the density (probability mass) acros
    
[^2]: PhyloGFN: 基于生成流网络的系统发育推断

    PhyloGFN: Phylogenetic inference with generative flow networks. (arXiv:2310.08774v1 [q-bio.PE])

    [http://arxiv.org/abs/2310.08774](http://arxiv.org/abs/2310.08774)

    PhyloGFN是一种基于生成流网络的系统发育推断方法，通过采样复杂的组合结构，能够产生多样且高质量的进化假设，并在边缘似然估计方面具有竞争力。

    

    系统发育学是计算生物学的一个分支，研究生物实体之间的进化关系。尽管有着悠久的历史和众多应用，但从序列数据推断系统发育树仍然具有挑战性：树空间的高复杂性对当前的组合和概率技术构成了重要障碍。在本文中，我们采用生成流网络（GFlowNets）的框架来解决系统发育学中的两个核心问题：基于最简原则的和贝叶斯的系统发育推断。由于GFlowNets适用于采样复杂的组合结构，它们是探索和采样树拓扑和进化距离的多模态后验分布的自然选择。我们证明了我们的摊还后验采样器PhyloGFN在真实基准数据集上产生多样且高质量的进化假设。PhyloGFN在边缘似然估计方面与之前的工作相比具有竞争力。

    Phylogenetics is a branch of computational biology that studies the evolutionary relationships among biological entities. Its long history and numerous applications notwithstanding, inference of phylogenetic trees from sequence data remains challenging: the high complexity of tree space poses a significant obstacle for the current combinatorial and probabilistic techniques. In this paper, we adopt the framework of generative flow networks (GFlowNets) to tackle two core problems in phylogenetics: parsimony-based and Bayesian phylogenetic inference. Because GFlowNets are well-suited for sampling complex combinatorial structures, they are a natural choice for exploring and sampling from the multimodal posterior distribution over tree topologies and evolutionary distances. We demonstrate that our amortized posterior sampler, PhyloGFN, produces diverse and high-quality evolutionary hypotheses on real benchmark datasets. PhyloGFN is competitive with prior works in marginal likelihood estimat
    
[^3]: Delta-AI: 稀疏图模型的摊还推理中的局部目标

    Delta-AI: Local objectives for amortized inference in sparse graphical models. (arXiv:2310.02423v1 [cs.LG])

    [http://arxiv.org/abs/2310.02423](http://arxiv.org/abs/2310.02423)

    Delta-AI算法提出了一种基于稀疏图模型的摊还推理方法，通过局部信用分配和离策略训练加快了训练速度。

    

    我们提出了一种新的算法，用于稀疏概率图模型（PGMs）的摊还推理，我们称之为Delta-AI。我们的方法基于这样的观察：当PGM中的变量采样被视为一个代理人采取的动作序列时，PGM的稀疏性使得代理人的策略学习目标能够进行局部信用分配。这导致了一个局部约束，可以转化为类似生成流网络（GFlowNets）中的局部损失，从而实现了离策略训练，但避免了每个参数更新需要实例化所有随机变量的需求，从而大大加快了训练速度。Delta-AI目标与一个可计算的学习采样器中的变量给定其马尔可夫毯子的条件分布相匹配，该采样器的结构类似于贝叶斯网络，在目标PGM下具有相同的条件分布。因此，训练后的采样器可以恢复感兴趣变量的边际分布和条件分布。

    We present a new algorithm for amortized inference in sparse probabilistic graphical models (PGMs), which we call $\Delta$-amortized inference ($\Delta$-AI). Our approach is based on the observation that when the sampling of variables in a PGM is seen as a sequence of actions taken by an agent, sparsity of the PGM enables local credit assignment in the agent's policy learning objective. This yields a local constraint that can be turned into a local loss in the style of generative flow networks (GFlowNets) that enables off-policy training but avoids the need to instantiate all the random variables for each parameter update, thus speeding up training considerably. The $\Delta$-AI objective matches the conditional distribution of a variable given its Markov blanket in a tractable learned sampler, which has the structure of a Bayesian network, with the same conditional distribution under the target PGM. As such, the trained sampler recovers marginals and conditional distributions of intere
    
[^4]: 量子经济的势能优势

    Potential Energy Advantage of Quantum Economy. (arXiv:2308.08025v1 [quant-ph])

    [http://arxiv.org/abs/2308.08025](http://arxiv.org/abs/2308.08025)

    量子计算在能源效率方面具有优势，并且能够在盈利和能源效率上超越经典计算。这使得量子计算成为计算行业更可持续的选择。

    

    随着大规模机器学习模型和语言模型的广泛部署，能源成本越来越关键。对于提供计算服务的公司来说，低能耗对于市场增长和政府法规来说都非常重要。本文研究了量子计算与经典计算之间的能源优势。我们在能源效率的背景下重新定义优势，与仅基于计算复杂性的传统量子优势不同。通过一个以能量使用为约束条件的Cournot竞争模型，我们证明量子计算公司在Nash均衡点上在盈利能力和能源效率方面都能超越经典对手。因此，量子计算可能代表计算行业更可持续的发展路径。此外，我们发现量子计算经济的能源利益取决于大规模计算。

    Energy cost is increasingly crucial in the modern computing industry with the wide deployment of large-scale machine learning models and language models. For the firms that provide computing services, low energy consumption is important both from the perspective of their own market growth and the government's regulations. In this paper, we study the energy benefits of quantum computing vis-a-vis classical computing. Deviating from the conventional notion of quantum advantage based solely on computational complexity, we redefine advantage in an energy efficiency context. Through a Cournot competition model constrained by energy usage, we demonstrate quantum computing firms can outperform classical counterparts in both profitability and energy efficiency at Nash equilibrium. Therefore quantum computing may represent a more sustainable pathway for the computing industry. Moreover, we discover that the energy benefits of quantum computing economies are contingent on large-scale computation
    
[^5]: 深度图核点过程

    Deep graph kernel point processes. (arXiv:2306.11313v1 [stat.ML])

    [http://arxiv.org/abs/2306.11313](http://arxiv.org/abs/2306.11313)

    本文提出了一种基于潜在图拓扑的图点过程方法，并开发了一种新颖的深度图核来描述事件之间的触发和抑制效应，该方法在合成和实际数据集上具有优越性。

    

    点过程模型广泛用于分析图中异步事件，反映不同类型事件之间的相互影响。预测未来事件的时间和类型是一项关键任务，并且图的大小和拓扑结构增加了问题的难度。最近的神经点过程模型揭示了捕捉复杂的事件类别之间依赖关系的可能性。然而，这些方法在每个目标事件类型的强度计算中使用了包括所有事件类别在内的未经滤波的事件记录。在本文中，我们提出了一种基于潜在图拓扑的图点过程方法。对应的无向图具有代表事件类别的节点和表示潜在贡献关系的边。然后，我们开发了一种新颖的深度图核来描述事件之间的触发和抑制效应。本质影响结构通过图神经网络-based的局部邻域信息聚合进行了融合。我们在合成和实际数据集上展示了我们提出的方法比最先进的模型更具优越性。

    Point process models are widely used to analyze asynchronous events occurring within a graph that reflect how different types of events influence one another. Predicting future events' times and types is a crucial task, and the size and topology of the graph add to the challenge of the problem. Recent neural point process models unveil the possibility of capturing intricate inter-event-category dependencies. However, such methods utilize an unfiltered history of events, including all event categories in the intensity computation for each target event type. In this work, we propose a graph point process method where event interactions occur based on a latent graph topology. The corresponding undirected graph has nodes representing event categories and edges indicating potential contribution relationships. We then develop a novel deep graph kernel to characterize the triggering and inhibiting effects between events. The intrinsic influence structures are incorporated via the graph neural
    

