# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improve Generalization Ability of Deep Wide Residual Network with A Suitable Scaling Factor](https://arxiv.org/abs/2403.04545) | 通过在深广残差网络中使用适当的缩放因子，可以提高泛化能力，即使允许缩放因子随深度减小，也可以实现最小最大速率。 |
| [^2] | [PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation](https://arxiv.org/abs/2402.04355) | PQMass是一种使用概率质量估计来评估生成模型质量的全面方法，能够直接处理高维数据，不依赖于假设或训练其他模型。 |
| [^3] | [SUB-PLAY: Adversarial Policies against Partially Observed Multi-Agent Reinforcement Learning Systems](https://arxiv.org/abs/2402.03741) | 该研究首次揭示了攻击者在多智能体竞争环境中即使受限于受害者的部分观测也能生成对抗策略的能力。 |
| [^4] | [Rates of Convergence in the Central Limit Theorem for Markov Chains, with an Application to TD Learning](https://arxiv.org/abs/2401.15719) | 本研究证明了一个非渐近的中心极限定理，并通过应用于TD学习，展示了其实际应用的可行性。 |
| [^5] | [PhyloGFN: Phylogenetic inference with generative flow networks.](http://arxiv.org/abs/2310.08774) | PhyloGFN是一种基于生成流网络的系统发育推断方法，通过采样复杂的组合结构，能够产生多样且高质量的进化假设，并在边缘似然估计方面具有竞争力。 |
| [^6] | [Amortizing intractable inference in large language models.](http://arxiv.org/abs/2310.04363) | 本论文提出了一种使用摊销的贝叶斯推理从难以处理的后验分布中进行抽样的方法，并利用生成流网络来实现大规模语言模型的微调，从而解决了在这些模型中处理推理问题的限制。 |
| [^7] | [Expected flow networks in stochastic environments and two-player zero-sum games.](http://arxiv.org/abs/2310.02779) | 该论文提出了预期流网络（EFlowNets）和对抗流网络（AFlowNets），分别应用于随机环境和双人零和游戏中。在随机任务中，EFlowNets表现优于其他方法，在双人游戏中，AFlowNets在自我对弈中找到了80%以上的最佳动作，并在竞赛中超过了AlphaZero。 |
| [^8] | [Delta-AI: Local objectives for amortized inference in sparse graphical models.](http://arxiv.org/abs/2310.02423) | Delta-AI算法提出了一种基于稀疏图模型的摊还推理方法，通过局部信用分配和离策略训练加快了训练速度。 |
| [^9] | [Potential Energy Advantage of Quantum Economy.](http://arxiv.org/abs/2308.08025) | 量子计算在能源效率方面具有优势，并且能够在盈利和能源效率上超越经典计算。这使得量子计算成为计算行业更可持续的选择。 |
| [^10] | [Deep graph kernel point processes.](http://arxiv.org/abs/2306.11313) | 本文提出了一种基于潜在图拓扑的图点过程方法，并开发了一种新颖的深度图核来描述事件之间的触发和抑制效应，该方法在合成和实际数据集上具有优越性。 |

# 详细

[^1]: 通过合适的缩放因子提高深广残差网络的泛化能力

    Improve Generalization Ability of Deep Wide Residual Network with A Suitable Scaling Factor

    [https://arxiv.org/abs/2403.04545](https://arxiv.org/abs/2403.04545)

    通过在深广残差网络中使用适当的缩放因子，可以提高泛化能力，即使允许缩放因子随深度减小，也可以实现最小最大速率。

    

    深残差神经网络（ResNets）在广泛的实际应用中取得了显著的成功。在本文中，我们确定了深广残差网络中残差分支上的合适缩放因子（用$\alpha$表示），以实现良好的泛化能力。我们展示了如果$\alpha$是一个常数，由残差神经切向核（RNTK）引发的函数类在深度趋近无穷时是渐近不可学习的。我们还强调了一个令人惊讶的现象：即使我们允许$\alpha$随着深度$L$的增加而减小，退化现象仍可能发生。然而，当$\alpha$与$L$快速减小时，使用深层RNTK进行核回归，并且在早停条件下可以实现最小最大速率，前提是目标回归函数落在与无限深度RNTK相关的再生核希尔伯特空间中。我们对合成数据和真实分类任务进行了模拟研究。

    arXiv:2403.04545v1 Announce Type: new  Abstract: Deep Residual Neural Networks (ResNets) have demonstrated remarkable success across a wide range of real-world applications. In this paper, we identify a suitable scaling factor (denoted by $\alpha$) on the residual branch of deep wide ResNets to achieve good generalization ability. We show that if $\alpha$ is a constant, the class of functions induced by Residual Neural Tangent Kernel (RNTK) is asymptotically not learnable, as the depth goes to infinity. We also highlight a surprising phenomenon: even if we allow $\alpha$ to decrease with increasing depth $L$, the degeneration phenomenon may still occur. However, when $\alpha$ decreases rapidly with $L$, the kernel regression with deep RNTK with early stopping can achieve the minimax rate provided that the target regression function falls in the reproducing kernel Hilbert space associated with the infinite-depth RNTK. Our simulation studies on synthetic data and real classification task
    
[^2]: PQMass: 使用概率质量估计的生成模型质量的概率评估

    PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation

    [https://arxiv.org/abs/2402.04355](https://arxiv.org/abs/2402.04355)

    PQMass是一种使用概率质量估计来评估生成模型质量的全面方法，能够直接处理高维数据，不依赖于假设或训练其他模型。

    

    我们提出了一种全面的基于样本的方法来评估生成模型的质量。所提出的方法能够估计两个样本集合来自同一分布的概率，为评估单个生成模型的性能或比较在同一数据集上训练的多个竞争模型提供了一个统计上严格的方法。该比较可以通过将空间划分为非重叠的区域并比较每个区域中的数据样本数量来进行。该方法仅需要生成模型和测试数据的样本。它能够直接处理高维数据，无需降维。显著的是，该方法不依赖于关于真实分布密度的假设，并且不依赖于训练或拟合任何辅助模型。相反，它着重于近似计算密度的积分（概率质量）。

    We propose a comprehensive sample-based method for assessing the quality of generative models. The proposed approach enables the estimation of the probability that two sets of samples are drawn from the same distribution, providing a statistically rigorous method for assessing the performance of a single generative model or the comparison of multiple competing models trained on the same dataset. This comparison can be conducted by dividing the space into non-overlapping regions and comparing the number of data samples in each region. The method only requires samples from the generative model and the test data. It is capable of functioning directly on high-dimensional data, obviating the need for dimensionality reduction. Significantly, the proposed method does not depend on assumptions regarding the density of the true distribution, and it does not rely on training or fitting any auxiliary models. Instead, it focuses on approximating the integral of the density (probability mass) acros
    
[^3]: SUB-PLAY：针对部分观测的多智能体强化学习系统的对抗策略

    SUB-PLAY: Adversarial Policies against Partially Observed Multi-Agent Reinforcement Learning Systems

    [https://arxiv.org/abs/2402.03741](https://arxiv.org/abs/2402.03741)

    该研究首次揭示了攻击者在多智能体竞争环境中即使受限于受害者的部分观测也能生成对抗策略的能力。

    

    最近在多智能体强化学习（MARL）领域取得的进展为无人机的群体控制、机械臂的协作操纵以及多目标包围等开辟了广阔的应用前景。然而，在MARL部署过程中存在潜在的安全威胁需要更多关注和深入调查。最近的研究表明，攻击者可以迅速利用受害者的漏洞生成对抗策略，导致受害者在特定任务中失败。例如，将超人级别的围棋AI的获胜率降低到约20%。这些研究主要关注两人竞争环境，并假设攻击者具有完整的全局状态观测。

    Recent advances in multi-agent reinforcement learning (MARL) have opened up vast application prospects, including swarm control of drones, collaborative manipulation by robotic arms, and multi-target encirclement. However, potential security threats during the MARL deployment need more attention and thorough investigation. Recent researches reveal that an attacker can rapidly exploit the victim's vulnerabilities and generate adversarial policies, leading to the victim's failure in specific tasks. For example, reducing the winning rate of a superhuman-level Go AI to around 20%. They predominantly focus on two-player competitive environments, assuming attackers possess complete global state observation.   In this study, we unveil, for the first time, the capability of attackers to generate adversarial policies even when restricted to partial observations of the victims in multi-agent competitive environments. Specifically, we propose a novel black-box attack (SUB-PLAY), which incorporate
    
[^4]: 关于马尔可夫链中心极限定理的收敛速度，及其在TD学习中的应用

    Rates of Convergence in the Central Limit Theorem for Markov Chains, with an Application to TD Learning

    [https://arxiv.org/abs/2401.15719](https://arxiv.org/abs/2401.15719)

    本研究证明了一个非渐近的中心极限定理，并通过应用于TD学习，展示了其实际应用的可行性。

    

    我们使用Stein方法证明了一个非渐近的、矢量值鞅差的中心极限定理，并利用泊松方程将结果推广到马尔可夫链的函数。然后我们展示了这些结果可以应用于建立基于平均的非渐近的TD学习的中心极限定理。

    We prove a non-asymptotic central limit theorem for vector-valued martingale differences using Stein's method, and use Poisson's equation to extend the result to functions of Markov Chains. We then show that these results can be applied to establish a non-asymptotic central limit theorem for Temporal Difference (TD) learning with averaging.
    
[^5]: PhyloGFN: 基于生成流网络的系统发育推断

    PhyloGFN: Phylogenetic inference with generative flow networks. (arXiv:2310.08774v1 [q-bio.PE])

    [http://arxiv.org/abs/2310.08774](http://arxiv.org/abs/2310.08774)

    PhyloGFN是一种基于生成流网络的系统发育推断方法，通过采样复杂的组合结构，能够产生多样且高质量的进化假设，并在边缘似然估计方面具有竞争力。

    

    系统发育学是计算生物学的一个分支，研究生物实体之间的进化关系。尽管有着悠久的历史和众多应用，但从序列数据推断系统发育树仍然具有挑战性：树空间的高复杂性对当前的组合和概率技术构成了重要障碍。在本文中，我们采用生成流网络（GFlowNets）的框架来解决系统发育学中的两个核心问题：基于最简原则的和贝叶斯的系统发育推断。由于GFlowNets适用于采样复杂的组合结构，它们是探索和采样树拓扑和进化距离的多模态后验分布的自然选择。我们证明了我们的摊还后验采样器PhyloGFN在真实基准数据集上产生多样且高质量的进化假设。PhyloGFN在边缘似然估计方面与之前的工作相比具有竞争力。

    Phylogenetics is a branch of computational biology that studies the evolutionary relationships among biological entities. Its long history and numerous applications notwithstanding, inference of phylogenetic trees from sequence data remains challenging: the high complexity of tree space poses a significant obstacle for the current combinatorial and probabilistic techniques. In this paper, we adopt the framework of generative flow networks (GFlowNets) to tackle two core problems in phylogenetics: parsimony-based and Bayesian phylogenetic inference. Because GFlowNets are well-suited for sampling complex combinatorial structures, they are a natural choice for exploring and sampling from the multimodal posterior distribution over tree topologies and evolutionary distances. We demonstrate that our amortized posterior sampler, PhyloGFN, produces diverse and high-quality evolutionary hypotheses on real benchmark datasets. PhyloGFN is competitive with prior works in marginal likelihood estimat
    
[^6]: 在大规模语言模型中摊销难以处理的推理问题

    Amortizing intractable inference in large language models. (arXiv:2310.04363v1 [cs.LG])

    [http://arxiv.org/abs/2310.04363](http://arxiv.org/abs/2310.04363)

    本论文提出了一种使用摊销的贝叶斯推理从难以处理的后验分布中进行抽样的方法，并利用生成流网络来实现大规模语言模型的微调，从而解决了在这些模型中处理推理问题的限制。

    

    自回归的大规模语言模型通过下一个词条件分布来压缩其训练数据中的知识，这限制了对该知识的可处理查询仅限于从头到尾的自回归抽样。然而，许多感兴趣的任务，包括序列延续、填充和其他形式的受约束生成，都涉及从难以处理的后验分布中进行抽样。我们通过使用摊销的贝叶斯推理从这些难以处理的后验分布中进行抽样来解决这个限制。这种摊销通过通过寻求多样性的强化学习算法 - 生成流网络 (GFlowNets) 来微调 LLMs 实现。我们凭经验证明，LLM微调的这种分布匹配范式可以作为最大似然训练和奖励最大化策略优化的有效替代方法。作为一个重要应用，我们将思维链推理解释为潜变量建模问题，并证明了...

    Autoregressive large language models (LLMs) compress knowledge from their training data through next-token conditional distributions. This limits tractable querying of this knowledge to start-to-end autoregressive sampling. However, many tasks of interest -- including sequence continuation, infilling, and other forms of constrained generation -- involve sampling from intractable posterior distributions. We address this limitation by using amortized Bayesian inference to sample from these intractable posteriors. Such amortization is algorithmically achieved by fine-tuning LLMs via diversity-seeking reinforcement learning algorithms: generative flow networks (GFlowNets). We empirically demonstrate that this distribution-matching paradigm of LLM fine-tuning can serve as an effective alternative to maximum-likelihood training and reward-maximizing policy optimization. As an important application, we interpret chain-of-thought reasoning as a latent variable modeling problem and demonstrate 
    
[^7]: 随机环境中的预期流网络和双人零和游戏

    Expected flow networks in stochastic environments and two-player zero-sum games. (arXiv:2310.02779v1 [cs.LG])

    [http://arxiv.org/abs/2310.02779](http://arxiv.org/abs/2310.02779)

    该论文提出了预期流网络（EFlowNets）和对抗流网络（AFlowNets），分别应用于随机环境和双人零和游戏中。在随机任务中，EFlowNets表现优于其他方法，在双人游戏中，AFlowNets在自我对弈中找到了80%以上的最佳动作，并在竞赛中超过了AlphaZero。

    

    生成流网络（GFlowNets）是训练用于匹配给定分布的序列采样模型。GFlowNets已成功应用于各种结构对象生成任务，能够迅速采样出多样化且高回报的对象。我们提出了预期流网络（EFlowNets），将GFlowNets扩展到随机环境中。我们展示了EFlowNets在随机任务（如蛋白质设计）中优于其他GFlowNet方案。然后，我们将EFlowNets的概念扩展到对抗环境中，为双人零和游戏提出了对抗流网络（AFlowNets）。我们展示了AFlowNets通过自我对弈在Connect-4游戏中能找到80%以上的最佳动作，并在竞赛中优于AlphaZero。

    Generative flow networks (GFlowNets) are sequential sampling models trained to match a given distribution. GFlowNets have been successfully applied to various structured object generation tasks, sampling a diverse set of high-reward objects quickly. We propose expected flow networks (EFlowNets), which extend GFlowNets to stochastic environments. We show that EFlowNets outperform other GFlowNet formulations in stochastic tasks such as protein design. We then extend the concept of EFlowNets to adversarial environments, proposing adversarial flow networks (AFlowNets) for two-player zero-sum games. We show that AFlowNets learn to find above 80% of optimal moves in Connect-4 via self-play and outperform AlphaZero in tournaments.
    
[^8]: Delta-AI: 稀疏图模型的摊还推理中的局部目标

    Delta-AI: Local objectives for amortized inference in sparse graphical models. (arXiv:2310.02423v1 [cs.LG])

    [http://arxiv.org/abs/2310.02423](http://arxiv.org/abs/2310.02423)

    Delta-AI算法提出了一种基于稀疏图模型的摊还推理方法，通过局部信用分配和离策略训练加快了训练速度。

    

    我们提出了一种新的算法，用于稀疏概率图模型（PGMs）的摊还推理，我们称之为Delta-AI。我们的方法基于这样的观察：当PGM中的变量采样被视为一个代理人采取的动作序列时，PGM的稀疏性使得代理人的策略学习目标能够进行局部信用分配。这导致了一个局部约束，可以转化为类似生成流网络（GFlowNets）中的局部损失，从而实现了离策略训练，但避免了每个参数更新需要实例化所有随机变量的需求，从而大大加快了训练速度。Delta-AI目标与一个可计算的学习采样器中的变量给定其马尔可夫毯子的条件分布相匹配，该采样器的结构类似于贝叶斯网络，在目标PGM下具有相同的条件分布。因此，训练后的采样器可以恢复感兴趣变量的边际分布和条件分布。

    We present a new algorithm for amortized inference in sparse probabilistic graphical models (PGMs), which we call $\Delta$-amortized inference ($\Delta$-AI). Our approach is based on the observation that when the sampling of variables in a PGM is seen as a sequence of actions taken by an agent, sparsity of the PGM enables local credit assignment in the agent's policy learning objective. This yields a local constraint that can be turned into a local loss in the style of generative flow networks (GFlowNets) that enables off-policy training but avoids the need to instantiate all the random variables for each parameter update, thus speeding up training considerably. The $\Delta$-AI objective matches the conditional distribution of a variable given its Markov blanket in a tractable learned sampler, which has the structure of a Bayesian network, with the same conditional distribution under the target PGM. As such, the trained sampler recovers marginals and conditional distributions of intere
    
[^9]: 量子经济的势能优势

    Potential Energy Advantage of Quantum Economy. (arXiv:2308.08025v1 [quant-ph])

    [http://arxiv.org/abs/2308.08025](http://arxiv.org/abs/2308.08025)

    量子计算在能源效率方面具有优势，并且能够在盈利和能源效率上超越经典计算。这使得量子计算成为计算行业更可持续的选择。

    

    随着大规模机器学习模型和语言模型的广泛部署，能源成本越来越关键。对于提供计算服务的公司来说，低能耗对于市场增长和政府法规来说都非常重要。本文研究了量子计算与经典计算之间的能源优势。我们在能源效率的背景下重新定义优势，与仅基于计算复杂性的传统量子优势不同。通过一个以能量使用为约束条件的Cournot竞争模型，我们证明量子计算公司在Nash均衡点上在盈利能力和能源效率方面都能超越经典对手。因此，量子计算可能代表计算行业更可持续的发展路径。此外，我们发现量子计算经济的能源利益取决于大规模计算。

    Energy cost is increasingly crucial in the modern computing industry with the wide deployment of large-scale machine learning models and language models. For the firms that provide computing services, low energy consumption is important both from the perspective of their own market growth and the government's regulations. In this paper, we study the energy benefits of quantum computing vis-a-vis classical computing. Deviating from the conventional notion of quantum advantage based solely on computational complexity, we redefine advantage in an energy efficiency context. Through a Cournot competition model constrained by energy usage, we demonstrate quantum computing firms can outperform classical counterparts in both profitability and energy efficiency at Nash equilibrium. Therefore quantum computing may represent a more sustainable pathway for the computing industry. Moreover, we discover that the energy benefits of quantum computing economies are contingent on large-scale computation
    
[^10]: 深度图核点过程

    Deep graph kernel point processes. (arXiv:2306.11313v1 [stat.ML])

    [http://arxiv.org/abs/2306.11313](http://arxiv.org/abs/2306.11313)

    本文提出了一种基于潜在图拓扑的图点过程方法，并开发了一种新颖的深度图核来描述事件之间的触发和抑制效应，该方法在合成和实际数据集上具有优越性。

    

    点过程模型广泛用于分析图中异步事件，反映不同类型事件之间的相互影响。预测未来事件的时间和类型是一项关键任务，并且图的大小和拓扑结构增加了问题的难度。最近的神经点过程模型揭示了捕捉复杂的事件类别之间依赖关系的可能性。然而，这些方法在每个目标事件类型的强度计算中使用了包括所有事件类别在内的未经滤波的事件记录。在本文中，我们提出了一种基于潜在图拓扑的图点过程方法。对应的无向图具有代表事件类别的节点和表示潜在贡献关系的边。然后，我们开发了一种新颖的深度图核来描述事件之间的触发和抑制效应。本质影响结构通过图神经网络-based的局部邻域信息聚合进行了融合。我们在合成和实际数据集上展示了我们提出的方法比最先进的模型更具优越性。

    Point process models are widely used to analyze asynchronous events occurring within a graph that reflect how different types of events influence one another. Predicting future events' times and types is a crucial task, and the size and topology of the graph add to the challenge of the problem. Recent neural point process models unveil the possibility of capturing intricate inter-event-category dependencies. However, such methods utilize an unfiltered history of events, including all event categories in the intensity computation for each target event type. In this work, we propose a graph point process method where event interactions occur based on a latent graph topology. The corresponding undirected graph has nodes representing event categories and edges indicating potential contribution relationships. We then develop a novel deep graph kernel to characterize the triggering and inhibiting effects between events. The intrinsic influence structures are incorporated via the graph neural
    

