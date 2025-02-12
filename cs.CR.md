# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DP-SGD with weight clipping.](http://arxiv.org/abs/2310.18001) | 本研究提出了一种带权重剪裁的差分隐私梯度下降方法，通过利用公共信息对全局模型进行改进，获得更精确的灵敏度界限和噪声水平调整，提供了更好的差分隐私保证。 |
| [^2] | [Exploring the Bitcoin Mesoscale.](http://arxiv.org/abs/2307.14409) | 本文研究了比特币用户网络的中小尺度结构特性，并发现其具有核心-外围结构和蝴蝶结构。此外，我们的研究还发现，BUN结构组织的演化与泡沫存在相关的波动有关，进一步证实了结构量与价格变动之间的相互作用。 |
| [^3] | [FairDP: Certified Fairness with Differential Privacy.](http://arxiv.org/abs/2305.16474) | FairDP是一种同时确保差分隐私和公平性的新型机制，通过独立为不同的个体群体训练模型，在训练过程中逐步整合来自群体模型的知识，制定综合模型以平衡隐私、效用和公平性的下游任务。相比现有方法，FairDP展示了更好的模型效益、隐私和公平性的权衡。 |

# 详细

[^1]: 带权重剪裁的差分隐私梯度下降方法

    DP-SGD with weight clipping. (arXiv:2310.18001v1 [cs.LG])

    [http://arxiv.org/abs/2310.18001](http://arxiv.org/abs/2310.18001)

    本研究提出了一种带权重剪裁的差分隐私梯度下降方法，通过利用公共信息对全局模型进行改进，获得更精确的灵敏度界限和噪声水平调整，提供了更好的差分隐私保证。

    

    最近，由于深度神经网络和其他依赖于目标函数优化的方法的高度流行，以及对数据隐私的关注，差分隐私梯度下降方法引起了极大的兴趣。为了在提供最小噪声的情况下实现差分隐私保证，能够准确地限制参与者将观察到的信息的灵敏度非常重要。在本研究中，我们提出了一种新颖的方法，弥补了传统梯度剪裁产生的偏差。通过利用关于当前全局模型及其在搜索领域中位置的公共信息，我们可以获得改进的梯度界限，从而实现更精确的灵敏度确定和噪声水平调整。我们扩展了最先进的算法，提供了更好的差分隐私保证，需要更少的噪声，并进行了实证评估。

    Recently, due to the popularity of deep neural networks and other methods whose training typically relies on the optimization of an objective function, and due to concerns for data privacy, there is a lot of interest in differentially private gradient descent methods. To achieve differential privacy guarantees with a minimum amount of noise, it is important to be able to bound precisely the sensitivity of the information which the participants will observe. In this study, we present a novel approach that mitigates the bias arising from traditional gradient clipping. By leveraging public information concerning the current global model and its location within the search domain, we can achieve improved gradient bounds, leading to enhanced sensitivity determinations and refined noise level adjustments. We extend the state of the art algorithms, present improved differential privacy guarantees requiring less noise and present an empirical evaluation.
    
[^2]: 探索比特币的中小尺度结构

    Exploring the Bitcoin Mesoscale. (arXiv:2307.14409v1 [q-fin.ST])

    [http://arxiv.org/abs/2307.14409](http://arxiv.org/abs/2307.14409)

    本文研究了比特币用户网络的中小尺度结构特性，并发现其具有核心-外围结构和蝴蝶结构。此外，我们的研究还发现，BUN结构组织的演化与泡沫存在相关的波动有关，进一步证实了结构量与价格变动之间的相互作用。

    

    比特币交易历史的开放可用性为以前所未有的细节水平研究该系统提供了可能性。本文致力于分析比特币用户网络（BUN）在其整个历史（即从2009年到2017年）中的中小尺度结构属性。从我们的分析中可以看出，BUN具有核心-外围结构，更深入的分析揭示了一定程度的蝴蝶结构（即具有强连通分量、IN分量和OUT分量以及一些附着在IN分量上的触须）。有趣的是，BUN结构组织的演化经历了与泡沫存在相关的波动，即在整个比特币历史上观察到的价格激增和下跌阶段：因此，我们的结果进一步证实了先前分析中观察到的结构量和价格变动之间的相互作用。

    The open availability of the entire history of the Bitcoin transactions opens up the possibility to study this system at an unprecedented level of detail. This contribution is devoted to the analysis of the mesoscale structural properties of the Bitcoin User Network (BUN), across its entire history (i.e. from 2009 to 2017). What emerges from our analysis is that the BUN is characterized by a core-periphery structure a deeper analysis of which reveals a certain degree of bow-tieness (i.e. the presence of a Strongly-Connected Component, an IN- and an OUT-component together with some tendrils attached to the IN-component). Interestingly, the evolution of the BUN structural organization experiences fluctuations that seem to be correlated with the presence of bubbles, i.e. periods of price surge and decline observed throughout the entire Bitcoin history: our results, thus, further confirm the interplay between structural quantities and price movements observed in previous analyses.
    
[^3]: FairDP: 具有差分隐私认证的公平性保障

    FairDP: Certified Fairness with Differential Privacy. (arXiv:2305.16474v1 [cs.LG])

    [http://arxiv.org/abs/2305.16474](http://arxiv.org/abs/2305.16474)

    FairDP是一种同时确保差分隐私和公平性的新型机制，通过独立为不同的个体群体训练模型，在训练过程中逐步整合来自群体模型的知识，制定综合模型以平衡隐私、效用和公平性的下游任务。相比现有方法，FairDP展示了更好的模型效益、隐私和公平性的权衡。

    

    本文介绍了一种名为FairDP的新型机制，旨在同时确保差分隐私(DP)和公平性。FairDP通过独立为不同的个体群体训练模型，在使用组特定的剪裁项来评估和限制DP的差异影响的同时操作。在训练过程中，该机制逐步整合来自群体模型的知识，制定综合模型以平衡隐私、效用和公平性的下游任务。广泛的理论和实证分析验证了FairDP的功效，与现有方法相比，展示了更好的模型效益、隐私和公平性的权衡。

    This paper introduces FairDP, a novel mechanism designed to simultaneously ensure differential privacy (DP) and fairness. FairDP operates by independently training models for distinct individual groups, using group-specific clipping terms to assess and bound the disparate impacts of DP. Throughout the training process, the mechanism progressively integrates knowledge from group models to formulate a comprehensive model that balances privacy, utility, and fairness in downstream tasks. Extensive theoretical and empirical analyses validate the efficacy of FairDP, demonstrating improved trade-offs between model utility, privacy, and fairness compared with existing methods.
    

