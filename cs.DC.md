# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Training Machine Learning models at the Edge: A Survey](https://arxiv.org/abs/2403.02619) | 这项调研深入探讨了边缘学习(EL)中优化机器学习模型训练的各种方法和方法论，旨在综合现有知识，识别挑战，并突出未来趋势。 |
| [^2] | [Federated Offline Policy Learning with Heterogeneous Observational Data.](http://arxiv.org/abs/2305.12407) | 本文提出了一种基于异构数据源的联邦政策学习算法，该算法基于本地策略聚合的方法，使用双重稳健线下策略评估和学习策略进行训练，可以在不交换原始数据的情况下学习个性化决策政策。我们建立了全局和局部后悔上限的理论模型，并用实验结果支持了理论发现。 |

# 详细

[^1]: 在边缘进行机器学习模型训练：一项调研

    Training Machine Learning models at the Edge: A Survey

    [https://arxiv.org/abs/2403.02619](https://arxiv.org/abs/2403.02619)

    这项调研深入探讨了边缘学习(EL)中优化机器学习模型训练的各种方法和方法论，旨在综合现有知识，识别挑战，并突出未来趋势。

    

    边缘计算(EC)近年来获得了显著关注，通过在边缘集成人工智能(AI)能力，承诺提高效率。虽然主要关注点在边缘部署和推断机器学习(ML)模型，但训练方面仍然较少被探讨。这项调研深入探讨了边缘学习(EL)，特别是在边缘优化ML模型训练的方面。其目标是全面探讨EL中的不同方法和方法论，综合现有知识，识别挑战，并突出未来趋势。利用Scopus的高级搜索，确定了关于EL的相关文献，显示了研究工作在分布式学习方法方面的聚焦，特别是联邦学习(FL)。此调研还提供了一个比较用于优化边缘学习的ML的技术的指南，以及对不同框架的探索。

    arXiv:2403.02619v1 Announce Type: new  Abstract: Edge Computing (EC) has gained significant traction in recent years, promising enhanced efficiency by integrating Artificial Intelligence (AI) capabilities at the edge. While the focus has primarily been on the deployment and inference of Machine Learning (ML) models at the edge, the training aspect remains less explored. This survey delves into Edge Learning (EL), specifically the optimization of ML model training at the edge. The objective is to comprehensively explore diverse approaches and methodologies in EL, synthesize existing knowledge, identify challenges, and highlight future trends. Utilizing Scopus' advanced search, relevant literature on EL was identified, revealing a concentration of research efforts in distributed learning methods, particularly Federated Learning (FL). This survey further provides a guideline for comparing techniques used to optimize ML for edge learning, along with an exploration of different frameworks, 
    
[^2]: 异构观测数据下的联邦弱化政策学习

    Federated Offline Policy Learning with Heterogeneous Observational Data. (arXiv:2305.12407v1 [cs.LG])

    [http://arxiv.org/abs/2305.12407](http://arxiv.org/abs/2305.12407)

    本文提出了一种基于异构数据源的联邦政策学习算法，该算法基于本地策略聚合的方法，使用双重稳健线下策略评估和学习策略进行训练，可以在不交换原始数据的情况下学习个性化决策政策。我们建立了全局和局部后悔上限的理论模型，并用实验结果支持了理论发现。

    

    本文考虑了基于异构数据源的观测数据学习个性化决策政策的问题。此外，我们在联邦设置中研究了这个问题，其中中央服务器旨在在分布在异构源上的数据上学习一个政策，而不交换它们的原始数据。我们提出了一个联邦政策学习算法，它基于使用双重稳健线下策略评估和学习策略训练的本地策略聚合的方法。我们提供了一种新的后悔分析方法来确立对全局后悔概念的有限样本上界，这个全局后悔概念跨越了客户端的分布。此外，我们针对每个单独的客户端建立了相应的局部后悔上界，该上界由相对于所有其他客户端的分布变化特征性地描述。我们用实验结果支持我们的理论发现。我们的分析和实验提供了异构客户端参与联邦学习的价值洞察。

    We consider the problem of learning personalized decision policies on observational data from heterogeneous data sources. Moreover, we examine this problem in the federated setting where a central server aims to learn a policy on the data distributed across the heterogeneous sources without exchanging their raw data. We present a federated policy learning algorithm based on aggregation of local policies trained with doubly robust offline policy evaluation and learning strategies. We provide a novel regret analysis for our approach that establishes a finite-sample upper bound on a notion of global regret across a distribution of clients. In addition, for any individual client, we establish a corresponding local regret upper bound characterized by the presence of distribution shift relative to all other clients. We support our theoretical findings with experimental results. Our analysis and experiments provide insights into the value of heterogeneous client participation in federation fo
    

