# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [UniAP: Unifying Inter- and Intra-Layer Automatic Parallelism by Mixed Integer Quadratic Programming](https://arxiv.org/abs/2307.16375) | UniAP是一种新型的自动并行化方法，通过混合整数二次规划统一跨层和内层的自动并行化。与现有方法相比，UniAP在吞吐量方面表现更好，并且减少了策略优化时间。 |
| [^2] | [Incentive Allocation in Vertical Federated Learning Based on Bankruptcy Problem.](http://arxiv.org/abs/2307.03515) | 本文提出了一种基于破产问题的方法来解决垂直联邦学习中激励分配的挑战，以确保公平性和稳定性。 |

# 详细

[^1]: UniAP: 通过混合整数二次规划统一跨层和内层自动并行化

    UniAP: Unifying Inter- and Intra-Layer Automatic Parallelism by Mixed Integer Quadratic Programming

    [https://arxiv.org/abs/2307.16375](https://arxiv.org/abs/2307.16375)

    UniAP是一种新型的自动并行化方法，通过混合整数二次规划统一跨层和内层的自动并行化。与现有方法相比，UniAP在吞吐量方面表现更好，并且减少了策略优化时间。

    

    分布式学习常用于训练深度学习模型，特别是大型模型。在分布式学习中，手动并行化方法需要大量人力，并且灵活性有限。因此，最近提出了自动并行化方法来自动化并行策略优化过程。现有的自动并行化方法存在次优解的问题，因为它们不会同时优化跨层并行化和内层并行化这两个类别的并行策略。在本文中，我们提出了一种名为UniAP的新型自动并行化方法，通过混合整数二次规划统一跨层和内层的自动并行化。据我们所知，UniAP是第一种能够同时优化这两个类别的并行策略以求得最优解的并行化方法。实验结果表明，UniAP在吞吐量方面胜过了最先进的方法，提高了最多1.71倍，并减少了策略优化的时间。

    Distributed learning is commonly used for training deep learning models, especially large models. In distributed learning, manual parallelism (MP) methods demand considerable human effort and have limited flexibility. Hence, automatic parallelism (AP) methods have recently been proposed for automating the parallel strategy optimization process. Existing AP methods suffer from sub-optimal solutions because they do not jointly optimize the two categories of parallel strategies (i.e., inter-layer parallelism and intra-layer parallelism). In this paper, we propose a novel AP method called UniAP, which unifies inter- and intra-layer automatic parallelism by mixed integer quadratic programming. To the best of our knowledge, UniAP is the first parallel method that can jointly optimize the two categories of parallel strategies to find an optimal solution. Experimental results show that UniAP outperforms state-of-the-art methods by up to 1.71$\times$ in throughput and reduces strategy optimizat
    
[^2]: 基于破产问题的垂直联邦学习中的激励分配

    Incentive Allocation in Vertical Federated Learning Based on Bankruptcy Problem. (arXiv:2307.03515v1 [cs.LG])

    [http://arxiv.org/abs/2307.03515](http://arxiv.org/abs/2307.03515)

    本文提出了一种基于破产问题的方法来解决垂直联邦学习中激励分配的挑战，以确保公平性和稳定性。

    

    垂直联邦学习（VFL）是一种有前景的方法，用于合作训练在不同参与方之间垂直划分的私有数据的机器学习模型。在VFL设置中，理想情况下，主动方（拥有带标签样本特征的参与方）通过与某些被动方（拥有相同样本但没有标签的额外特征的参与方）合作，在保护隐私的情况下改进其机器学习模型。然而，激励被动方参与VFL可能具有挑战性。本文重点研究了基于被动方在VFL过程中的贡献来为他们分配激励的问题。我们将这个问题定义为核心游戏论概念的一种变体——破产问题，并使用塔木德划分规则来解决它。我们在合成和真实数据集上评估了我们提出的方法，并展示它确保了激励的公平性和稳定性。

    Vertical federated learning (VFL) is a promising approach for collaboratively training machine learning models using private data partitioned vertically across different parties. Ideally in a VFL setting, the active party (party possessing features of samples with labels) benefits by improving its machine learning model through collaboration with some passive parties (parties possessing additional features of the same samples without labels) in a privacy preserving manner. However, motivating passive parties to participate in VFL can be challenging. In this paper, we focus on the problem of allocating incentives to the passive parties by the active party based on their contributions to the VFL process. We formulate this problem as a variant of the Nucleolus game theory concept, known as the Bankruptcy Problem, and solve it using the Talmud's division rule. We evaluate our proposed method on synthetic and real-world datasets and show that it ensures fairness and stability in incentive a
    

