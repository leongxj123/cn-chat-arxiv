# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Low-Cost Privacy-Aware Decentralized Learning](https://arxiv.org/abs/2403.11795) | ZIP-DL是一种低成本的隐私感知去中心化学习算法，通过向每个模型更新添加相关噪声，在保护隐私的同时实现了较高的模型准确性，具有较好的收敛速度和隐私保证。 |
| [^2] | [FedPop: Federated Population-based Hyperparameter Tuning.](http://arxiv.org/abs/2308.08634) | FedPop是一种用于解决联邦学习中超参数调优问题的新算法，它采用基于人口的进化算法来优化客户端和服务器上的超参数。 |

# 详细

[^1]: 低成本隐私感知去中心化学习

    Low-Cost Privacy-Aware Decentralized Learning

    [https://arxiv.org/abs/2403.11795](https://arxiv.org/abs/2403.11795)

    ZIP-DL是一种低成本的隐私感知去中心化学习算法，通过向每个模型更新添加相关噪声，在保护隐私的同时实现了较高的模型准确性，具有较好的收敛速度和隐私保证。

    

    本文介绍了一种新颖的隐私感知去中心化学习（DL）算法ZIP-DL，该算法依赖于在模型训练过程中向每个模型更新添加相关噪声。这种技术确保了由于其相关性，在聚合过程中添加的噪声几乎相互抵消，从而最小化对模型准确性的影响。此外，ZIP-DL不需要多次通信轮进行噪声抵消，解决了隐私保护与通信开销之间的常见权衡。我们为收敛速度和隐私保证提供了理论保证，从而使ZIP-DL可应用于实际场景。我们的广泛实验研究表明，ZIP-DL在易受攻击性和准确性之间取得了最佳权衡。特别是，与基线DL相比，ZIP-DL（i）将可追踪攻击的有效性降低了多达52个点，（ii）准确性提高了高达37个百分点。

    arXiv:2403.11795v1 Announce Type: new  Abstract: This paper introduces ZIP-DL, a novel privacy-aware decentralized learning (DL) algorithm that relies on adding correlated noise to each model update during the model training process. This technique ensures that the added noise almost neutralizes itself during the aggregation process due to its correlation, thus minimizing the impact on model accuracy. In addition, ZIP-DL does not require multiple communication rounds for noise cancellation, addressing the common trade-off between privacy protection and communication overhead. We provide theoretical guarantees for both convergence speed and privacy guarantees, thereby making ZIP-DL applicable to practical scenarios. Our extensive experimental study shows that ZIP-DL achieves the best trade-off between vulnerability and accuracy. In particular, ZIP-DL (i) reduces the effectiveness of a linkability attack by up to 52 points compared to baseline DL, and (ii) achieves up to 37 more accuracy
    
[^2]: FedPop: 联邦式基于人口的超参数调优

    FedPop: Federated Population-based Hyperparameter Tuning. (arXiv:2308.08634v1 [cs.LG])

    [http://arxiv.org/abs/2308.08634](http://arxiv.org/abs/2308.08634)

    FedPop是一种用于解决联邦学习中超参数调优问题的新算法，它采用基于人口的进化算法来优化客户端和服务器上的超参数。

    

    联邦学习（FL）是一种分布式机器学习（ML）范式，多个客户端在不集中本地数据的情况下共同训练ML模型。与传统的ML流程类似，FL中的客户端本地优化和服务器聚合过程对超参数（HP）的选择非常敏感。尽管在集中式ML中对调优HP进行了广泛研究，但将这些方法应用于FL时会产生次优结果。这主要是因为它们的“调优后训练”框架对于计算能力有限的FL不合适。虽然一些方法已经提出用于FL中的HP调优，但这些方法仅限于客户端本地更新的HP。在这项工作中，我们提出了一种名为联邦式基于人口的超参数调优（FedPop）的新型HP调优算法，以解决这个重要但具有挑战性的问题。FedPop采用基于人口的进化算法来优化HP，此算法适用于客户端和服务器上的各种HP类型。

    Federated Learning (FL) is a distributed machine learning (ML) paradigm, in which multiple clients collaboratively train ML models without centralizing their local data. Similar to conventional ML pipelines, the client local optimization and server aggregation procedure in FL are sensitive to the hyperparameter (HP) selection. Despite extensive research on tuning HPs for centralized ML, these methods yield suboptimal results when employed in FL. This is mainly because their "training-after-tuning" framework is unsuitable for FL with limited client computation power. While some approaches have been proposed for HP-Tuning in FL, they are limited to the HPs for client local updates. In this work, we propose a novel HP-tuning algorithm, called Federated Population-based Hyperparameter Tuning (FedPop), to address this vital yet challenging problem. FedPop employs population-based evolutionary algorithms to optimize the HPs, which accommodates various HP types at both client and server sides
    

