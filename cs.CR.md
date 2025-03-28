# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Evaluating Membership Inference Attacks and Defenses in Federated Learning](https://arxiv.org/abs/2402.06289) | 这篇论文评估了联邦学习中成员推断攻击和防御的情况。评估揭示了两个重要发现：多时序的模型信息有助于提高攻击的有效性，多空间的模型信息有助于提高攻击的效果。这篇论文还评估了两种防御机制的效用和隐私权衡。 |

# 详细

[^1]: 在联邦学习中评估成员推断攻击和防御

    Evaluating Membership Inference Attacks and Defenses in Federated Learning

    [https://arxiv.org/abs/2402.06289](https://arxiv.org/abs/2402.06289)

    这篇论文评估了联邦学习中成员推断攻击和防御的情况。评估揭示了两个重要发现：多时序的模型信息有助于提高攻击的有效性，多空间的模型信息有助于提高攻击的效果。这篇论文还评估了两种防御机制的效用和隐私权衡。

    

    成员推断攻击(MIAs)对于隐私保护的威胁在联邦学习中日益增长。半诚实的攻击者，例如服务器，可以根据观察到的模型信息确定一个特定样本是否属于目标客户端。本文对现有的MIAs和相应的防御策略进行了评估。我们对MIAs的评估揭示了两个重要发现。首先，结合多个通信轮次的模型信息(多时序)相比于利用单个时期的模型信息提高了MIAs的整体有效性。其次，在非目标客户端(Multi-spatial)中融入模型显著提高了MIAs的效果，特别是当客户端的数据是同质的时候。这凸显了在MIAs中考虑时序和空间模型信息的重要性。接下来，我们通过隐私-效用权衡评估了两种类型的防御机制对MIAs的有效性。

    Membership Inference Attacks (MIAs) pose a growing threat to privacy preservation in federated learning. The semi-honest attacker, e.g., the server, may determine whether a particular sample belongs to a target client according to the observed model information. This paper conducts an evaluation of existing MIAs and corresponding defense strategies. Our evaluation on MIAs reveals two important findings about the trend of MIAs. Firstly, combining model information from multiple communication rounds (Multi-temporal) enhances the overall effectiveness of MIAs compared to utilizing model information from a single epoch. Secondly, incorporating models from non-target clients (Multi-spatial) significantly improves the effectiveness of MIAs, particularly when the clients' data is homogeneous. This highlights the importance of considering the temporal and spatial model information in MIAs. Next, we assess the effectiveness via privacy-utility tradeoff for two type defense mechanisms against MI
    

