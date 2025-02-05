# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [UFID: A Unified Framework for Input-level Backdoor Detection on Diffusion Models](https://arxiv.org/abs/2404.01101) | 扩散模型容易受到后门攻击，本文提出了一个统一的框架用于输入级后门检测，弥补了该领域的空白，并不需要访问模型的白盒信息。 |
| [^2] | [SelfFed: Self-supervised Federated Learning for Data Heterogeneity and Label Scarcity in IoMT.](http://arxiv.org/abs/2307.01514) | 这篇论文提出了一种名为SelfFed的自监督联邦学习框架，用于解决IoMT中的数据异质性和标签匮乏问题。该框架包括预训练和微调两个阶段，通过分散训练和增强建模来克服数据异质性和标签稀缺问题。 |

# 详细

[^1]: UFID: 一个统一的框架用于扩散模型上的输入级后门检测

    UFID: A Unified Framework for Input-level Backdoor Detection on Diffusion Models

    [https://arxiv.org/abs/2404.01101](https://arxiv.org/abs/2404.01101)

    扩散模型容易受到后门攻击，本文提出了一个统一的框架用于输入级后门检测，弥补了该领域的空白，并不需要访问模型的白盒信息。

    

    扩散模型容易受到后门攻击，即恶意攻击者在训练阶段通过对部分训练样本进行毒化来注入后门。为了减轻后门攻击的威胁，对后门检测进行了大量研究。然而，没有人为扩散模型设计了专门的后门检测方法，使得这一领域较少被探索。此外，大多数先前的方法主要集中在传统神经网络的分类任务上，很难轻松地将其适应生成任务上的后门检测。此外，大多数先前的方法需要访问模型权重和架构的白盒访问，或概率logits作为额外信息，这并不总是切实可行的。在本文中

    arXiv:2404.01101v1 Announce Type: cross  Abstract: Diffusion Models are vulnerable to backdoor attacks, where malicious attackers inject backdoors by poisoning some parts of the training samples during the training stage. This poses a serious threat to the downstream users, who query the diffusion models through the API or directly download them from the internet. To mitigate the threat of backdoor attacks, there have been a plethora of investigations on backdoor detections. However, none of them designed a specialized backdoor detection method for diffusion models, rendering the area much under-explored. Moreover, these prior methods mainly focus on the traditional neural networks in the classification task, which cannot be adapted to the backdoor detections on the generative task easily. Additionally, most of the prior methods require white-box access to model weights and architectures, or the probability logits as additional information, which are not always practical. In this paper
    
[^2]: SelfFed: 自监督的联邦学习用于IoMT中的数据异质性和标签匮乏问题

    SelfFed: Self-supervised Federated Learning for Data Heterogeneity and Label Scarcity in IoMT. (arXiv:2307.01514v1 [cs.LG])

    [http://arxiv.org/abs/2307.01514](http://arxiv.org/abs/2307.01514)

    这篇论文提出了一种名为SelfFed的自监督联邦学习框架，用于解决IoMT中的数据异质性和标签匮乏问题。该框架包括预训练和微调两个阶段，通过分散训练和增强建模来克服数据异质性和标签稀缺问题。

    

    基于自监督学习的联邦学习范式在行业和研究领域中引起了很大的兴趣，因为它可以协作学习未标记但孤立的数据。然而，自监督的联邦学习策略在标签稀缺和数据异质性（即数据分布不同）方面存在性能下降的问题。在本文中，我们提出了适用于医疗物联网（IoMT）的SelfFed框架。我们的SelfFed框架分为两个阶段。第一个阶段是预训练范式，使用基于Swin Transformer的编码器以分散的方式进行增强建模。SelfFed框架的第一个阶段有助于克服数据异质性问题。第二个阶段是微调范式，引入对比网络和一种在有限标记数据上进行训练的新型聚合策略，用于目标任务的分散训练。这个微调阶段克服了标签稀缺问题。

    Self-supervised learning in federated learning paradigm has been gaining a lot of interest both in industry and research due to the collaborative learning capability on unlabeled yet isolated data. However, self-supervised based federated learning strategies suffer from performance degradation due to label scarcity and diverse data distributions, i.e., data heterogeneity. In this paper, we propose the SelfFed framework for Internet of Medical Things (IoMT). Our proposed SelfFed framework works in two phases. The first phase is the pre-training paradigm that performs augmentive modeling using Swin Transformer based encoder in a decentralized manner. The first phase of SelfFed framework helps to overcome the data heterogeneity issue. The second phase is the fine-tuning paradigm that introduces contrastive network and a novel aggregation strategy that is trained on limited labeled data for a target task in a decentralized manner. This fine-tuning stage overcomes the label scarcity problem
    

