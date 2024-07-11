# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Non-Coherent Over-the-Air Decentralized Gradient Descent](https://arxiv.org/abs/2211.10777) | 提出了一种适用于无线系统的DGD算法，通过无相干空中共识方案实现无需智能体协调、拓扑信息或信道状态信息的分布式优化。 |
| [^2] | [Balancing Privacy and Security in Federated Learning with FedGT: A Group Testing Framework.](http://arxiv.org/abs/2305.05506) | 该论文提出了FedGT框架，通过群体测试的方法在联邦学习中识别并删除恶意客户，从而平衡了隐私和安全，保护数据隐私并提高了识别恶意客户的能力。 |

# 详细

[^1]: 无相干空中分布式梯度下降

    Non-Coherent Over-the-Air Decentralized Gradient Descent

    [https://arxiv.org/abs/2211.10777](https://arxiv.org/abs/2211.10777)

    提出了一种适用于无线系统的DGD算法，通过无相干空中共识方案实现无需智能体协调、拓扑信息或信道状态信息的分布式优化。

    

    分布式梯度下降（DGD）是一种流行的算法，用于解决诸如远程感知、分布式推断、多智能体协调和联邦学习等各种领域的分布式优化问题。然而，在受到噪声、衰落和带宽受限的无线系统上执行DGD会带来挑战，需要调度传输以减轻干扰，并获取拓扑和信道状态信息，这在无线分布式系统中是复杂的任务。本文提出了一种专为无线系统定制的DGD算法。与现有方法不同，它在无需进行智能体协调、拓扑信息或信道状态信息的情况下运行。其核心是一种无相干空中（NCOTA）共识方案，利用了无线信道的噪声能量叠加特性。通过随机化传输策略来适应半双工操作，发射机将位置映射到

    arXiv:2211.10777v2 Announce Type: replace-cross  Abstract: Decentralized Gradient Descent (DGD) is a popular algorithm used to solve decentralized optimization problems in diverse domains such as remote sensing, distributed inference, multi-agent coordination, and federated learning. Yet, executing DGD over wireless systems affected by noise, fading and limited bandwidth presents challenges, requiring scheduling of transmissions to mitigate interference and the acquisition of topology and channel state information -- complex tasks in wireless decentralized systems. This paper proposes a DGD algorithm tailored to wireless systems. Unlike existing approaches, it operates without inter-agent coordination, topology information, or channel state information. Its core is a Non-Coherent Over-The-Air (NCOTA) consensus scheme, exploiting a noisy energy superposition property of wireless channels. With a randomized transmission strategy to accommodate half-duplex operation, transmitters map loca
    
[^2]: 在联邦学习中平衡隐私与安全：FedGT的群体测试框架

    Balancing Privacy and Security in Federated Learning with FedGT: A Group Testing Framework. (arXiv:2305.05506v1 [cs.LG])

    [http://arxiv.org/abs/2305.05506](http://arxiv.org/abs/2305.05506)

    该论文提出了FedGT框架，通过群体测试的方法在联邦学习中识别并删除恶意客户，从而平衡了隐私和安全，保护数据隐私并提高了识别恶意客户的能力。

    

    我们提出FedGT，一个新颖的框架，用于在联邦学习中识别恶意客户并进行安全聚合。受到群体测试的启发，该框架利用重叠的客户组来检测恶意客户的存在，并通过译码操作识别它们。然后，将这些被识别的客户从模型的训练中删除，并在其余客户之间执行训练。FedGT在隐私和安全之间取得平衡，允许改进识别能力同时仍保护数据隐私。具体而言，服务器学习每个组中客户的聚合模型。通过对MNIST和CIFAR-10数据集进行大量实验，证明了FedGT的有效性，展示了其识别恶意客户的能力，具有低误检和虚警概率，产生高模型效用。

    We propose FedGT, a novel framework for identifying malicious clients in federated learning with secure aggregation. Inspired by group testing, the framework leverages overlapping groups of clients to detect the presence of malicious clients in the groups and to identify them via a decoding operation. The identified clients are then removed from the training of the model, which is performed over the remaining clients. FedGT strikes a balance between privacy and security, allowing for improved identification capabilities while still preserving data privacy. Specifically, the server learns the aggregated model of the clients in each group. The effectiveness of FedGT is demonstrated through extensive experiments on the MNIST and CIFAR-10 datasets, showing its ability to identify malicious clients with low misdetection and false alarm probabilities, resulting in high model utility.
    

