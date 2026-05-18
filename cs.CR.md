# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Approximate and Weighted Data Reconstruction Attack in Federated Learning](https://arxiv.org/abs/2308.06822) | 提出了一种基于插值的近似方法和层次加权损失函数，用于攻击FedAvg场景中的数据重构攻击。 |

# 详细

[^1]: 联邦学习中的近似和加权数据重构攻击

    Approximate and Weighted Data Reconstruction Attack in Federated Learning

    [https://arxiv.org/abs/2308.06822](https://arxiv.org/abs/2308.06822)

    提出了一种基于插值的近似方法和层次加权损失函数，用于攻击FedAvg场景中的数据重构攻击。

    

    联邦学习（FL）是一种分布式学习范例，使多个客户端能够在不共享私人数据的情况下合作构建机器学习模型。虽然FL被认为是通过设计保护隐私的，但最近的数据重构攻击表明，攻击者可以基于在FL中共享的参数恢复客户端的训练数据。然而，大多数现有方法未能攻击最广泛使用的水平联邦平均（FedAvg）场景，在此场景中，客户端在多个局部训练步骤之后共享模型参数。为了解决这个问题，我们提出了一种基于插值的近似方法，通过生成客户端局部训练过程的中间模型更新，使攻击FedAvg场景变得可行。然后，我们设计了一种层次加权损失函数来改善重构的数据质量。我们为不同层次的模型更新分配不同的权重

    arXiv:2308.06822v2 Announce Type: replace-cross  Abstract: Federated Learning (FL) is a distributed learning paradigm that enables multiple clients to collaborate on building a machine learning model without sharing their private data. Although FL is considered privacy-preserved by design, recent data reconstruction attacks demonstrate that an attacker can recover clients' training data based on the parameters shared in FL. However, most existing methods fail to attack the most widely used horizontal Federated Averaging (FedAvg) scenario, where clients share model parameters after multiple local training steps. To tackle this issue, we propose an interpolation-based approximation method, which makes attacking FedAvg scenarios feasible by generating the intermediate model updates of the clients' local training processes. Then, we design a layer-wise weighted loss function to improve the data quality of reconstruction. We assign different weights to model updates in different layers conc
    

