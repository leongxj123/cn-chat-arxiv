# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Revealing CNN Architectures via Side-Channel Analysis in Dataflow-based Inference Accelerators.](http://arxiv.org/abs/2311.00579) | 本文通过评估数据流加速器上的侧信道信息，提出了一种攻击方法来恢复CNN模型的架构。该攻击利用了数据流映射的数据重用以及架构线索，成功恢复了流行的CNN模型Lenet，Alexnet和VGGnet16的结构。 |
| [^2] | [Breaking NoC Anonymity using Flow Correlation Attack.](http://arxiv.org/abs/2309.15687) | 本文研究了NoC架构中现有匿名路由协议的安全性，并展示了现有的匿名路由对基于机器学习的流相关攻击易受攻击。我们提出了一种轻量级的匿名路由，使用流量混淆技术，可以抵御基于机器学习的流相关攻击。 |

# 详细

[^1]: 通过数据流推理加速器中的侧信道分析揭示CNN架构

    Revealing CNN Architectures via Side-Channel Analysis in Dataflow-based Inference Accelerators. (arXiv:2311.00579v1 [cs.CR])

    [http://arxiv.org/abs/2311.00579](http://arxiv.org/abs/2311.00579)

    本文通过评估数据流加速器上的侧信道信息，提出了一种攻击方法来恢复CNN模型的架构。该攻击利用了数据流映射的数据重用以及架构线索，成功恢复了流行的CNN模型Lenet，Alexnet和VGGnet16的结构。

    

    卷积神经网络（CNN）广泛应用于各个领域。最近在基于数据流的CNN加速器的进展使得CNN推理可以在资源有限的边缘设备上进行。这些数据流加速器利用卷积层的固有数据重用来高效处理CNN模型。隐藏CNN模型的架构对于隐私和安全至关重要。本文评估了基于内存的侧信道信息，以从数据流加速器中恢复CNN架构。所提出的攻击利用了CNN加速器上数据流映射的空间和时间数据重用以及架构线索来恢复CNN模型的结构。实验结果表明，我们提出的侧信道攻击可以恢复流行的CNN模型Lenet，Alexnet和VGGnet16的结构。

    Convolution Neural Networks (CNNs) are widely used in various domains. Recent advances in dataflow-based CNN accelerators have enabled CNN inference in resource-constrained edge devices. These dataflow accelerators utilize inherent data reuse of convolution layers to process CNN models efficiently. Concealing the architecture of CNN models is critical for privacy and security. This paper evaluates memory-based side-channel information to recover CNN architectures from dataflow-based CNN inference accelerators. The proposed attack exploits spatial and temporal data reuse of the dataflow mapping on CNN accelerators and architectural hints to recover the structure of CNN models. Experimental results demonstrate that our proposed side-channel attack can recover the structures of popular CNN models, namely Lenet, Alexnet, and VGGnet16.
    
[^2]: 打破NoC匿名性使用流相关攻击

    Breaking NoC Anonymity using Flow Correlation Attack. (arXiv:2309.15687v1 [cs.CR])

    [http://arxiv.org/abs/2309.15687](http://arxiv.org/abs/2309.15687)

    本文研究了NoC架构中现有匿名路由协议的安全性，并展示了现有的匿名路由对基于机器学习的流相关攻击易受攻击。我们提出了一种轻量级的匿名路由，使用流量混淆技术，可以抵御基于机器学习的流相关攻击。

    

    网络片上互连（NoC）广泛用作当今多核片上系统（SoC）设计中的内部通信结构。片上通信的安全性至关重要，因为利用共享的NoC中的任何漏洞对攻击者来说都是一个富矿。NoC安全依赖于对各种攻击的有效防范措施。我们研究了NoC架构中现有匿名路由协议的安全性。具体而言，本文作出了两个重要贡献。我们展示了现有的匿名路由对基于机器学习（ML）的流相关攻击是易受攻击的。我们提出了一种轻量级的匿名路由，使用流量混淆技术，可以抵御基于ML的流相关攻击。使用实际和合成流量进行的实验研究表明，我们提出的攻击能够成功地对抗NoC架构中最先进的匿名路由，对于多种流量模式的分类准确率高达99％，同时。

    Network-on-Chip (NoC) is widely used as the internal communication fabric in today's multicore System-on-Chip (SoC) designs. Security of the on-chip communication is crucial because exploiting any vulnerability in shared NoC would be a goldmine for an attacker. NoC security relies on effective countermeasures against diverse attacks. We investigate the security strength of existing anonymous routing protocols in NoC architectures. Specifically, this paper makes two important contributions. We show that the existing anonymous routing is vulnerable to machine learning (ML) based flow correlation attacks on NoCs. We propose a lightweight anonymous routing that use traffic obfuscation techniques which can defend against ML-based flow correlation attacks. Experimental studies using both real and synthetic traffic reveal that our proposed attack is successful against state-of-the-art anonymous routing in NoC architectures with a high accuracy (up to 99%) for diverse traffic patterns, while o
    

