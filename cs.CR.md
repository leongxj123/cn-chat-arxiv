# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A backdoor attack against link prediction tasks with graph neural networks.](http://arxiv.org/abs/2401.02663) | 本文研究了一种针对图神经网络链接预测任务的后门攻击方法，发现GNN模型容易受到后门攻击，提出了针对该任务的后门攻击方式。 |
| [^2] | [Revealing CNN Architectures via Side-Channel Analysis in Dataflow-based Inference Accelerators.](http://arxiv.org/abs/2311.00579) | 本文通过评估数据流加速器上的侧信道信息，提出了一种攻击方法来恢复CNN模型的架构。该攻击利用了数据流映射的数据重用以及架构线索，成功恢复了流行的CNN模型Lenet，Alexnet和VGGnet16的结构。 |
| [^3] | [Breaking NoC Anonymity using Flow Correlation Attack.](http://arxiv.org/abs/2309.15687) | 本文研究了NoC架构中现有匿名路由协议的安全性，并展示了现有的匿名路由对基于机器学习的流相关攻击易受攻击。我们提出了一种轻量级的匿名路由，使用流量混淆技术，可以抵御基于机器学习的流相关攻击。 |

# 详细

[^1]: 用于图神经网络链接预测任务的后门攻击

    A backdoor attack against link prediction tasks with graph neural networks. (arXiv:2401.02663v1 [cs.LG])

    [http://arxiv.org/abs/2401.02663](http://arxiv.org/abs/2401.02663)

    本文研究了一种针对图神经网络链接预测任务的后门攻击方法，发现GNN模型容易受到后门攻击，提出了针对该任务的后门攻击方式。

    

    图神经网络（GNN）是一类能够处理图结构数据的深度学习模型，在各种实际应用中表现出显著的性能。最近的研究发现，GNN模型容易受到后门攻击。当具体的模式（称为后门触发器，例如子图、节点等）出现在输入数据中时，嵌入在GNN模型中的后门会被激活，将输入数据误分类为攻击者指定的目标类标签，而当输入中没有后门触发器时，嵌入在GNN模型中的后门不会被激活，模型正常工作。后门攻击具有极高的隐蔽性，给GNN模型带来严重的安全风险。目前，对GNN的后门攻击研究主要集中在图分类和节点分类等任务上，对链接预测任务的后门攻击研究较少。在本文中，我们提出一种后门攻击方法。

    Graph Neural Networks (GNNs) are a class of deep learning models capable of processing graph-structured data, and they have demonstrated significant performance in a variety of real-world applications. Recent studies have found that GNN models are vulnerable to backdoor attacks. When specific patterns (called backdoor triggers, e.g., subgraphs, nodes, etc.) appear in the input data, the backdoor embedded in the GNN models is activated, which misclassifies the input data into the target class label specified by the attacker, whereas when there are no backdoor triggers in the input, the backdoor embedded in the GNN models is not activated, and the models work normally. Backdoor attacks are highly stealthy and expose GNN models to serious security risks. Currently, research on backdoor attacks against GNNs mainly focus on tasks such as graph classification and node classification, and backdoor attacks against link prediction tasks are rarely studied. In this paper, we propose a backdoor a
    
[^2]: 通过数据流推理加速器中的侧信道分析揭示CNN架构

    Revealing CNN Architectures via Side-Channel Analysis in Dataflow-based Inference Accelerators. (arXiv:2311.00579v1 [cs.CR])

    [http://arxiv.org/abs/2311.00579](http://arxiv.org/abs/2311.00579)

    本文通过评估数据流加速器上的侧信道信息，提出了一种攻击方法来恢复CNN模型的架构。该攻击利用了数据流映射的数据重用以及架构线索，成功恢复了流行的CNN模型Lenet，Alexnet和VGGnet16的结构。

    

    卷积神经网络（CNN）广泛应用于各个领域。最近在基于数据流的CNN加速器的进展使得CNN推理可以在资源有限的边缘设备上进行。这些数据流加速器利用卷积层的固有数据重用来高效处理CNN模型。隐藏CNN模型的架构对于隐私和安全至关重要。本文评估了基于内存的侧信道信息，以从数据流加速器中恢复CNN架构。所提出的攻击利用了CNN加速器上数据流映射的空间和时间数据重用以及架构线索来恢复CNN模型的结构。实验结果表明，我们提出的侧信道攻击可以恢复流行的CNN模型Lenet，Alexnet和VGGnet16的结构。

    Convolution Neural Networks (CNNs) are widely used in various domains. Recent advances in dataflow-based CNN accelerators have enabled CNN inference in resource-constrained edge devices. These dataflow accelerators utilize inherent data reuse of convolution layers to process CNN models efficiently. Concealing the architecture of CNN models is critical for privacy and security. This paper evaluates memory-based side-channel information to recover CNN architectures from dataflow-based CNN inference accelerators. The proposed attack exploits spatial and temporal data reuse of the dataflow mapping on CNN accelerators and architectural hints to recover the structure of CNN models. Experimental results demonstrate that our proposed side-channel attack can recover the structures of popular CNN models, namely Lenet, Alexnet, and VGGnet16.
    
[^3]: 打破NoC匿名性使用流相关攻击

    Breaking NoC Anonymity using Flow Correlation Attack. (arXiv:2309.15687v1 [cs.CR])

    [http://arxiv.org/abs/2309.15687](http://arxiv.org/abs/2309.15687)

    本文研究了NoC架构中现有匿名路由协议的安全性，并展示了现有的匿名路由对基于机器学习的流相关攻击易受攻击。我们提出了一种轻量级的匿名路由，使用流量混淆技术，可以抵御基于机器学习的流相关攻击。

    

    网络片上互连（NoC）广泛用作当今多核片上系统（SoC）设计中的内部通信结构。片上通信的安全性至关重要，因为利用共享的NoC中的任何漏洞对攻击者来说都是一个富矿。NoC安全依赖于对各种攻击的有效防范措施。我们研究了NoC架构中现有匿名路由协议的安全性。具体而言，本文作出了两个重要贡献。我们展示了现有的匿名路由对基于机器学习（ML）的流相关攻击是易受攻击的。我们提出了一种轻量级的匿名路由，使用流量混淆技术，可以抵御基于ML的流相关攻击。使用实际和合成流量进行的实验研究表明，我们提出的攻击能够成功地对抗NoC架构中最先进的匿名路由，对于多种流量模式的分类准确率高达99％，同时。

    Network-on-Chip (NoC) is widely used as the internal communication fabric in today's multicore System-on-Chip (SoC) designs. Security of the on-chip communication is crucial because exploiting any vulnerability in shared NoC would be a goldmine for an attacker. NoC security relies on effective countermeasures against diverse attacks. We investigate the security strength of existing anonymous routing protocols in NoC architectures. Specifically, this paper makes two important contributions. We show that the existing anonymous routing is vulnerable to machine learning (ML) based flow correlation attacks on NoCs. We propose a lightweight anonymous routing that use traffic obfuscation techniques which can defend against ML-based flow correlation attacks. Experimental studies using both real and synthetic traffic reveal that our proposed attack is successful against state-of-the-art anonymous routing in NoC architectures with a high accuracy (up to 99%) for diverse traffic patterns, while o
    

