# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards a Dynamic Future with Adaptable Computing and Network Convergence (ACNC)](https://arxiv.org/abs/2403.07573) | 本文提出了可适应性CNC（ACNC）的概念，作为一种自主的机器学习（ML）辅助机制，旨在联合编排计算和网络资源，满足对动态和大量用户请求的严格要求。 |
| [^2] | [Zen: Near-Optimal Sparse Tensor Synchronization for Distributed DNN Training.](http://arxiv.org/abs/2309.13254) | 这篇论文介绍了Zen，一种用于分布式DNN训练中近似最优稀疏张量同步的方法。通过分析流行的DNN模型中稀疏张量的特性，并系统地探索设计空间，找到了最佳的通信方案。通过减少通信流量和提高训练效率，Zen有效地提升了分布式训练的性能。 |

# 详细

[^1]: 迈向具有可适应性计算和网络融合的动态未来（ACNC）

    Towards a Dynamic Future with Adaptable Computing and Network Convergence (ACNC)

    [https://arxiv.org/abs/2403.07573](https://arxiv.org/abs/2403.07573)

    本文提出了可适应性CNC（ACNC）的概念，作为一种自主的机器学习（ML）辅助机制，旨在联合编排计算和网络资源，满足对动态和大量用户请求的严格要求。

    

    在推进6G的背景下，预计会出现实质性的范式转变，突出了由大量连接和严格遵守服务质量/体验（QoS/E）先决条件所特征化的全面的一切对一切交互。即将面临的挑战源于资源稀缺，促使有意识地向计算-网络融合（CNC）过渡，作为联合资源编排的有前途的方法。虽然基于CNC的机制引起了人们的关注，但它们在实现未来服务方面的有效性，特别是在类似Metaverse的使用情景中，可能会由于用户、服务和资源不断变化的特性而受到限制。因此，本文提出了可适应性CNC（ACNC）的概念，作为一种自主的机器学习（ML）辅助机制，旨在联合编排计算和网络资源，满足对动态和大量用户请求的严格要求。

    arXiv:2403.07573v1 Announce Type: cross  Abstract: In the context of advancing 6G, a substantial paradigm shift is anticipated, highlighting comprehensive everything-to-everything interactions characterized by numerous connections and stringent adherence to Quality of Service/Experience (QoS/E) prerequisites. The imminent challenge stems from resource scarcity, prompting a deliberate transition to Computing-Network Convergence (CNC) as an auspicious approach for joint resource orchestration. While CNC-based mechanisms have garnered attention, their effectiveness in realizing future services, particularly in use cases like the Metaverse, may encounter limitations due to the continually changing nature of users, services, and resources. Hence, this paper presents the concept of Adaptable CNC (ACNC) as an autonomous Machine Learning (ML)-aided mechanism crafted for the joint orchestration of computing and network resources, catering to dynamic and voluminous user requests with stringent r
    
[^2]: Zen：用于分布式DNN训练的近似最优稀疏张量同步

    Zen: Near-Optimal Sparse Tensor Synchronization for Distributed DNN Training. (arXiv:2309.13254v1 [cs.LG])

    [http://arxiv.org/abs/2309.13254](http://arxiv.org/abs/2309.13254)

    这篇论文介绍了Zen，一种用于分布式DNN训练中近似最优稀疏张量同步的方法。通过分析流行的DNN模型中稀疏张量的特性，并系统地探索设计空间，找到了最佳的通信方案。通过减少通信流量和提高训练效率，Zen有效地提升了分布式训练的性能。

    

    分布式训练是使用多个GPU扩展深度神经网络(DNN)训练的事实标准。分布式训练的性能瓶颈在于渐变同步的通信。最近，实践者观察到渐变张量中存在稀疏性，表明可以减少通信的流量并提高端到端的训练效率。然而，完全发挥稀疏性的最佳通信方案仍然缺失。本文旨在解决这一问题。我们首先分析了流行DNN模型中稀疏张量的特性，以了解稀疏性的基本原理。然后，我们系统地探索了稀疏张量通信方案的设计空间并找到了最优解。

    Distributed training is the de facto standard to scale up the training of Deep Neural Networks (DNNs) with multiple GPUs. The performance bottleneck of distributed training lies in communications for gradient synchronization. Recently, practitioners have observed sparsity in gradient tensors, suggesting the potential to reduce the traffic volume in communication and improve end-to-end training efficiency. Yet, the optimal communication scheme to fully leverage sparsity is still missing. This paper aims to address this gap. We first analyze the characteristics of sparse tensors in popular DNN models to understand the fundamentals of sparsity. We then systematically explore the design space of communication schemes for sparse tensors and find the optimal one. % We then find the optimal scheme based on the characteristics by systematically exploring the design space. We also develop a gradient synchronization system called Zen that approximately realizes it for sparse tensors. We demonstr
    

