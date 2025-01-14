# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AI in Supply Chain Risk Assessment: A Systematic Literature Review and Bibliometric Analysis.](http://arxiv.org/abs/2401.10895) | 本文通过系统文献综述和文献计量分析，填补了供应链风险评估中新兴人工智能/机器学习技术的研究空白，为了解这些技术在实践中的实际影响提供了关键见解。 |
| [^2] | [Towards Real-time Training of Physics-informed Neural Networks: Applications in Ultrafast Ultrasound Blood Flow Imaging.](http://arxiv.org/abs/2309.04755) | 本研究提出了一种实时训练基于物理信息的神经网络（PINN）的框架，用于解决Navier-Stokes方程，以实现超快速超声血流成像。该框架将Navier-Stokes方程离散化为稳态，并通过迁移学习顺序求解稳态方程。此外，采用平均恒定随机梯度下降作为初始化，并提出了一种并行训练方案，适用于所有时间戳。 |

# 详细

[^1]: 供应链风险评估中的人工智能：一项系统文献综述和文献计量分析

    AI in Supply Chain Risk Assessment: A Systematic Literature Review and Bibliometric Analysis. (arXiv:2401.10895v1 [cs.LG])

    [http://arxiv.org/abs/2401.10895](http://arxiv.org/abs/2401.10895)

    本文通过系统文献综述和文献计量分析，填补了供应链风险评估中新兴人工智能/机器学习技术的研究空白，为了解这些技术在实践中的实际影响提供了关键见解。

    

    通过整合人工智能和机器学习技术，供应链风险评估(SCRA)经历了深刻的演变，革新了预测能力和风险缓解策略。这种演变的重要性在于在现代供应链中确保运营的韧性和连续性，需要稳健的风险管理策略。以往的综述已经概述了已建立的方法，但忽视了新兴的人工智能/机器学习技术，在理解其在SCRA中的实际影响方面存在明显的研究空白。本文进行了系统的文献综述，并结合了全面的文献计量分析。我们仔细研究了1717篇论文，并从2014年至2023年之间发表的48篇文章中获得了关键见解。该综述填补了这一研究空白，通过回答关键研究问题，探究了现有的人工智能/机器学习技术、方法论、研究结果和未来发展方向。

    Supply chain risk assessment (SCRA) has witnessed a profound evolution through the integration of artificial intelligence (AI) and machine learning (ML) techniques, revolutionizing predictive capabilities and risk mitigation strategies. The significance of this evolution stems from the critical role of robust risk management strategies in ensuring operational resilience and continuity within modern supply chains. Previous reviews have outlined established methodologies but have overlooked emerging AI/ML techniques, leaving a notable research gap in understanding their practical implications within SCRA. This paper conducts a systematic literature review combined with a comprehensive bibliometric analysis. We meticulously examined 1,717 papers and derived key insights from a select group of 48 articles published between 2014 and 2023. The review fills this research gap by addressing pivotal research questions, and exploring existing AI/ML techniques, methodologies, findings, and future 
    
[^2]: 实时训练基于物理信息的神经网络：超快速超声血流成像的应用

    Towards Real-time Training of Physics-informed Neural Networks: Applications in Ultrafast Ultrasound Blood Flow Imaging. (arXiv:2309.04755v1 [cs.CE])

    [http://arxiv.org/abs/2309.04755](http://arxiv.org/abs/2309.04755)

    本研究提出了一种实时训练基于物理信息的神经网络（PINN）的框架，用于解决Navier-Stokes方程，以实现超快速超声血流成像。该框架将Navier-Stokes方程离散化为稳态，并通过迁移学习顺序求解稳态方程。此外，采用平均恒定随机梯度下降作为初始化，并提出了一种并行训练方案，适用于所有时间戳。

    

    基于物理信息的神经网络（PINN）是纳维-斯托克斯方程的最杰出求解器之一，而纳维-斯托克斯方程广泛应用于血流的控制方程。然而，目前的方法仅依赖于完整的纳维-斯托克斯方程，对于超快速多普勒超声，这一最新技术应用于\emph{体内}复杂血流动力学的展示，每秒获取数千帧（或时间戳），这是不切实际的。本文首先提出了一种新的PINN训练框架，通过将纳维-斯托克斯方程离散化为稳态，并通过迁移学习顺序求解稳态纳维-斯托克斯方程，为解决纳维-斯托克斯方程提供了新的训练框架，称为SeqPINN。在SeqPINN的成功基础上，我们采用了平均恒定随机梯度下降（SGD）作为初始化的思想，并提出了一种并行训练方案，适用于所有时间戳。为了确保良好的泛化初始化，我们借鉴了

    Physics-informed Neural Network (PINN) is one of the most preeminent solvers of Navier-Stokes equations, which are widely used as the governing equation of blood flow. However, current approaches, relying on full Navier-Stokes equations, are impractical for ultrafast Doppler ultrasound, the state-of-the-art technique for depiction of complex blood flow dynamics \emph{in vivo} through acquired thousands of frames (or, timestamps) per second. In this article, we first propose a novel training framework of PINN for solving Navier-Stokes equations by discretizing Navier-Stokes equations into steady state and sequentially solving steady-state Navier-Stokes equations with transfer learning. The novel training framework is coined as SeqPINN. Upon the success of SeqPINN, we adopt the idea of averaged constant stochastic gradient descent (SGD) as initialization and propose a parallel training scheme for all timestamps. To ensure an initialization that generalizes well, we borrow the concept of 
    

