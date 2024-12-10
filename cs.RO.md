# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RealDex: Towards Human-like Grasping for Robotic Dexterous Hand](https://arxiv.org/abs/2402.13853) | RealDex数据集捕捉了真实的灵巧手抓取动作，利用多模态数据使得训练灵巧手更加自然和精确，同时提出了一种先进的灵巧抓取动作生成框架，有效利用多模态大型语言模型，在类人机器人的自动感知、认知和操纵方面具有巨大潜力。 |
| [^2] | [Adapt On-the-Go: Behavior Modulation for Single-Life Robot Deployment.](http://arxiv.org/abs/2311.01059) | 本研究提出了一种名为ROAM的方法，通过利用先前学习到的行为来实时调节机器人在部署过程中应对未曾见过的情况。在测试中，ROAM可以在单个阶段内实现快速适应，并且在模拟环境和真实场景中取得了成功，具有较高的效率和适应性。 |
| [^3] | [Your Room is not Private: Gradient Inversion Attack on Reinforcement Learning.](http://arxiv.org/abs/2306.09273) | 这篇论文提出了一种针对值函数算法和梯度算法的攻击方法，利用梯度反转重建状态、动作和监督信号，以解决嵌入式人工智能中的隐私泄露问题。 |

# 详细

[^1]: RealDex: 实现机器人灵巧手类人式抓取

    RealDex: Towards Human-like Grasping for Robotic Dexterous Hand

    [https://arxiv.org/abs/2402.13853](https://arxiv.org/abs/2402.13853)

    RealDex数据集捕捉了真实的灵巧手抓取动作，利用多模态数据使得训练灵巧手更加自然和精确，同时提出了一种先进的灵巧抓取动作生成框架，有效利用多模态大型语言模型，在类人机器人的自动感知、认知和操纵方面具有巨大潜力。

    

    在本文中，我们介绍了RealDex，一个开创性的数据集，捕捉了融入了人类行为模式的真实灵巧手抓取动作，同时通过多视角和多模态视觉数据进行了丰富。利用远程操作系统，我们可以实时无缝同步人-机器人手姿势。这些类人动作的集合对于训练灵巧手更自然、更精确地模仿人类动作至关重要。RealDex在推动类人机器人在真实场景中自动感知、认知和操纵方面具有巨大潜力。此外，我们介绍了一种前沿的灵巧抓取动作生成框架，该框架符合人类经验，并通过有效利用多模态大型语言模型增强了在现实世界中的适用性。广泛的实验证明了我们的方法在RealDex和其他开放数据集上的优越性能。完整的数据集和代码将会公开发布。

    arXiv:2402.13853v1 Announce Type: cross  Abstract: In this paper, we introduce RealDex, a pioneering dataset capturing authentic dexterous hand grasping motions infused with human behavioral patterns, enriched by multi-view and multimodal visual data. Utilizing a teleoperation system, we seamlessly synchronize human-robot hand poses in real time. This collection of human-like motions is crucial for training dexterous hands to mimic human movements more naturally and precisely. RealDex holds immense promise in advancing humanoid robot for automated perception, cognition, and manipulation in real-world scenarios. Moreover, we introduce a cutting-edge dexterous grasping motion generation framework, which aligns with human experience and enhances real-world applicability through effectively utilizing Multimodal Large Language Models. Extensive experiments have demonstrated the superior performance of our method on RealDex and other open datasets. The complete dataset and code will be made 
    
[^2]: 在部署时进行实时调节：用于单机器人部署的行为调控

    Adapt On-the-Go: Behavior Modulation for Single-Life Robot Deployment. (arXiv:2311.01059v1 [cs.RO])

    [http://arxiv.org/abs/2311.01059](http://arxiv.org/abs/2311.01059)

    本研究提出了一种名为ROAM的方法，通过利用先前学习到的行为来实时调节机器人在部署过程中应对未曾见过的情况。在测试中，ROAM可以在单个阶段内实现快速适应，并且在模拟环境和真实场景中取得了成功，具有较高的效率和适应性。

    

    为了在现实世界中取得成功，机器人必须应对训练过程中未曾见过的情况。本研究探讨了在部署过程中针对这些新场景的实时调节问题，通过利用先前学习到的多样化行为库。我们的方法，RObust Autonomous Modulation（ROAM），引入了基于预训练行为的感知价值的机制，以在特定情况下选择和调整预训练行为。关键是，这种调节过程在测试时的单个阶段内完成，无需任何人类监督。我们对选择机制进行了理论分析，并证明了ROAM使得机器人能够在模拟环境和真实的四足动物Go1上快速适应动态变化，甚至在脚上套着滚轮滑鞋的情况下成功前进。与现有方法相比，我们的方法在面对各种分布情况的部署时能够以超过2倍的效率进行调节，通过有效选择来实现适应。

    To succeed in the real world, robots must cope with situations that differ from those seen during training. We study the problem of adapting on-the-fly to such novel scenarios during deployment, by drawing upon a diverse repertoire of previously learned behaviors. Our approach, RObust Autonomous Modulation (ROAM), introduces a mechanism based on the perceived value of pre-trained behaviors to select and adapt pre-trained behaviors to the situation at hand. Crucially, this adaptation process all happens within a single episode at test time, without any human supervision. We provide theoretical analysis of our selection mechanism and demonstrate that ROAM enables a robot to adapt rapidly to changes in dynamics both in simulation and on a real Go1 quadruped, even successfully moving forward with roller skates on its feet. Our approach adapts over 2x as efficiently compared to existing methods when facing a variety of out-of-distribution situations during deployment by effectively choosing
    
[^3]: 你的房间不是私密的：关于强化学习的梯度反转攻击

    Your Room is not Private: Gradient Inversion Attack on Reinforcement Learning. (arXiv:2306.09273v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2306.09273](http://arxiv.org/abs/2306.09273)

    这篇论文提出了一种针对值函数算法和梯度算法的攻击方法，利用梯度反转重建状态、动作和监督信号，以解决嵌入式人工智能中的隐私泄露问题。

    

    嵌入式人工智能的显著发展吸引了人们的极大关注，该技术使得机器人可以在虚拟环境中导航、感知和互动。由于计算机视觉和大型语言模型方面的显著进展，隐私问题在嵌入式人工智能领域变得至关重要，因为机器人可以访问大量个人信息。然而，关于强化学习算法中的隐私泄露问题，尤其是关于值函数算法和梯度算法的问题，在研究中尚未得到充分考虑。本文旨在通过提出一种攻击值函数算法和梯度算法的方法，利用梯度反转重建状态、动作和监督信号，来解决这一问题。选择使用梯度进行攻击是因为常用的联邦学习技术仅利用基于私人用户数据计算的梯度来优化模型，而不存储或传输用户数据。

    The prominence of embodied Artificial Intelligence (AI), which empowers robots to navigate, perceive, and engage within virtual environments, has attracted significant attention, owing to the remarkable advancements in computer vision and large language models. Privacy emerges as a pivotal concern within the realm of embodied AI, as the robot accesses substantial personal information. However, the issue of privacy leakage in embodied AI tasks, particularly in relation to reinforcement learning algorithms, has not received adequate consideration in research. This paper aims to address this gap by proposing an attack on the value-based algorithm and the gradient-based algorithm, utilizing gradient inversion to reconstruct states, actions, and supervision signals. The choice of using gradients for the attack is motivated by the fact that commonly employed federated learning techniques solely utilize gradients computed based on private user data to optimize models, without storing or trans
    

