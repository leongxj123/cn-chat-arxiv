# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-Agent Diagnostics for Robustness via Illuminated Diversity.](http://arxiv.org/abs/2401.13460) | MADRID是一种新方法，通过生成多样化的对抗场景来揭示预训练多Agent策略的战略漏洞，并通过遗憾值衡量漏洞的程度。 |
| [^2] | [Learning to Control and Coordinate Mixed Traffic Through Robot Vehicles at Complex and Unsignalized Intersections.](http://arxiv.org/abs/2301.05294) | 本研究提出了一种去中心化的多智能体强化学习方法，用于控制和协调混合交通，特别是人驾驶车辆和机器人车辆在实际复杂交叉口的应用。实验结果表明，使用5%的机器人车辆可以有效防止交叉口内的拥堵形成。 |

# 详细

[^1]: 通过多样性启示的多Agent诊断方法用于稳健性

    Multi-Agent Diagnostics for Robustness via Illuminated Diversity. (arXiv:2401.13460v1 [cs.LG])

    [http://arxiv.org/abs/2401.13460](http://arxiv.org/abs/2401.13460)

    MADRID是一种新方法，通过生成多样化的对抗场景来揭示预训练多Agent策略的战略漏洞，并通过遗憾值衡量漏洞的程度。

    

    在快速发展的多Agent系统领域中，确保在陌生和敌对环境中的稳健性至关重要。尽管这些系统在熟悉环境中表现出色，但在新情况下往往会因为训练阶段的过拟合而失败。在既包含合作又包含竞争行为的环境中，这一问题尤为突出，体现了过拟合和泛化挑战的双重性质。为了解决这个问题，我们提出了通过多样性启示的多Agent稳健性诊断（MADRID），这是一种生成多Agent策略中暴露战略漏洞的多样化对抗场景的新方法。MADRID利用开放式学习的概念，导航对抗环境的广阔空间，使用目标策略的遗憾值来衡量这些环境的漏洞。我们在11vs11版的Google Research Football上评估了MADRID的有效性。

    In the rapidly advancing field of multi-agent systems, ensuring robustness in unfamiliar and adversarial settings is crucial. Notwithstanding their outstanding performance in familiar environments, these systems often falter in new situations due to overfitting during the training phase. This is especially pronounced in settings where both cooperative and competitive behaviours are present, encapsulating a dual nature of overfitting and generalisation challenges. To address this issue, we present Multi-Agent Diagnostics for Robustness via Illuminated Diversity (MADRID), a novel approach for generating diverse adversarial scenarios that expose strategic vulnerabilities in pre-trained multi-agent policies. Leveraging the concepts from open-ended learning, MADRID navigates the vast space of adversarial settings, employing a target policy's regret to gauge the vulnerabilities of these settings. We evaluate the effectiveness of MADRID on the 11vs11 version of Google Research Football, one o
    
[^2]: 通过机器人车辆在复杂和无信号的交叉口中学习控制和协调混合交通

    Learning to Control and Coordinate Mixed Traffic Through Robot Vehicles at Complex and Unsignalized Intersections. (arXiv:2301.05294v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.05294](http://arxiv.org/abs/2301.05294)

    本研究提出了一种去中心化的多智能体强化学习方法，用于控制和协调混合交通，特别是人驾驶车辆和机器人车辆在实际复杂交叉口的应用。实验结果表明，使用5%的机器人车辆可以有效防止交叉口内的拥堵形成。

    

    交叉口是现代大都市交通中必不可少的道路基础设施。然而，由于交通事故或缺乏交通协调机制（如交通信号灯），它们也可能成为交通流的瓶颈。最近，提出了各种超越传统控制方法的控制和协调机制，以提高交叉口交通的效率。在这些方法中，控制可预见的包含人驾驶车辆（HVs）和机器人车辆（RVs）的混合交通已经出现。在本项目中，我们提出了一种去中心化的多智能体强化学习方法，用于实际复杂交叉口的混合交通的控制和协调，这是一个以前未被探索过的主题。我们进行了全面的实验，以展示我们方法的有效性。特别是，我们展示了在实际交通条件下，使用5%的RVs，我们可以防止复杂交叉口内的拥堵形成。

    Intersections are essential road infrastructures for traffic in modern metropolises. However, they can also be the bottleneck of traffic flows as a result of traffic incidents or the absence of traffic coordination mechanisms such as traffic lights. Recently, various control and coordination mechanisms that are beyond traditional control methods have been proposed to improve the efficiency of intersection traffic. Amongst these methods, the control of foreseeable mixed traffic that consists of human-driven vehicles (HVs) and robot vehicles (RVs) has emerged. In this project, we propose a decentralized multi-agent reinforcement learning approach for the control and coordination of mixed traffic at real-world, complex intersections--a topic that has not been previously explored. Comprehensive experiments are conducted to show the effectiveness of our approach. In particular, we show that using 5% RVs, we can prevent congestion formation inside a complex intersection under the actual traf
    

