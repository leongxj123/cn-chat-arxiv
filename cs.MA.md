# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Picture Is Worth a Graph: Blueprint Debate on Graph for Multimodal Reasoning](https://arxiv.org/abs/2403.14972) | 提出了一种演绎式的图谱辩论方法（BDoG），在多模态推理中防止意见陈腐化和减少由图像引入的分心概念，实验证明其在科学问答和MMBench上取得了最先进的结果。 |
| [^2] | [Cooperation and Control in Delegation Games](https://arxiv.org/abs/2402.15821) | 本文在委托游戏中探讨了控制问题（代理人未能按照其委托人的偏好行事）和合作问题（代理人未能良好地协作），并分析了对齐和能力对委托人福利的影响。en_tdlr: This paper explores the issues of control (agents failing to act in line with their principals' preferences) and cooperation (agents failing to work well together) in delegation games, analyzing how alignment and capabilities impact principals' welfare. |

# 详细

[^1]: 一图胜千言：多模态推理中的图谱辩论

    A Picture Is Worth a Graph: Blueprint Debate on Graph for Multimodal Reasoning

    [https://arxiv.org/abs/2403.14972](https://arxiv.org/abs/2403.14972)

    提出了一种演绎式的图谱辩论方法（BDoG），在多模态推理中防止意见陈腐化和减少由图像引入的分心概念，实验证明其在科学问答和MMBench上取得了最先进的结果。

    

    本文介绍了一项旨在将多智能体辩论引入多模态推理的试点研究。该研究解决了两个关键挑战：由于过度总结而导致意见陈腐化，以及由于图像引入转移性概念而导致注意力分散的问题。这些挑战源自现有辩论方案的归纳（自下而上）性质。为解决这一问题，我们提出了一种演绎（自上而下）的辩论方法，称为图谱辩论（BDoG）。在BDoG中，辩论仅限于蓝图图中，以防止通过世界级摘要而导致意见陈腐化。此外，通过在图中的分支中存储证据，BDoG缓解了频繁但无关的概念带来的分散注意力现象。大量实验验证了BDoG，在科学问答和MMBench中取得了最新成果，并相较于先前的方法具有显著改进。

    arXiv:2403.14972v1 Announce Type: new  Abstract: This paper presents a pilot study aimed at introducing multi-agent debate into multimodal reasoning. The study addresses two key challenges: the trivialization of opinions resulting from excessive summarization and the diversion of focus caused by distractor concepts introduced from images. These challenges stem from the inductive (bottom-up) nature of existing debating schemes. To address the issue, we propose a deductive (top-down) debating approach called Blueprint Debate on Graphs (BDoG). In BDoG, debates are confined to a blueprint graph to prevent opinion trivialization through world-level summarization. Moreover, by storing evidence in branches within the graph, BDoG mitigates distractions caused by frequent but irrelevant concepts. Extensive experiments validate BDoG, achieving state-of-the-art results in Science QA and MMBench with significant improvements over previous methods.
    
[^2]: 委托游戏中的合作与控制

    Cooperation and Control in Delegation Games

    [https://arxiv.org/abs/2402.15821](https://arxiv.org/abs/2402.15821)

    本文在委托游戏中探讨了控制问题（代理人未能按照其委托人的偏好行事）和合作问题（代理人未能良好地协作），并分析了对齐和能力对委托人福利的影响。en_tdlr: This paper explores the issues of control (agents failing to act in line with their principals' preferences) and cooperation (agents failing to work well together) in delegation games, analyzing how alignment and capabilities impact principals' welfare.

    

    许多涉及人类和机器的感兴趣的场景 - 从虚拟个人助理到自动驾驶车辆 - 可以自然地建模为委托人（人类）委托给代理人（机器），这些代理人之后代表他们的委托人相互交互。我们将这些多委托人，多代理人的情况称为委托游戏。在这类游戏中，存在两种重要的失败模式：控制问题（代理人未能按照其委托人的偏好行事）和合作问题（代理人未能良好地协作）。在本文中，我们形式化和分析这些问题，进一步将其解释为对齐（参与者是否具有相似的偏好？）和能力（参与者在满足这些偏好方面的能力如何？）的问题。我们理论上和经验上展示了这些措施如何确定委托人的福利，如何可以使用有限的观察来估计这些措施，

    arXiv:2402.15821v1 Announce Type: cross  Abstract: Many settings of interest involving humans and machines -- from virtual personal assistants to autonomous vehicles -- can naturally be modelled as principals (humans) delegating to agents (machines), which then interact with each other on their principals' behalf. We refer to these multi-principal, multi-agent scenarios as delegation games. In such games, there are two important failure modes: problems of control (where an agent fails to act in line their principal's preferences) and problems of cooperation (where the agents fail to work well together). In this paper we formalise and analyse these problems, further breaking them down into issues of alignment (do the players have similar preferences?) and capabilities (how competent are the players at satisfying those preferences?). We show -- theoretically and empirically -- how these measures determine the principals' welfare, how they can be estimated using limited observations, and 
    

