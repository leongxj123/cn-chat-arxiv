# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Revisiting Transferable Adversarial Image Examples: Attack Categorization, Evaluation Guidelines, and New Insights.](http://arxiv.org/abs/2310.11850) | 本论文重新审视了可转移的对抗性图像示例的评估方法，提出了新的攻击分类策略，并通过大规模评估揭示了一些新的见解和共识挑战。 |
| [^2] | [Learning Environment-Aware Affordance for 3D Articulated Object Manipulation under Occlusions.](http://arxiv.org/abs/2309.07510) | 本论文提出了一个环境感知的可供性框架，考虑了物体级的可行性先验和环境约束，以解决多个遮挡的复杂情况下的三维关节物体操作问题。 |

# 详细

[^1]: 重新审视可转移的对抗性图像示例：攻击分类，评估指南和新见解

    Revisiting Transferable Adversarial Image Examples: Attack Categorization, Evaluation Guidelines, and New Insights. (arXiv:2310.11850v1 [cs.CR])

    [http://arxiv.org/abs/2310.11850](http://arxiv.org/abs/2310.11850)

    本论文重新审视了可转移的对抗性图像示例的评估方法，提出了新的攻击分类策略，并通过大规模评估揭示了一些新的见解和共识挑战。

    

    可转移的对抗性示例在现实世界的黑盒攻击场景中引发了关键的安全问题。然而，在这项工作中，我们发现了常见评估实践中的两个主要问题：(1) 对于攻击的可转移性，缺乏系统化的，一对一的攻击比较和公平的超参数设置。(2) 对于攻击的隐蔽性，简单地没有比较。为了解决这些问题，我们通过(1) 提出一种新的攻击分类策略，并在可转移性方面进行系统化和公平的同类别分析，以及(2) 从攻击回溯的角度考虑多样的难以察觉的度量和更细粒度的隐蔽特性来建立新的评估指南。为此，我们对ImageNet上的可转移的对抗性示例进行了首次大规模评估，涉及对9种代表性防御的23种代表性攻击。我们的评估提供了一些新的见解，包括挑战共识的见解。

    Transferable adversarial examples raise critical security concerns in real-world, black-box attack scenarios. However, in this work, we identify two main problems in common evaluation practices: (1) For attack transferability, lack of systematic, one-to-one attack comparison and fair hyperparameter settings. (2) For attack stealthiness, simply no comparisons. To address these problems, we establish new evaluation guidelines by (1) proposing a novel attack categorization strategy and conducting systematic and fair intra-category analyses on transferability, and (2) considering diverse imperceptibility metrics and finer-grained stealthiness characteristics from the perspective of attack traceback. To this end, we provide the first large-scale evaluation of transferable adversarial examples on ImageNet, involving 23 representative attacks against 9 representative defenses. Our evaluation leads to a number of new insights, including consensus-challenging ones: (1) Under a fair attack hyper
    
[^2]: 学习环境感知的遮挡下三维关节物体操作的可供性

    Learning Environment-Aware Affordance for 3D Articulated Object Manipulation under Occlusions. (arXiv:2309.07510v1 [cs.RO])

    [http://arxiv.org/abs/2309.07510](http://arxiv.org/abs/2309.07510)

    本论文提出了一个环境感知的可供性框架，考虑了物体级的可行性先验和环境约束，以解决多个遮挡的复杂情况下的三维关节物体操作问题。

    

    在多样的环境中感知和操作三维关节物体对于家庭助理机器人至关重要。最近的研究表明，点级可供性为下游操作任务提供了可行性先验。然而，现有工作主要集中在单个物体场景中的均质代理，忽视了环境和代理形态所施加的现实约束，如遮挡和物理限制。在本文中，我们提出了一个环境感知的可供性框架，结合了物体级可行性先验和环境约束。与以物体为中心的可供性方法不同，学习环境感知的可供性面临着由各种遮挡的复杂性引起的组合爆炸挑战，这些遮挡以其数量、几何形状、位置和姿势来刻画。为了解决这个问题并提高数据效率，我们引入了一种新颖的对比式可供性学习框架，能够在含有遮挡的场景中进行训练。

    Perceiving and manipulating 3D articulated objects in diverse environments is essential for home-assistant robots. Recent studies have shown that point-level affordance provides actionable priors for downstream manipulation tasks. However, existing works primarily focus on single-object scenarios with homogeneous agents, overlooking the realistic constraints imposed by the environment and the agent's morphology, e.g., occlusions and physical limitations. In this paper, we propose an environment-aware affordance framework that incorporates both object-level actionable priors and environment constraints. Unlike object-centric affordance approaches, learning environment-aware affordance faces the challenge of combinatorial explosion due to the complexity of various occlusions, characterized by their quantities, geometries, positions and poses. To address this and enhance data efficiency, we introduce a novel contrastive affordance learning framework capable of training on scenes containin
    

