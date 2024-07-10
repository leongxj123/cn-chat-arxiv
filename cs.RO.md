# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Are Large Language Models Aligned with People's Social Intuitions for Human-Robot Interactions?](https://arxiv.org/abs/2403.05701) | 该研究测试大型语言模型在人机互动中是否能够捕捉到人们的行为判断和沟通偏好，结果表明GPT-4在生成社会可接受行为方面表现出色。 |
| [^2] | [Working Backwards: Learning to Place by Picking](https://arxiv.org/abs/2312.02352) | 通过逆向抓取过程并利用拾取和放置问题的对称性，提出了一种通过拾取的放置方法，并用自主收集的演示直接训练策略，实现在接触受限环境下物体放置任务的自主收集和泛化。 |

# 详细

[^1]: 大型语言模型是否与人们的社交直觉相一致，用于人机互动？

    Are Large Language Models Aligned with People's Social Intuitions for Human-Robot Interactions?

    [https://arxiv.org/abs/2403.05701](https://arxiv.org/abs/2403.05701)

    该研究测试大型语言模型在人机互动中是否能够捕捉到人们的行为判断和沟通偏好，结果表明GPT-4在生成社会可接受行为方面表现出色。

    

    大型语言模型（LLMs）越来越多地用于机器人技术，特别是高层次的行动规划。与此同时，许多机器人应用涉及人类监督员或合作者。因此，对LLMs生成与人们偏好和价值观相一致的社会可接受行动至关重要。在这项工作中，我们测试LLMs是否捕捉到人们在人机互动（HRI）场景中行为判断和沟通偏好方面的直觉。为了评估，我们重现了三个HRI用户研究，将LLMs的输出与真实参与者的输出进行比较。我们发现GPT-4在非常出色地表现，生成的答案与两项研究的用户答案具有很强相关性——第一项研究涉及在各种情境中选择最合适的沟通举动给机器人（$r_s$ = 0.82），第二项涉及判断行为的可取性、意图性和令人惊讶性。

    arXiv:2403.05701v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used in robotics, especially for high-level action planning. Meanwhile, many robotics applications involve human supervisors or collaborators. Hence, it is crucial for LLMs to generate socially acceptable actions that align with people's preferences and values. In this work, we test whether LLMs capture people's intuitions about behavior judgments and communication preferences in human-robot interaction (HRI) scenarios. For evaluation, we reproduce three HRI user studies, comparing the output of LLMs with that of real participants. We find that GPT-4 strongly outperforms other models, generating answers that correlate strongly with users' answers in two studies $\unicode{x2014}$ the first study dealing with selecting the most appropriate communicative act for a robot in various situations ($r_s$ = 0.82), and the second with judging the desirability, intentionality, and surprisingness of beh
    
[^2]: 逆向学习：通过捡取学习放置

    Working Backwards: Learning to Place by Picking

    [https://arxiv.org/abs/2312.02352](https://arxiv.org/abs/2312.02352)

    通过逆向抓取过程并利用拾取和放置问题的对称性，提出了一种通过拾取的放置方法，并用自主收集的演示直接训练策略，实现在接触受限环境下物体放置任务的自主收集和泛化。

    

    我们提出了一种通过拾取（PvP）的放置方法，可以自主收集适用于一系列放置任务的现实世界演示，其中物体必须被操纵到特定的接触限制位置。通过PvP，我们通过颠倒抓取过程并利用拾取和放置问题固有的对称性，接近于机器人物体放置演示的收集。具体而言，我们从一组最初位于目标放置位置的物体的抓取序列中获得放置演示。我们的系统可以在接触受限环境中收集数百个演示，而无需人类干预，这是通过结合两个模块实现的：触觉重新抓取和用于抓取的顺从控制。我们通过行为克隆直接从视觉观察中通过自主收集的演示中训练策略。通过这样做，策略可以推广到超出训练环境范围的物体放置场景。

    arXiv:2312.02352v2 Announce Type: replace-cross  Abstract: We present placing via picking (PvP), a method to autonomously collect real-world demonstrations for a family of placing tasks in which objects must be manipulated to specific contact-constrained locations. With PvP, we approach the collection of robotic object placement demonstrations by reversing the grasping process and exploiting the inherent symmetry of the pick and place problems. Specifically, we obtain placing demonstrations from a set of grasp sequences of objects initially located at their target placement locations. Our system can collect hundreds of demonstrations in contact-constrained environments without human intervention by combining two modules: tactile regrasping and compliant control for grasps. We train a policy directly from visual observations through behavioral cloning, using the autonomously-collected demonstrations. By doing so, the policy can generalize to object placement scenarios outside of the tra
    

