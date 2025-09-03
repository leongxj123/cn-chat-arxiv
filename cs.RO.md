# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Dyna-LfLH: Learning Agile Navigation in Dynamic Environments from Learned Hallucination](https://arxiv.org/abs/2403.17231) | 提出了一种新的自监督学习方法Dyna-LfLH，通过学习幻觉中的动态环境，安全地学习地面机器人在动态环境中灵活导航。 |

# 详细

[^1]: Dyna-LfLH:从学到的幻觉中学会在动态环境中学习灵活导航

    Dyna-LfLH: Learning Agile Navigation in Dynamic Environments from Learned Hallucination

    [https://arxiv.org/abs/2403.17231](https://arxiv.org/abs/2403.17231)

    提出了一种新的自监督学习方法Dyna-LfLH，通过学习幻觉中的动态环境，安全地学习地面机器人在动态环境中灵活导航。

    

    这篇论文提出了一种自监督学习方法，用于安全地学习地面机器人的运动规划器，以在密集且动态的障碍物环境中导航。针对高度混乱、快速移动、难以预测的障碍物，传统的运动规划器可能无法跟上有限的机载计算。对于基于学习的规划器，很难获取高质量的演示以进行模仿学习，同时强化学习在探索过程中由于高碰撞概率而效率低下。为了安全有效地提供训练数据，LfH方法基于过去成功的导航经验在相对简单或完全开放的环境中综合困难的导航环境，但遗憾的是无法解决动态障碍物问题。在我们的新方法Dyna-LfLH中，我们设计并学习了一种新颖的潜在分布和样本。

    arXiv:2403.17231v1 Announce Type: cross  Abstract: This paper presents a self-supervised learning method to safely learn a motion planner for ground robots to navigate environments with dense and dynamic obstacles. When facing highly-cluttered, fast-moving, hard-to-predict obstacles, classical motion planners may not be able to keep up with limited onboard computation. For learning-based planners, high-quality demonstrations are difficult to acquire for imitation learning while reinforcement learning becomes inefficient due to the high probability of collision during exploration. To safely and efficiently provide training data, the Learning from Hallucination (LfH) approaches synthesize difficult navigation environments based on past successful navigation experiences in relatively easy or completely open ones, but unfortunately cannot address dynamic obstacles. In our new Dynamic Learning from Learned Hallucination (Dyna-LfLH), we design and learn a novel latent distribution and sample
    

