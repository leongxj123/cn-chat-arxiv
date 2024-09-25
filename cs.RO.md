# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SceneVerse: Scaling 3D Vision-Language Learning for Grounded Scene Understanding](https://arxiv.org/abs/2401.09340) | 本研究通过系统性地扩展室内环境中的3D视觉-语言学习，提出了首个百万规模的3D视觉-语言数据集SceneVerse，以解决3D视觉-语言对齐面临的几个重要挑战。 |
| [^2] | [GUARD: A Safe Reinforcement Learning Benchmark.](http://arxiv.org/abs/2305.13681) | GUARD是一个广义统一安全强化学习开发基准测试平台，是目前广泛遍布且包含各种RL代理、任务和安全约束规范的一站式基准测试，能够全面涵盖最先进的安全RL算法，并具有高度的可自定义性。 |

# 详细

[^1]: SceneVerse：为基于场景的场景理解扩展3D视觉-语言学习

    SceneVerse: Scaling 3D Vision-Language Learning for Grounded Scene Understanding

    [https://arxiv.org/abs/2401.09340](https://arxiv.org/abs/2401.09340)

    本研究通过系统性地扩展室内环境中的3D视觉-语言学习，提出了首个百万规模的3D视觉-语言数据集SceneVerse，以解决3D视觉-语言对齐面临的几个重要挑战。

    

    3D视觉-语言对齐，即将语言与3D物理环境对齐，是发展具身体能力的智能体的基石。与2D领域最近的进展相比，将语言与3D场景对齐面临着几个重要挑战：（i）3D场景固有复杂性，由于多样的物体配置、丰富的属性和错综复杂的关系；（ii）支持基于场景学习的配对3D视觉-语言数据的稀缺性；以及（iii）缺乏从基于场景的3D数据中提炼知识的统一学习框架。在这项工作中，我们旨在通过系统地扩展室内环境中的3D视觉-语言学习，从而解决3D视觉-语言领域中的这三大挑战。我们介绍首个百万规模的3D视觉-语言数据集SceneVerse，包含约68K个3D室内场景，包括250万个视觉语言

    arXiv:2401.09340v2 Announce Type: replace-cross  Abstract: 3D vision-language grounding, which focuses on aligning language with the 3D physical environment, stands as a cornerstone in the development of embodied agents. In comparison to recent advancements in the 2D domain, grounding language in 3D scenes faces several significant challenges: (i) the inherent complexity of 3D scenes due to the diverse object configurations, their rich attributes, and intricate relationships; (ii) the scarcity of paired 3D vision-language data to support grounded learning; and (iii) the absence of a unified learning framework to distill knowledge from grounded 3D data. In this work, we aim to address these three major challenges in 3D vision-language by examining the potential of systematically upscaling 3D vision-language learning in indoor environments. We introduce the first million-scale 3D vision-language dataset, SceneVerse, encompassing about 68K 3D indoor scenes and comprising 2.5M vision-langu
    
[^2]: GUARD: 一个安全强化学习基准测试平台

    GUARD: A Safe Reinforcement Learning Benchmark. (arXiv:2305.13681v1 [cs.LG])

    [http://arxiv.org/abs/2305.13681](http://arxiv.org/abs/2305.13681)

    GUARD是一个广义统一安全强化学习开发基准测试平台，是目前广泛遍布且包含各种RL代理、任务和安全约束规范的一站式基准测试，能够全面涵盖最先进的安全RL算法，并具有高度的可自定义性。

    

    由于试错的性质，将RL算法应用于安全关键的现实应用（例如自动驾驶、人机交互、机器人操作等）通常是具有挑战性的，因为这些错误是不可容忍的。最近，安全RL（即约束RL）已经在文献中迅速出现，其中代理在满足约束条件的同时，探索环境。由于算法和任务的多样性，比较现有的安全RL算法仍然很困难。为了填补这一空白，我们介绍了GUARD，一个广义统一安全强化学习开发基准测试平台。与现有基准相比，GUARD具有几个优点。首先，GUARD是一个广义基准测试平台，具有各种RL代理、任务和安全约束规范。其次，GUARD全面涵盖了最先进的安全RL算法，并具有自包含的实现。第三，GUARD在任务和算法方面具有高度的可自定义性。我们提供了状态下现有方法在GUARD上的基准测试结果。

    Due to the trial-and-error nature, it is typically challenging to apply RL algorithms to safety-critical real-world applications, such as autonomous driving, human-robot interaction, robot manipulation, etc, where such errors are not tolerable. Recently, safe RL (i.e. constrained RL) has emerged rapidly in the literature, in which the agents explore the environment while satisfying constraints. Due to the diversity of algorithms and tasks, it remains difficult to compare existing safe RL algorithms. To fill that gap, we introduce GUARD, a Generalized Unified SAfe Reinforcement Learning Development Benchmark. GUARD has several advantages compared to existing benchmarks. First, GUARD is a generalized benchmark with a wide variety of RL agents, tasks, and safety constraint specifications. Second, GUARD comprehensively covers state-of-the-art safe RL algorithms with self-contained implementations. Third, GUARD is highly customizable in tasks and algorithms. We present a comparison of state
    

