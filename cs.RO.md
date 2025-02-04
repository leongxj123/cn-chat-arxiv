# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Make a Donut: Hierarchical EMD-Space Planning for Zero-Shot Deformable Manipulation with Tools](https://arxiv.org/abs/2311.02787) | 引入了一种无需演示的分层规划方法，利用大型语言模型来解决复杂的长时间任务，为每个阶段提供工具名称和Python代码。 |

# 详细

[^1]: 制作一个甜甜圈：用于零样本变形操纵的分层EMD空间规划与工具

    Make a Donut: Hierarchical EMD-Space Planning for Zero-Shot Deformable Manipulation with Tools

    [https://arxiv.org/abs/2311.02787](https://arxiv.org/abs/2311.02787)

    引入了一种无需演示的分层规划方法，利用大型语言模型来解决复杂的长时间任务，为每个阶段提供工具名称和Python代码。

    

    变形物体操纵是机器人领域中最迷人又最艰巨的挑战之一。虽然先前的技术主要依赖于通过演示学习潜在动态，通常表示为粒子或图像之一，但存在一个重要限制：获取适当的演示，特别是对于长时间任务，可能是困难的。此外，完全基于演示进行学习可能会阻碍模型超越演示任务的能力。在这项工作中，我们介绍了一种无需演示的分层规划方法，能够处理复杂的长时间任务而无需任何训练。我们利用大型语言模型（LLMs）来表达与指定任务对应的高层、阶段-by-阶段计划。对于每个单独阶段，LLM提供工具的名称和Python代码，以制作中间子目标点云。

    arXiv:2311.02787v2 Announce Type: replace-cross  Abstract: Deformable object manipulation stands as one of the most captivating yet formidable challenges in robotics. While previous techniques have predominantly relied on learning latent dynamics through demonstrations, typically represented as either particles or images, there exists a pertinent limitation: acquiring suitable demonstrations, especially for long-horizon tasks, can be elusive. Moreover, basing learning entirely on demonstrations can hamper the model's ability to generalize beyond the demonstrated tasks. In this work, we introduce a demonstration-free hierarchical planning approach capable of tackling intricate long-horizon tasks without necessitating any training. We employ large language models (LLMs) to articulate a high-level, stage-by-stage plan corresponding to a specified task. For every individual stage, the LLM provides both the tool's name and the Python code to craft intermediate subgoal point clouds. With the
    

