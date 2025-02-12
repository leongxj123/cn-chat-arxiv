# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Denoising Task Difficulty-based Curriculum for Training Diffusion Models](https://arxiv.org/abs/2403.10348) | 研究通过全面研究任务难度，发现较早时间步长的去噪任务更具挑战性，提出了基于去噪任务难度的渐进式课程训练方法。 |

# 详细

[^1]: 基于去噪任务难度的渐进式课程训练扩散模型

    Denoising Task Difficulty-based Curriculum for Training Diffusion Models

    [https://arxiv.org/abs/2403.10348](https://arxiv.org/abs/2403.10348)

    研究通过全面研究任务难度，发现较早时间步长的去噪任务更具挑战性，提出了基于去噪任务难度的渐进式课程训练方法。

    

    基于扩散的生成模型已成为生成建模领域强大的工具。尽管对各个时间步长和噪声水平之间的去噪进行了广泛研究，但关于去噪任务的相对难度仍存在争议。我们的研究对任务难度进行了全面的研究，重点关注收敛行为和时间步长间连续概率分布的相对熵变化。我们的观察显示，较早时间步长的去噪存在收敛缓慢和较高的相对熵，表明在这些较低时间步长上任务难度增加。基于这些观察，我们引入了一种由易到难的学习方案，借鉴渐进式学习的思想。

    arXiv:2403.10348v1 Announce Type: cross  Abstract: Diffusion-based generative models have emerged as powerful tools in the realm of generative modeling. Despite extensive research on denoising across various timesteps and noise levels, a conflict persists regarding the relative difficulties of the denoising tasks. While various studies argue that lower timesteps present more challenging tasks, others contend that higher timesteps are more difficult. To address this conflict, our study undertakes a comprehensive examination of task difficulties, focusing on convergence behavior and changes in relative entropy between consecutive probability distributions across timesteps. Our observational study reveals that denoising at earlier timesteps poses challenges characterized by slower convergence and higher relative entropy, indicating increased task difficulty at these lower timesteps. Building on these observations, we introduce an easy-to-hard learning scheme, drawing from curriculum learn
    

