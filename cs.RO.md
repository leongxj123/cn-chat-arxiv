# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SCANet: Correcting LEGO Assembly Errors with Self-Correct Assembly Network](https://arxiv.org/abs/2403.18195) | 介绍了单步组装错误校正任务和LEGO错误校正组装数据集（LEGO-ECA），提出了用于这一任务的自校正组装网络（SCANet）。 |
| [^2] | [Simplifying Complex Observation Models in Continuous POMDP Planning with Probabilistic Guarantees and Practice.](http://arxiv.org/abs/2311.07745) | 本研究在解决具有高维度和连续观测的部分可观测马尔可夫决策过程中，提出了一种基于统计总变差距离的新型概率界限，能够简化观测模型并保证解决方案的质量。 |

# 详细

[^1]: 用自校正组装网络纠正LEGO组装错误

    SCANet: Correcting LEGO Assembly Errors with Self-Correct Assembly Network

    [https://arxiv.org/abs/2403.18195](https://arxiv.org/abs/2403.18195)

    介绍了单步组装错误校正任务和LEGO错误校正组装数据集（LEGO-ECA），提出了用于这一任务的自校正组装网络（SCANet）。

    

    在机器人学和3D视觉中，自主组装面临着重大挑战，尤其是确保组装正确性。主流方法如MEPNet目前专注于基于手动提供的图像进行组件组装。然而，这些方法在需要长期规划的任务中往往难以取得满意的结果。在同一时间，我们观察到整合自校正模块可以在一定程度上缓解这些问题。受此问题启发，我们引入了单步组装错误校正任务，其中涉及识别和纠正组件组装错误。为支持这一领域的研究，我们提出了LEGO错误校正组装数据集（LEGO-ECA），包括用于组装步骤和组装失败实例的手动图像。此外，我们提出了自校正组装网络（SCANet），这是一种新颖的方法来解决这一任务。SCANet将组装的部件视为查询，

    arXiv:2403.18195v1 Announce Type: cross  Abstract: Autonomous assembly in robotics and 3D vision presents significant challenges, particularly in ensuring assembly correctness. Presently, predominant methods such as MEPNet focus on assembling components based on manually provided images. However, these approaches often fall short in achieving satisfactory results for tasks requiring long-term planning. Concurrently, we observe that integrating a self-correction module can partially alleviate such issues. Motivated by this concern, we introduce the single-step assembly error correction task, which involves identifying and rectifying misassembled components. To support research in this area, we present the LEGO Error Correction Assembly Dataset (LEGO-ECA), comprising manual images for assembly steps and instances of assembly failures. Additionally, we propose the Self-Correct Assembly Network (SCANet), a novel method to address this task. SCANet treats assembled components as queries, de
    
[^2]: 在具有概率保证和实践的连续POMDP规划中简化复杂的观测模型

    Simplifying Complex Observation Models in Continuous POMDP Planning with Probabilistic Guarantees and Practice. (arXiv:2311.07745v4 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2311.07745](http://arxiv.org/abs/2311.07745)

    本研究在解决具有高维度和连续观测的部分可观测马尔可夫决策过程中，提出了一种基于统计总变差距离的新型概率界限，能够简化观测模型并保证解决方案的质量。

    

    解决具有高维度和连续观测（如相机图像）的部分可观测马尔可夫决策过程(POMDP)对于许多实际机器人和规划问题是必需的。最近的研究建议使用机器学习的概率模型作为观测模型，但它们目前在线部署时计算成本过高。我们探讨了在规划中使用简化观测模型的影响，同时保持对解决方案质量的形式化保证。我们的主要贡献是一种基于简化模型的统计总变差距离的新型概率界限。我们证明，通过推广最近的粒子置信度MDP集中界限的结果，它将理论POMDP值与简化模型下的实际规划值进行了约束。我们的计算可以分为离线和在线部分，并且我们可以得到形式化的保证，而无需

    Solving partially observable Markov decision processes (POMDPs) with high dimensional and continuous observations, such as camera images, is required for many real life robotics and planning problems. Recent researches suggested machine learned probabilistic models as observation models, but their use is currently too computationally expensive for online deployment. We deal with the question of what would be the implication of using simplified observation models for planning, while retaining formal guarantees on the quality of the solution. Our main contribution is a novel probabilistic bound based on a statistical total variation distance of the simplified model. We show that it bounds the theoretical POMDP value w.r.t. original model, from the empirical planned value with the simplified model, by generalizing recent results of particle-belief MDP concentration bounds. Our calculations can be separated into offline and online parts, and we arrive at formal guarantees without having to
    

