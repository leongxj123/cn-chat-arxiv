# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NavCoT: Boosting LLM-Based Vision-and-Language Navigation via Learning Disentangled Reasoning](https://arxiv.org/abs/2403.07376) | 本文提出了一种名为NavCoT的新策略，在视觉与语言导航中通过学习解耦推理，实现了自主导航决策，有效减轻了领域差距。 |

# 详细

[^1]: NavCoT: 通过学习解耦推理提升基于LLM的视觉与语言导航

    NavCoT: Boosting LLM-Based Vision-and-Language Navigation via Learning Disentangled Reasoning

    [https://arxiv.org/abs/2403.07376](https://arxiv.org/abs/2403.07376)

    本文提出了一种名为NavCoT的新策略，在视觉与语言导航中通过学习解耦推理，实现了自主导航决策，有效减轻了领域差距。

    

    视觉与语言导航(VLN)作为具有重要研究价值的具身人工智能问题，需要一个具身代理根据自然语言指示穿越复杂的3D环境。最近的研究突出了大型语言模型(LLMs)在VLN中提高导航推理准确性和可解释性的潜力。然而，它们主要在离线方式下的使用通常在VLN任务和LLM训练语料库之间遭受显著的领域差距。本文引入了一种名为导航思维链(NavCoT)的新型策略，我们通过完成领域内高效参数训练，实现自主导航决策，有效减轻领域差距的成本。具体地，在每个时间步，LLM被提示通过作为世界模型来预测导航思维链：1)根据

    arXiv:2403.07376v1 Announce Type: cross  Abstract: Vision-and-Language Navigation (VLN), as a crucial research problem of Embodied AI, requires an embodied agent to navigate through complex 3D environments following natural language instructions. Recent research has highlighted the promising capacity of large language models (LLMs) in VLN by improving navigational reasoning accuracy and interpretability. However, their predominant use in an offline manner usually suffers from substantial domain gap between the VLN task and the LLM training corpus. This paper introduces a novel strategy called Navigational Chain-of-Thought (NavCoT), where we fulfill parameter-efficient in-domain training to enable self-guided navigational decision, leading to a significant mitigation of the domain gap in a cost-effective manner. Specifically, at each timestep, the LLM is prompted to forecast the navigational chain-of-thought by: 1) acting as a world model to imagine the next observation according to the
    

