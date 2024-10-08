# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [IR-Aware ECO Timing Optimization Using Reinforcement Learning](https://arxiv.org/abs/2402.07781) | 本文提出了一种使用强化学习进行IR感知的ECO时序优化的方法，该方法通过门尺寸调整纠正由IR降低引起的时序退化，并且相较于传统方法在性能和运行时间上都具有优势。 |

# 详细

[^1]: 使用强化学习进行IR感知的ECO时序优化

    IR-Aware ECO Timing Optimization Using Reinforcement Learning

    [https://arxiv.org/abs/2402.07781](https://arxiv.org/abs/2402.07781)

    本文提出了一种使用强化学习进行IR感知的ECO时序优化的方法，该方法通过门尺寸调整纠正由IR降低引起的时序退化，并且相较于传统方法在性能和运行时间上都具有优势。

    

    在晚期阶段的工程变更订单（ECOs）通过最小的设计修复来从过多的IR降低导致的时序偏移中恢复。本文将IR感知的时序分析和使用强化学习（RL）进行ECO时序优化相结合。该方法在物理设计和功耗网格综合之后运行，并通过门尺寸调整纠正由IR降低引起的时序退化。它将拉格朗日松弛（LR）技术融入一种新颖的RL框架中，该框架训练一个关系图卷积网络（R-GCN）代理，按顺序调整门尺寸以修复时序违规。R-GCN代理优于传统的仅使用LR的算法：在开放式45nm工艺中，它将延迟-面积权衡曲线的帕累托前沿向左移动，并通过在等质量时使用训练模型进行快速推理，节省运行时间。RL模型可在时序规范间转移，并可通过零样本学习或微调在未见设计上转移。

    Engineering change orders (ECOs) in late stages make minimal design fixes to recover from timing shifts due to excessive IR drops. This paper integrates IR-drop-aware timing analysis and ECO timing optimization using reinforcement learning (RL). The method operates after physical design and power grid synthesis, and rectifies IR-drop-induced timing degradation through gate sizing. It incorporates the Lagrangian relaxation (LR) technique into a novel RL framework, which trains a relational graph convolutional network (R-GCN) agent to sequentially size gates to fix timing violations. The R-GCN agent outperforms a classical LR-only algorithm: in an open 45nm technology, it (a) moves the Pareto front of the delay-area tradeoff curve to the left and (b) saves runtime over the classical method by running fast inference using trained models at iso-quality. The RL model is transferable across timing specifications, and transferable to unseen designs with zero-shot learning or fine tuning.
    

