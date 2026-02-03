# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [InterDreamer: Zero-Shot Text to 3D Dynamic Human-Object Interaction](https://arxiv.org/abs/2403.19652) | 通过解耦交互语义和动态，本文展示了在没有直接训练文本-交互对数据的情况下生成人物-物体交互的潜力。 |
| [^2] | [Factorized Tensor Networks for Multi-Task and Multi-Domain Learning.](http://arxiv.org/abs/2310.06124) | 本文提出了一种分解张量网络（FTN），它可以克服多任务多领域学习中的共享信息利用挑战，并在准确性、存储成本、计算量和样本复杂度等方面实现高效率。实验结果表明，FTN相对于现有方法需要更少的任务特定参数，并且可以适应大量的目标领域和任务。 |

# 详细

[^1]: InterDreamer：零样本文本到三维动态人物-物体交互

    InterDreamer: Zero-Shot Text to 3D Dynamic Human-Object Interaction

    [https://arxiv.org/abs/2403.19652](https://arxiv.org/abs/2403.19652)

    通过解耦交互语义和动态，本文展示了在没有直接训练文本-交互对数据的情况下生成人物-物体交互的潜力。

    

    arXiv:2403.19652v1 宣布类型：跨领域 摘要：在广泛的动作捕捉数据和相应的文本注释上训练的扩散模型已经显著推动了文本条件的人体运动生成。然而，将这种成功延伸到三维动态人物-物体交互（HOI）生成面临着显著挑战，主要是由于缺乏大规模交互数据和与这些交互一致的全面描述。本文采取了行动，并展示了在没有直接训练文本-交互对数据的情况下生成人物-物体交互的潜力。我们在实现这一点的关键见解是交互语义和动态可以解耦。无法通过监督训练学习交互语义，我们转而利用预训练的大型模型，将来自大型语言模型和文本到运动模型的知识相辅相成。尽管这样的知识提供了对交互语义的高级控制，但不能提供到不成对交互文本的直接学习。

    arXiv:2403.19652v1 Announce Type: cross  Abstract: Text-conditioned human motion generation has experienced significant advancements with diffusion models trained on extensive motion capture data and corresponding textual annotations. However, extending such success to 3D dynamic human-object interaction (HOI) generation faces notable challenges, primarily due to the lack of large-scale interaction data and comprehensive descriptions that align with these interactions. This paper takes the initiative and showcases the potential of generating human-object interactions without direct training on text-interaction pair data. Our key insight in achieving this is that interaction semantics and dynamics can be decoupled. Being unable to learn interaction semantics through supervised training, we instead leverage pre-trained large models, synergizing knowledge from a large language model and a text-to-motion model. While such knowledge offers high-level control over interaction semantics, it c
    
[^2]: 分解张量网络用于多任务和多领域学习

    Factorized Tensor Networks for Multi-Task and Multi-Domain Learning. (arXiv:2310.06124v1 [cs.LG])

    [http://arxiv.org/abs/2310.06124](http://arxiv.org/abs/2310.06124)

    本文提出了一种分解张量网络（FTN），它可以克服多任务多领域学习中的共享信息利用挑战，并在准确性、存储成本、计算量和样本复杂度等方面实现高效率。实验结果表明，FTN相对于现有方法需要更少的任务特定参数，并且可以适应大量的目标领域和任务。

    

    多任务和多领域学习方法旨在使用单个统一的网络共同学习多个任务/领域，或者先后学习它们。关键挑战和机会是利用任务和领域之间的共享信息，提高统一网络的效率，包括准确性、存储成本、计算量或样本复杂度。本文提出了一种分解张量网络（FTN），可以通过增加少量附加参数实现与独立单任务/领域网络相当的准确性。FTN使用源模型的冻结主干网络，并逐步添加任务/领域特定的低秩张量因子到共享的冻结网络中。这种方法可以适应大量目标领域和任务，而不会出现灾难性遗忘。此外，与现有方法相比，FTN需要较少的任务特定参数。我们在广泛使用的多领域和多任务数据集上进行了实验。

    Multi-task and multi-domain learning methods seek to learn multiple tasks/domains, jointly or one after another, using a single unified network. The key challenge and opportunity is to exploit shared information across tasks and domains to improve the efficiency of the unified network. The efficiency can be in terms of accuracy, storage cost, computation, or sample complexity. In this paper, we propose a factorized tensor network (FTN) that can achieve accuracy comparable to independent single-task/domain networks with a small number of additional parameters. FTN uses a frozen backbone network from a source model and incrementally adds task/domain-specific low-rank tensor factors to the shared frozen network. This approach can adapt to a large number of target domains and tasks without catastrophic forgetting. Furthermore, FTN requires a significantly smaller number of task-specific parameters compared to existing methods. We performed experiments on widely used multi-domain and multi-
    

