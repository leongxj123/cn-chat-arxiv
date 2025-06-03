# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generating by Understanding: Neural Visual Generation with Logical Symbol Groundings.](http://arxiv.org/abs/2310.17451) | 这篇论文提出了一种神经符号学习方法，AbdGen，用于将知识推理系统与神经视觉生成模型集成。它解决了符号赋值和规则学习的问题，通过量化诱导方法实现可靠高效的符号赋值，通过对比元诱导方法实现精确的规则学习。 |

# 详细

[^1]: 通过理解生成：具有逻辑符号基础的神经视觉生成

    Generating by Understanding: Neural Visual Generation with Logical Symbol Groundings. (arXiv:2310.17451v1 [cs.AI])

    [http://arxiv.org/abs/2310.17451](http://arxiv.org/abs/2310.17451)

    这篇论文提出了一种神经符号学习方法，AbdGen，用于将知识推理系统与神经视觉生成模型集成。它解决了符号赋值和规则学习的问题，通过量化诱导方法实现可靠高效的符号赋值，通过对比元诱导方法实现精确的规则学习。

    

    尽管近年来神经视觉生成模型取得了很大的成功，但将其与强大的符号知识推理系统集成仍然是一个具有挑战性的任务。主要挑战有两个方面：一个是符号赋值，即将神经视觉生成器的潜在因素与知识推理系统中的有意义的符号进行绑定。另一个是规则学习，即学习新的规则，这些规则控制数据的生成过程，以增强知识推理系统。为了解决这些符号基础问题，我们提出了一种神经符号学习方法，Abductive Visual Generation (AbdGen)，用于基于诱导学习框架将逻辑编程系统与神经视觉生成模型集成起来。为了实现可靠高效的符号赋值，引入了量化诱导方法，通过语义编码本中的最近邻查找生成诱导提案。为了实现精确的规则学习，引入了对比元诱导方法。

    Despite the great success of neural visual generative models in recent years, integrating them with strong symbolic knowledge reasoning systems remains a challenging task. The main challenges are two-fold: one is symbol assignment, i.e. bonding latent factors of neural visual generators with meaningful symbols from knowledge reasoning systems. Another is rule learning, i.e. learning new rules, which govern the generative process of the data, to augment the knowledge reasoning systems. To deal with these symbol grounding problems, we propose a neural-symbolic learning approach, Abductive Visual Generation (AbdGen), for integrating logic programming systems with neural visual generative models based on the abductive learning framework. To achieve reliable and efficient symbol assignment, the quantized abduction method is introduced for generating abduction proposals by the nearest-neighbor lookups within semantic codebooks. To achieve precise rule learning, the contrastive meta-abduction
    

