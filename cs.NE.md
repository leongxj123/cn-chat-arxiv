# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Extension of Recurrent Kernels to different Reservoir Computing topologies.](http://arxiv.org/abs/2401.14557) | 该研究通过提供特定RC体系结构与相应循环内核形式等价性的经验分析，填补了Leaky RC、Sparse RC和Deep RC等已建立的RC范例尚未进行分析的空白。此外，研究还揭示了稀疏连接在RC体系结构中的作用，并提出了一种依赖储备大小的最佳稀疏性水平。最后，该研究的系统分析表明，在Deep RC模型中，通过减小尺寸的连续储备可以更好地实现收敛。 |
| [^2] | [Synergizing Quality-Diversity with Descriptor-Conditioned Reinforcement Learning.](http://arxiv.org/abs/2401.08632) | 将质量多样性优化与描述符条件加强学习相结合，以克服进化算法的局限性，并在生成既多样又高性能的解决方案集合方面取得成功。 |
| [^3] | [A Prefrontal Cortex-inspired Architecture for Planning in Large Language Models.](http://arxiv.org/abs/2310.00194) | 这个论文提出了一个受前额叶皮层启发的大型语言模型规划架构，利用多个基于LLM的模块实现规划的自主协调，从而在处理需要多步推理或目标导向规划的任务时取得了较好的效果。 |

# 详细

[^1]: 不同的循环内核拓展到不同的储备计算拓扑的研究

    Extension of Recurrent Kernels to different Reservoir Computing topologies. (arXiv:2401.14557v1 [cs.LG])

    [http://arxiv.org/abs/2401.14557](http://arxiv.org/abs/2401.14557)

    该研究通过提供特定RC体系结构与相应循环内核形式等价性的经验分析，填补了Leaky RC、Sparse RC和Deep RC等已建立的RC范例尚未进行分析的空白。此外，研究还揭示了稀疏连接在RC体系结构中的作用，并提出了一种依赖储备大小的最佳稀疏性水平。最后，该研究的系统分析表明，在Deep RC模型中，通过减小尺寸的连续储备可以更好地实现收敛。

    

    近年来，由于其快速高效的计算能力，储备计算（RC）变得越来越受欢迎。标准的RC在渐近极限下已被证明与循环内核等效，这有助于分析其表达能力。然而，许多已建立的RC范例，如Leaky RC、Sparse RC和Deep RC，尚未以这种方式进行分析。本研究旨在通过提供特定RC体系结构与相应循环内核形式等价性的经验分析来填补这一空白。我们通过改变每个体系结构中实施的激活函数进行收敛研究。我们的研究还揭示了稀疏连接在RC体系结构中的作用，并提出了一种依赖储备大小的最佳稀疏性水平。此外，我们的系统分析表明，在Deep RC模型中，通过减小尺寸的连续储备可以更好地实现收敛。

    Reservoir Computing (RC) has become popular in recent years due to its fast and efficient computational capabilities. Standard RC has been shown to be equivalent in the asymptotic limit to Recurrent Kernels, which helps in analyzing its expressive power. However, many well-established RC paradigms, such as Leaky RC, Sparse RC, and Deep RC, are yet to be analyzed in such a way. This study aims to fill this gap by providing an empirical analysis of the equivalence of specific RC architectures with their corresponding Recurrent Kernel formulation. We conduct a convergence study by varying the activation function implemented in each architecture. Our study also sheds light on the role of sparse connections in RC architectures and propose an optimal sparsity level that depends on the reservoir size. Furthermore, our systematic analysis shows that in Deep RC models, convergence is better achieved with successive reservoirs of decreasing sizes.
    
[^2]: 将质量多样性与描述符条件加强学习相结合

    Synergizing Quality-Diversity with Descriptor-Conditioned Reinforcement Learning. (arXiv:2401.08632v1 [cs.NE])

    [http://arxiv.org/abs/2401.08632](http://arxiv.org/abs/2401.08632)

    将质量多样性优化与描述符条件加强学习相结合，以克服进化算法的局限性，并在生成既多样又高性能的解决方案集合方面取得成功。

    

    智能的基本特征之一是找到新颖和有创造性的解决方案来解决给定的挑战或适应未预料到的情况。质量多样性优化是一类进化算法，可以生成既多样又高性能的解决方案集合。其中，MAP-Elites是一个著名的例子，已成功应用于各种领域，包括进化机器人学。然而，MAP-Elites通过遗传算法的随机突变进行发散搜索，因此仅限于进化低维解决方案的种群。PGA-MAP-Elites通过受深度强化学习启发的基于梯度的变异算子克服了这一限制，从而实现了大型神经网络的进化。尽管在许多环境中性能优秀，但PGA-MAP-Elites在一些任务中失败，其中基于梯度的变异算子的收敛搜索阻碍了多样性。在这项工作中，我们...

    A fundamental trait of intelligence involves finding novel and creative solutions to address a given challenge or to adapt to unforeseen situations. Reflecting this, Quality-Diversity optimization is a family of Evolutionary Algorithms, that generates collections of both diverse and high-performing solutions. Among these, MAP-Elites is a prominent example, that has been successfully applied to a variety of domains, including evolutionary robotics. However, MAP-Elites performs a divergent search with random mutations originating from Genetic Algorithms, and thus, is limited to evolving populations of low-dimensional solutions. PGA-MAP-Elites overcomes this limitation using a gradient-based variation operator inspired by deep reinforcement learning which enables the evolution of large neural networks. Although high-performing in many environments, PGA-MAP-Elites fails on several tasks where the convergent search of the gradient-based variation operator hinders diversity. In this work, we
    
[^3]: 受前额叶皮层启发的大型语言模型规划架构

    A Prefrontal Cortex-inspired Architecture for Planning in Large Language Models. (arXiv:2310.00194v1 [cs.AI])

    [http://arxiv.org/abs/2310.00194](http://arxiv.org/abs/2310.00194)

    这个论文提出了一个受前额叶皮层启发的大型语言模型规划架构，利用多个基于LLM的模块实现规划的自主协调，从而在处理需要多步推理或目标导向规划的任务时取得了较好的效果。

    

    大型语言模型（LLM）在许多任务上展现出惊人的性能，但它们经常在需要多步推理或目标导向规划的任务中遇到困难。为了解决这个问题，我们从人脑中获取灵感，即通过前额叶皮层（PFC）中专门模块的重复交互来完成规划。这些模块执行冲突监测、状态预测、状态评估、任务分解和任务协调等功能。我们发现LLM有时能够单独执行这些功能，但在服务于一个目标时往往难以自主协调它们。因此，我们提出了一个带有多个基于LLM（GPT-4）模块的黑盒架构。该架构通过专门的PFC启发模块的交互将一个更大的问题分解为多个对LLM的简短自动调用，从而改善规划能力。我们在两个具有挑战性的规划任务上评估了组合架构。

    Large language models (LLMs) demonstrate impressive performance on a wide variety of tasks, but they often struggle with tasks that require multi-step reasoning or goal-directed planning. To address this, we take inspiration from the human brain, in which planning is accomplished via the recurrent interaction of specialized modules in the prefrontal cortex (PFC). These modules perform functions such as conflict monitoring, state prediction, state evaluation, task decomposition, and task coordination. We find that LLMs are sometimes capable of carrying out these functions in isolation, but struggle to autonomously coordinate them in the service of a goal. Therefore, we propose a black box architecture with multiple LLM-based (GPT-4) modules. The architecture improves planning through the interaction of specialized PFC-inspired modules that break down a larger problem into multiple brief automated calls to the LLM. We evaluate the combined architecture on two challenging planning tasks -
    

