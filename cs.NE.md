# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Neural Rewriting System to Solve Algorithmic Problems](https://arxiv.org/abs/2402.17407) | 提出了一种受重写系统启发的神经架构，用于学习算法任务，通过Selector、Solver和Combiner三个专门模块实现算法任务的简化，具有较好的外推能力 |
| [^2] | [Continual Developmental Neurosimulation Using Embodied Computational Agents.](http://arxiv.org/abs/2103.05753) | 通过使用发育Braitenberg Vehicles代理，我们提出了一种具有发育启发的学习代理设计，实现了计算自主性的具身经验和形态发生增长模拟，并考虑了发育轨迹在神经系统形态生成、发育学习和可塑性等方面的作用。 |

# 详细

[^1]: 用神经重写系统解决算法问题

    A Neural Rewriting System to Solve Algorithmic Problems

    [https://arxiv.org/abs/2402.17407](https://arxiv.org/abs/2402.17407)

    提出了一种受重写系统启发的神经架构，用于学习算法任务，通过Selector、Solver和Combiner三个专门模块实现算法任务的简化，具有较好的外推能力

    

    现代神经网络架构仍然难以学习需要系统应用组合规则来解决超出分布问题实例的算法程序。在这项工作中，我们提出了一种原创方法来学习受重写系统启发的算法任务，重写系统是符号人工智能中的经典框架。我们展示了重写系统可以被实现为一个由专门模块组成的神经架构：选择器识别要处理的目标子表达式，求解器通过计算相应的结果简化子表达式，组合器通过用提供的解决方案替换子表达式生成原始表达式的新版本。我们在三种涉及简化涉及列表、算术和代数表达式的符号公式的算法任务上评估我们的模型。我们测试了所提架构的外推能力

    arXiv:2402.17407v1 Announce Type: cross  Abstract: Modern neural network architectures still struggle to learn algorithmic procedures that require to systematically apply compositional rules to solve out-of-distribution problem instances. In this work, we propose an original approach to learn algorithmic tasks inspired by rewriting systems, a classic framework in symbolic artificial intelligence. We show that a rewriting system can be implemented as a neural architecture composed by specialized modules: the Selector identifies the target sub-expression to process, the Solver simplifies the sub-expression by computing the corresponding result, and the Combiner produces a new version of the original expression by replacing the sub-expression with the solution provided. We evaluate our model on three types of algorithmic tasks that require simplifying symbolic formulas involving lists, arithmetic, and algebraic expressions. We test the extrapolation capabilities of the proposed architectu
    
[^2]: 持续发展的神经仿真：基于具身计算代理的方法

    Continual Developmental Neurosimulation Using Embodied Computational Agents. (arXiv:2103.05753v2 [q-bio.NC] UPDATED)

    [http://arxiv.org/abs/2103.05753](http://arxiv.org/abs/2103.05753)

    通过使用发育Braitenberg Vehicles代理，我们提出了一种具有发育启发的学习代理设计，实现了计算自主性的具身经验和形态发生增长模拟，并考虑了发育轨迹在神经系统形态生成、发育学习和可塑性等方面的作用。

    

    通过综合发育生物学、认知科学和计算建模，我们可以获得很多知识。我们的研究目标是基于Braitenberg Vehicles设计开发受发育启发的学习代理。利用这些代理体现了计算自主性的具身特性，不断靠近对具身经验和形态发生增长作为认知发展能力组成部分的建模。我们考虑生物和认知发展对成年表型生成和可用发展路径的影响。持续发展神经仿真使我们能够考虑发育轨迹在连接神经系统形态发生、发育学习和可塑性等相关现象中所起的作用。由于与持续学习紧密相关，我们的方法与发育具身紧密集成，可以使用一种称为发育Braitenberg Vehicles (dBVs)的代理来实现。

    There is much to learn through synthesis of Developmental Biology, Cognitive Science and Computational Modeling. Our path forward is to present a design for developmentally-inspired learning agents based on Braitenberg Vehicles. Using these agents to exemplify the embodied nature of computational autonomy, we move closer to modeling embodied experience and morphogenetic growth as components of cognitive developmental capacity. We consider biological and cognitive development which influence the generation of adult phenotypes and the contingency of available developmental pathways. Continual developmental neurosimulation allows us to consider the role of developmental trajectories in bridging the related phenomena of nervous system morphogenesis, developmental learning, and plasticity. Being closely tied to continual learning, our approach is tightly integrated with developmental embodiment, and can be implemented using a type of agent called developmental Braitenberg Vehicles (dBVs). T
    

