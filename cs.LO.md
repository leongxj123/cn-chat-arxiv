# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NeuRes: Learning Proofs of Propositional Satisfiability](https://arxiv.org/abs/2402.08365) | NeuRes是一种神经符号证明为基础的SAT解析器，能够证明不可满足性并加速找到可满足真值分配的过程。 |
| [^2] | [MacroSwarm: A Field-based Compositional Framework for Swarm Programming.](http://arxiv.org/abs/2401.10969) | MacroSwarm是一种基于场的群体编程框架，通过可组合的功能模块实现复杂的群体行为，通过将感知场映射为执行目标场，提供了一种系统化的设计和实现群体行为的方法。 |

# 详细

[^1]: NeuRes: 学习命题可满足性的证明

    NeuRes: Learning Proofs of Propositional Satisfiability

    [https://arxiv.org/abs/2402.08365](https://arxiv.org/abs/2402.08365)

    NeuRes是一种神经符号证明为基础的SAT解析器，能够证明不可满足性并加速找到可满足真值分配的过程。

    

    我们介绍了一种神经符号证明为基础的SAT解析器NeuRes。与其他神经SAT解算法不同，NeuRes能够证明不可满足性，而不仅仅是预测它。NeuRes通过采用命题推理来证明不可满足性并加速在不可满足和可满足公式中找到满足真值分配的过程。为了实现这一点，我们提出了一种新颖的架构，它结合了图神经网络和指针网络的元素，从动态图结构中自动选择节点对，这对于生成解析证明是至关重要的。我们使用与NeuroSAT相同的随机公式分布编制了一个包含教师证明和真值分配的数据集，对我们的模型进行训练和评估。在实验证明中，我们展示了NeuRes在不同分布上比NeuroSAT解决更多的测试公式，并且需要更少的数据。

    We introduce NeuRes, a neuro-symbolic proof-based SAT solver. Unlike other neural SAT solving methods, NeuRes is capable of proving unsatisfiability as opposed to merely predicting it. By design, NeuRes operates in a certificate-driven fashion by employing propositional resolution to prove unsatisfiability and to accelerate the process of finding satisfying truth assignments in case of unsat and sat formulas, respectively. To realize this, we propose a novel architecture that adapts elements from Graph Neural Networks and Pointer Networks to autoregressively select pairs of nodes from a dynamic graph structure, which is essential to the generation of resolution proofs. Our model is trained and evaluated on a dataset of teacher proofs and truth assignments that we compiled with the same random formula distribution used by NeuroSAT. In our experiments, we show that NeuRes solves more test formulas than NeuroSAT by a rather wide margin on different distributions while being much more data
    
[^2]: MacroSwarm: 一种基于场的组合框架用于群体编程

    MacroSwarm: A Field-based Compositional Framework for Swarm Programming. (arXiv:2401.10969v1 [cs.AI])

    [http://arxiv.org/abs/2401.10969](http://arxiv.org/abs/2401.10969)

    MacroSwarm是一种基于场的群体编程框架，通过可组合的功能模块实现复杂的群体行为，通过将感知场映射为执行目标场，提供了一种系统化的设计和实现群体行为的方法。

    

    群体行为工程是一项旨在研究协调简单智能体团体内计算和行动的方法和技术，以实现复杂的全局目标，如图案形成、集体移动、聚类和分布式感知。尽管在群体（无人机、机器人、车辆）分析和工程方面取得了一些进展，但仍然需要通用的设计和实现方法和工具，以系统化的方式定义复杂的群体行为。为了对此做出贡献，本文提出了一种新的基于场的协调方法，称为MacroSwarm，以可重用且完全可组合的功能模块为基础，嵌入集体计算和协调。基于集成计算的宏编程范式，MacroSwarm提出了将每个群体行为块表示为将感知场映射为执行目标场的纯函数的思路。

    Swarm behaviour engineering is an area of research that seeks to investigate methods and techniques for coordinating computation and action within groups of simple agents to achieve complex global goals like pattern formation, collective movement, clustering, and distributed sensing. Despite recent progress in the analysis and engineering of swarms (of drones, robots, vehicles), there is still a need for general design and implementation methods and tools that can be used to define complex swarm behaviour in a principled way. To contribute to this quest, this article proposes a new field-based coordination approach, called MacroSwarm, to design and program swarm behaviour in terms of reusable and fully composable functional blocks embedding collective computation and coordination. Based on the macroprogramming paradigm of aggregate computing, MacroSwarm builds on the idea of expressing each swarm behaviour block as a pure function mapping sensing fields into actuation goal fields, e.g.
    

