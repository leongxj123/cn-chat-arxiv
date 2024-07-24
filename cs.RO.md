# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Development of Compositionality and Generalization through Interactive Learning of Language and Action of Robots](https://arxiv.org/abs/2403.19995) | 提出了一个融合视觉、本体感知和语言的大脑启发式神经网络模型，通过预测编码和主动推断的框架，基于自由能原理，实现了语言组合性和感觉运动技能的联合发展。 |
| [^2] | [AutoGPT+P: Affordance-based Task Planning with Large Language Models](https://arxiv.org/abs/2402.10778) | 提出了AutoGPT+P，它结合了基于Affordance的场景表示和规划系统，可以解决具有不确定性的任务规划问题。 |
| [^3] | [Constrained Stein Variational Trajectory Optimization.](http://arxiv.org/abs/2308.12110) | CSVTO是一种受限斯坦变分轨迹优化算法，它通过斯坦变分梯度下降方法生成多样的约束满足轨迹集合，提高了在具有任意约束的问题中的优化性能和鲁棒性。 |

# 详细

[^1]: 通过交互学习语言和机器人动作实现组合性和泛化能力的发展

    Development of Compositionality and Generalization through Interactive Learning of Language and Action of Robots

    [https://arxiv.org/abs/2403.19995](https://arxiv.org/abs/2403.19995)

    提出了一个融合视觉、本体感知和语言的大脑启发式神经网络模型，通过预测编码和主动推断的框架，基于自由能原理，实现了语言组合性和感觉运动技能的联合发展。

    

    人类擅长将学到的行为应用于未学习过的情境。这种泛化行为的一个关键组成部分是我们能够将整体分解成可重复利用的部分的能力，即组合性。机器人领域的一个基本问题是涉及这种特征。“在个体只学习部分语言组合及其相应的感觉运动模式时，如何通过联想学习同时发展语言的组合性和感觉运动技能？”为了解决这个问题，我们提出了一个融合视觉、本体感知和语言的大脑启发式神经网络模型，将其纳入基于自由能原理的预测编码和主动推断框架中。通过与机器人手臂进行的各种模拟实验评估了这个模型的有效性和能力。我们的结果表明，在学习中对于遗忘。

    arXiv:2403.19995v1 Announce Type: new  Abstract: Humans excel at applying learned behavior to unlearned situations. A crucial component of this generalization behavior is our ability to compose/decompose a whole into reusable parts, an attribute known as compositionality. One of the fundamental questions in robotics concerns this characteristic. "How can linguistic compositionality be developed concomitantly with sensorimotor skills through associative learning, particularly when individuals only learn partial linguistic compositions and their corresponding sensorimotor patterns?" To address this question, we propose a brain-inspired neural network model that integrates vision, proprioception, and language into a framework of predictive coding and active inference, based on the free-energy principle. The effectiveness and capabilities of this model were assessed through various simulation experiments conducted with a robot arm. Our results show that generalization in learning to unlear
    
[^2]: 基于Affordance的任务规划与大型语言模型的AutoGPT+P

    AutoGPT+P: Affordance-based Task Planning with Large Language Models

    [https://arxiv.org/abs/2402.10778](https://arxiv.org/abs/2402.10778)

    提出了AutoGPT+P，它结合了基于Affordance的场景表示和规划系统，可以解决具有不确定性的任务规划问题。

    

    最近关于任务规划的一些新进展利用了大型语言模型（LLMs），通过将这些模型与经典规划算法结合起来来提高泛化能力，以解决它们在推理能力上固有的局限性。然而，这些方法面临着动态捕捉任务规划问题的初始状态的挑战。为了缓解这一问题，我们提出了AutoGPT+P，这是一个系统，将基于Affordance的场景表示与一个规划系统相结合。Affordance包括了一个代理在环境中和其中存在的物体上的动作可能性。因此，从基于Affordance的场景表示中推导出规划域，允许使用任意对象进行符号规划。AutoGPT+P利用这种表示来为用户用自然语言指定的任务制定和执行计划。除了在封闭世界假设下解决规划任务外，AutoGPT+P还可以处理具有不确定性的规划任务。

    arXiv:2402.10778v1 Announce Type: cross  Abstract: Recent advances in task planning leverage Large Language Models (LLMs) to improve generalizability by combining such models with classical planning algorithms to address their inherent limitations in reasoning capabilities. However, these approaches face the challenge of dynamically capturing the initial state of the task planning problem. To alleviate this issue, we propose AutoGPT+P, a system that combines an affordance-based scene representation with a planning system. Affordances encompass the action possibilities of an agent on the environment and objects present in it. Thus, deriving the planning domain from an affordance-based scene representation allows symbolic planning with arbitrary objects. AutoGPT+P leverages this representation to derive and execute a plan for a task specified by the user in natural language. In addition to solving planning tasks under a closed-world assumption, AutoGPT+P can also handle planning with inc
    
[^3]: 受限斯坦变分轨迹优化

    Constrained Stein Variational Trajectory Optimization. (arXiv:2308.12110v1 [cs.RO] CROSS LISTED)

    [http://arxiv.org/abs/2308.12110](http://arxiv.org/abs/2308.12110)

    CSVTO是一种受限斯坦变分轨迹优化算法，它通过斯坦变分梯度下降方法生成多样的约束满足轨迹集合，提高了在具有任意约束的问题中的优化性能和鲁棒性。

    

    我们提出了一种受限斯坦变分轨迹优化（CSVTO）算法，用于在一组轨迹上进行带约束的轨迹优化。我们将受限轨迹优化视为一种新颖的对轨迹分布约束的函数最小化形式，避免将约束视为目标函数的惩罚，从而使我们能够生成多样的满足约束的轨迹集合。我们的方法使用斯坦变分梯度下降（SVGD）寻找一组粒子，近似表示一个低成本轨迹的分布，并遵守约束。CSVTO适用于具有任意等式和不等式约束的问题，并包括一种新颖的粒子重新采样步骤来避免局部最小值。通过明确生成多样的轨迹集合，CSVTO能够更好地避免不良的局部最小值，并且对初始化更具鲁棒性。我们证明，CSVTO在具有高度约束的挑战性问题上优于基线方法。

    We present Constrained Stein Variational Trajectory Optimization (CSVTO), an algorithm for performing trajectory optimization with constraints on a set of trajectories in parallel. We frame constrained trajectory optimization as a novel form of constrained functional minimization over trajectory distributions, which avoids treating the constraints as a penalty in the objective and allows us to generate diverse sets of constraint-satisfying trajectories. Our method uses Stein Variational Gradient Descent (SVGD) to find a set of particles that approximates a distribution over low-cost trajectories while obeying constraints. CSVTO is applicable to problems with arbitrary equality and inequality constraints and includes a novel particle resampling step to escape local minima. By explicitly generating diverse sets of trajectories, CSVTO is better able to avoid poor local minima and is more robust to initialization. We demonstrate that CSVTO outperforms baselines in challenging highly-constr
    

