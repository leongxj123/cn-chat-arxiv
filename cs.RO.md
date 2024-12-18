# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FootstepNet: an Efficient Actor-Critic Method for Fast On-line Bipedal Footstep Planning and Forecasting](https://arxiv.org/abs/2403.12589) | 提出了一种基于深度强化学习技术的高效足部规划方法，在线推理的计算需求极低，无启发式，并依赖连续的动作生成可行的足步。 |
| [^2] | [Evaluating Decision Optimality of Autonomous Driving via Metamorphic Testing](https://arxiv.org/abs/2402.18393) | 本文着重于评估自动驾驶系统的决策质量，提出了检测非最佳决策场景的方法，通过新颖的变形关系暴露最佳决策违规。 |
| [^3] | [Multi-Object Graph Affordance Network: Enabling Goal-Oriented Planning through Compound Object Affordances.](http://arxiv.org/abs/2309.10426) | 多对象图形可用性网络（MOGAN）模型化了复合对象的可用性，预测了将新对象放置在复合对象上的效果。在构建具有特定高度或属性的塔等任务中，我们使用基于搜索的规划找到具有适当可用性的对象堆叠操作的序列。我们的系统能够正确建模非常复杂的复合对象的可用性，并在模拟和真实环境中展示了其适用性。 |

# 详细

[^1]: FootstepNet: 一种用于快速在线双足足部步态规划和预测的高效演员-评论方法

    FootstepNet: an Efficient Actor-Critic Method for Fast On-line Bipedal Footstep Planning and Forecasting

    [https://arxiv.org/abs/2403.12589](https://arxiv.org/abs/2403.12589)

    提出了一种基于深度强化学习技术的高效足部规划方法，在线推理的计算需求极低，无启发式，并依赖连续的动作生成可行的足步。

    

    设计一个人形运动控制器是具有挑战性的，通常分为几个子问题。其中之一是足部步态规划，其中定义了足步的顺序。即使在更简单的环境中，找到一个最小序列，甚至一个可行的序列，也构成了一个复杂的优化问题。在文献中，这个问题通常通过基于搜索的算法（例如A*的变种）来解决。然而，这样的方法要么计算开销很大，要么依赖手工调整多个参数。在这项工作中，首先，我们提出了一种有效的步态规划方法，用于在带有障碍物的局部环境中导航，基于最先进的深度强化学习（DRL）技术，对在线推理要求非常低的计算需求。我们的方法是无启发的，依赖于一组连续的动作来生成可行的足步。相比之下，其他方法需要选择一组相关

    arXiv:2403.12589v1 Announce Type: cross  Abstract: Designing a humanoid locomotion controller is challenging and classically split up in sub-problems. Footstep planning is one of those, where the sequence of footsteps is defined. Even in simpler environments, finding a minimal sequence, or even a feasible sequence, yields a complex optimization problem. In the literature, this problem is usually addressed by search-based algorithms (e.g. variants of A*). However, such approaches are either computationally expensive or rely on hand-crafted tuning of several parameters. In this work, at first, we propose an efficient footstep planning method to navigate in local environments with obstacles, based on state-of-the art Deep Reinforcement Learning (DRL) techniques, with very low computational requirements for on-line inference. Our approach is heuristic-free and relies on a continuous set of actions to generate feasible footsteps. In contrast, other methods necessitate the selection of a rel
    
[^2]: 通过变形测试评估自动驾驶的决策最佳性

    Evaluating Decision Optimality of Autonomous Driving via Metamorphic Testing

    [https://arxiv.org/abs/2402.18393](https://arxiv.org/abs/2402.18393)

    本文着重于评估自动驾驶系统的决策质量，提出了检测非最佳决策场景的方法，通过新颖的变形关系暴露最佳决策违规。

    

    arXiv:2402.18393v1 公告类型：新摘要：自动驾驶系统（ADS）的测试在ADS开发中至关重要，目前主要关注的是安全性。然而，评估非安全关键性能，特别是ADS制定最佳决策并为自动车辆（AV）生成最佳路径的能力同样重要，以确保AV的智能性并降低风险。目前，鲜有工作致力于评估ADS的最佳决策性能，因为缺乏相应的预言和生成有非最佳决策的场景难度较大。本文侧重于评估ADS的决策质量，并提出首个用于检测非最佳决策场景（NoDSs）的方法，即ADS未计算AV的最佳路径的情况。首先，为解决预言问题，我们提出了一种旨在暴露最佳决策违规情况的新颖变形关系（MR）。这个MR确定了性能最佳决策的违规。

    arXiv:2402.18393v1 Announce Type: new  Abstract: Autonomous Driving System (ADS) testing is crucial in ADS development, with the current primary focus being on safety. However, the evaluation of non-safety-critical performance, particularly the ADS's ability to make optimal decisions and produce optimal paths for autonomous vehicles (AVs), is equally vital to ensure the intelligence and reduce risks of AVs. Currently, there is little work dedicated to assessing ADSs' optimal decision-making performance due to the lack of corresponding oracles and the difficulty in generating scenarios with non-optimal decisions. In this paper, we focus on evaluating the decision-making quality of an ADS and propose the first method for detecting non-optimal decision scenarios (NoDSs), where the ADS does not compute optimal paths for AVs. Firstly, to deal with the oracle problem, we propose a novel metamorphic relation (MR) aimed at exposing violations of optimal decisions. The MR identifies the propert
    
[^3]: 多对象图形可用性网络：通过复合对象可用性实现目标导向规划

    Multi-Object Graph Affordance Network: Enabling Goal-Oriented Planning through Compound Object Affordances. (arXiv:2309.10426v1 [cs.RO])

    [http://arxiv.org/abs/2309.10426](http://arxiv.org/abs/2309.10426)

    多对象图形可用性网络（MOGAN）模型化了复合对象的可用性，预测了将新对象放置在复合对象上的效果。在构建具有特定高度或属性的塔等任务中，我们使用基于搜索的规划找到具有适当可用性的对象堆叠操作的序列。我们的系统能够正确建模非常复杂的复合对象的可用性，并在模拟和真实环境中展示了其适用性。

    

    学习对象的可用性是机器人学习领域的有效工具。虽然数据驱动的模型深入探讨了单个或成对对象的可用性，但在调查由复杂形状的任意数量的对象组成的复合对象的可用性方面存在明显差距。在本研究中，我们提出了多对象图形可用性网络（MOGAN），它建模了复合对象的可用性，并预测将新对象放置在现有复合对象上的效果。给定不同的任务，例如构建具有特定高度或属性的塔，我们使用基于搜索的规划找到具有适当可用性的对象堆叠操作的序列。我们展示了我们的系统能够正确建模包括堆叠的球体和杯子、杆和包围杆的环等非常复杂的复合对象的可用性。我们在模拟和真实环境中展示了我们系统的适用性。

    Learning object affordances is an effective tool in the field of robot learning. While the data-driven models delve into the exploration of affordances of single or paired objects, there is a notable gap in the investigation of affordances of compound objects that are composed of an arbitrary number of objects with complex shapes. In this study, we propose Multi-Object Graph Affordance Network (MOGAN) that models compound object affordances and predicts the effect of placing new objects on top of the existing compound. Given different tasks, such as building towers of specific heights or properties, we used a search based planning to find the sequence of stack actions with the objects of suitable affordances. We showed that our system was able to correctly model the affordances of very complex compound objects that include stacked spheres and cups, poles, and rings that enclose the poles. We demonstrated the applicability of our system in both simulated and real-world environments, com
    

