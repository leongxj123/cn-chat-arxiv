# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Constraint-Generation Policy Optimization (CGPO): Nonlinear Programming for Policy Optimization in Mixed Discrete-Continuous MDPs.](http://arxiv.org/abs/2401.12243) | Constraint-Generation Policy Optimization (CGPO)是一种针对混合离散连续MDPs的策略优化方法，能够提供有界的策略误差保证，推导出最优策略，并生成最坏情况的状态轨迹来诊断策略缺陷。 |
| [^2] | [MutateNN: Mutation Testing of Image Recognition Models Deployed on Hardware Accelerators.](http://arxiv.org/abs/2306.01697) | MutateNN是一种用于探索硬件加速器上深度学习图像识别模型鲁棒性的工具，提供突变测试和分析能力，且有效性已在多种预训练深度神经网络模型中得到验证。 |

# 详细

[^1]: Constraint-Generation Policy Optimization (CGPO): 针对混合离散连续MDPs中的策略优化的非线性规划

    Constraint-Generation Policy Optimization (CGPO): Nonlinear Programming for Policy Optimization in Mixed Discrete-Continuous MDPs. (arXiv:2401.12243v1 [math.OC])

    [http://arxiv.org/abs/2401.12243](http://arxiv.org/abs/2401.12243)

    Constraint-Generation Policy Optimization (CGPO)是一种针对混合离散连续MDPs的策略优化方法，能够提供有界的策略误差保证，推导出最优策略，并生成最坏情况的状态轨迹来诊断策略缺陷。

    

    我们提出了Constraint-Generation Policy Optimization (CGPO)方法，用于在混合离散连续Markov Decision Processes (DC-MDPs)中优化策略参数。CGPO不仅能够提供有界的策略误差保证，覆盖具有表达能力的非线性动力学的无数初始状态范围的DC-MDPs，而且在结束时可以明确地推导出最优策略。此外，CGPO还能够生成最坏情况的状态轨迹来诊断策略缺陷，并提供最优行动的反事实解释。为了实现这些结果，CGPO提出了一个双层的混合整数非线性优化框架，用于在定义的表达能力类别（即分段(非)线性）内优化策略，并将其转化为一个最优的约束生成方法，通过对抗性生成最坏情况的状态轨迹。此外，借助现代非线性优化器，CGPO可以获得解决方案。

    We propose Constraint-Generation Policy Optimization (CGPO) for optimizing policy parameters within compact and interpretable policy classes for mixed discrete-continuous Markov Decision Processes (DC-MDPs). CGPO is not only able to provide bounded policy error guarantees over an infinite range of initial states for many DC-MDPs with expressive nonlinear dynamics, but it can also provably derive optimal policies in cases where it terminates with zero error. Furthermore, CGPO can generate worst-case state trajectories to diagnose policy deficiencies and provide counterfactual explanations of optimal actions. To achieve such results, CGPO proposes a bi-level mixed-integer nonlinear optimization framework for optimizing policies within defined expressivity classes (i.e. piecewise (non)-linear) and reduces it to an optimal constraint generation methodology that adversarially generates worst-case state trajectories. Furthermore, leveraging modern nonlinear optimizers, CGPO can obtain soluti
    
[^2]: MutateNN：用于硬件加速器上图像识别模型的突变测试

    MutateNN: Mutation Testing of Image Recognition Models Deployed on Hardware Accelerators. (arXiv:2306.01697v1 [cs.LG])

    [http://arxiv.org/abs/2306.01697](http://arxiv.org/abs/2306.01697)

    MutateNN是一种用于探索硬件加速器上深度学习图像识别模型鲁棒性的工具，提供突变测试和分析能力，且有效性已在多种预训练深度神经网络模型中得到验证。

    

    随着人工智能的研究进展，解决现实世界问题并推动技术发展的新机遇应运而生。图像识别模型特别是被分配了感知任务，以解决复杂的现实世界挑战并导致新的解决方案。此外，这类模型的计算复杂度和资源需求也有所增加。为了解决这个问题，模型优化和硬件加速已成为关键技术，但有效整合这些概念是一个具有挑战性和容易出错的过程。为了让开发人员和研究人员能够探索在不同硬件加速设备上部署的深度学习图像识别模型的鲁棒性，我们提出了MutateNN，这是一个为此目的提供突变测试和分析能力的工具。为了展示其功能，我们对7个广为人知的预训练深度神经网络模型进行了21种变异。我们在4种不同类型的硬件加速器上部署了我们的变异体，分析了它们的行为，并评估了MutateNN在检测出不正确或不精确的模型行为方面的有效性。

    With the research advancement of Artificial Intelligence in the last years, there are new opportunities to mitigate real-world problems and advance technologically. Image recognition models in particular, are assigned with perception tasks to mitigate complex real-world challenges and lead to new solutions. Furthermore, the computational complexity and demand for resources of such models has also increased. To mitigate this, model optimization and hardware acceleration has come into play, but effectively integrating such concepts is a challenging and error-prone process.  In order to allow developers and researchers to explore the robustness of deep learning image recognition models deployed on different hardware acceleration devices, we propose MutateNN, a tool that provides mutation testing and analysis capabilities for that purpose. To showcase its capabilities, we utilized 21 mutations for 7 widely-known pre-trained deep neural network models. We deployed our mutants on 4 different
    

