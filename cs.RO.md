# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Safe Guaranteed Exploration for Non-linear Systems](https://arxiv.org/abs/2402.06562) | 本文提出了一个新颖的安全保证探索框架，使用最优控制实现了对非线性系统的有限时间样本复杂度界限的保证探索，同时具有证明的安全性和广泛适用性。 |

# 详细

[^1]: 非线性系统的安全保证探索

    Safe Guaranteed Exploration for Non-linear Systems

    [https://arxiv.org/abs/2402.06562](https://arxiv.org/abs/2402.06562)

    本文提出了一个新颖的安全保证探索框架，使用最优控制实现了对非线性系统的有限时间样本复杂度界限的保证探索，同时具有证明的安全性和广泛适用性。

    

    在具有先验未知约束的环境中进行安全探索是限制机器人自主性的基本挑战。虽然安全性至关重要，但对足够探索的保证对于确保自主任务完成也至关重要。为了解决这些挑战，我们提出了一个新颖的安全保证探索框架，利用最优控制实现前所未有的结果：具有有限时间样本复杂度界限的非线性系统的保证探索，同时在任意高概率下被证明是安全的。该框架具有广泛的适用性，可适用于具有复杂非线性动力学和未知领域的许多实际场景。基于这个框架，我们提出了一种高效的算法，SageMPC，采用模型预测控制进行安全保证探索。SageMPC通过整合三种技术来提高效率：i) 利用Lipschitz边界，ii) 目标导向探索，和iii) 逐步调整风格的重新规划，同时保持高效性。

    Safely exploring environments with a-priori unknown constraints is a fundamental challenge that restricts the autonomy of robots. While safety is paramount, guarantees on sufficient exploration are also crucial for ensuring autonomous task completion. To address these challenges, we propose a novel safe guaranteed exploration framework using optimal control, which achieves first-of-its-kind results: guaranteed exploration for non-linear systems with finite time sample complexity bounds, while being provably safe with arbitrarily high probability. The framework is general and applicable to many real-world scenarios with complex non-linear dynamics and unknown domains. Based on this framework we propose an efficient algorithm, SageMPC, SAfe Guaranteed Exploration using Model Predictive Control. SageMPC improves efficiency by incorporating three techniques: i) exploiting a Lipschitz bound, ii) goal-directed exploration, and iii) receding horizon style re-planning, all while maintaining th
    

