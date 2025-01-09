# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-Fidelity Bayesian Optimization With Across-Task Transferable Max-Value Entropy Search](https://arxiv.org/abs/2403.09570) | 本文引入了一种新颖的信息理论获取函数，用于平衡在连续的优化任务中获得最优值或解信息的需求。 |

# 详细

[^1]: 基于多保真度的贝叶斯优化方法及跨任务可转移的最大值熵搜索

    Multi-Fidelity Bayesian Optimization With Across-Task Transferable Max-Value Entropy Search

    [https://arxiv.org/abs/2403.09570](https://arxiv.org/abs/2403.09570)

    本文引入了一种新颖的信息理论获取函数，用于平衡在连续的优化任务中获得最优值或解信息的需求。

    

    在许多应用中，设计者面临一系列优化任务，任务的目标是昂贵评估的黑盒函数形式。本文介绍了一种新的信息理论获取函数，用于平衡需要获取不同任务的最优值或解的信息和通过参数的转移传递。

    arXiv:2403.09570v1 Announce Type: new  Abstract: In many applications, ranging from logistics to engineering, a designer is faced with a sequence of optimization tasks for which the objectives are in the form of black-box functions that are costly to evaluate. For example, the designer may need to tune the hyperparameters of neural network models for different learning tasks over time. Rather than evaluating the objective function for each candidate solution, the designer may have access to approximations of the objective functions, for which higher-fidelity evaluations entail a larger cost. Existing multi-fidelity black-box optimization strategies select candidate solutions and fidelity levels with the goal of maximizing the information accrued about the optimal value or solution for the current task. Assuming that successive optimization tasks are related, this paper introduces a novel information-theoretic acquisition function that balances the need to acquire information about the 
    

