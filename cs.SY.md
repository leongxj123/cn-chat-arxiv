# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Scaling Learning based Policy Optimization for Temporal Tasks via Dropout](https://arxiv.org/abs/2403.15826) | 本文介绍了一种基于模型的方法用于训练在高度非线性环境中运行的自主智能体的反馈控制器，通过对任务进行形式化表述，实现对特定任务目标的定量满足语义，并利用前馈神经网络学习反馈控制器。 |

# 详细

[^1]: 通过Dropout对时间任务进行比例学习的策略优化扩展

    Scaling Learning based Policy Optimization for Temporal Tasks via Dropout

    [https://arxiv.org/abs/2403.15826](https://arxiv.org/abs/2403.15826)

    本文介绍了一种基于模型的方法用于训练在高度非线性环境中运行的自主智能体的反馈控制器，通过对任务进行形式化表述，实现对特定任务目标的定量满足语义，并利用前馈神经网络学习反馈控制器。

    

    本文介绍了一种基于模型的方法，用于训练在高度非线性环境中运行的自主智能体的反馈控制器。我们希望经过训练的策略能够确保该智能体满足特定的任务目标，这些目标以离散时间信号时间逻辑（DT-STL）表示。通过将任务重新表述为形式化框架（如DT-STL），一个优势是允许定量满足语义。换句话说，给定一个轨迹和一个DT-STL公式，我们可以计算鲁棒性，这可以解释为轨迹与满足该公式的轨迹集之间的近似有符号距离。我们利用反馈控制器，并假设使用前馈神经网络来学习这些反馈控制器。我们展示了这个学习问题与训练递归神经网络（RNNs）类似的地方，其中递归单元的数量与智能体的时间视野成比例。

    arXiv:2403.15826v1 Announce Type: cross  Abstract: This paper introduces a model-based approach for training feedback controllers for an autonomous agent operating in a highly nonlinear environment. We desire the trained policy to ensure that the agent satisfies specific task objectives, expressed in discrete-time Signal Temporal Logic (DT-STL). One advantage for reformulation of a task via formal frameworks, like DT-STL, is that it permits quantitative satisfaction semantics. In other words, given a trajectory and a DT-STL formula, we can compute the robustness, which can be interpreted as an approximate signed distance between the trajectory and the set of trajectories satisfying the formula. We utilize feedback controllers, and we assume a feed forward neural network for learning these feedback controllers. We show how this learning problem is similar to training recurrent neural networks (RNNs), where the number of recurrent units is proportional to the temporal horizon of the agen
    

