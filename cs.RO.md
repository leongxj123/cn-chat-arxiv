# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [H2O+: An Improved Framework for Hybrid Offline-and-Online RL with Dynamics Gaps.](http://arxiv.org/abs/2309.12716) | H2O+是一种改进的混合离线和在线强化学习框架，通过综合考虑真实和模拟环境的动力学差距，同时利用有限的离线数据和不完美的模拟器进行策略学习，并在广泛的仿真和实际机器人实验中展示了卓越的性能和灵活性。 |

# 详细

[^1]: H2O+: 一种改进的混合离线和在线强化学习框架，用于动力学差距问题

    H2O+: An Improved Framework for Hybrid Offline-and-Online RL with Dynamics Gaps. (arXiv:2309.12716v1 [cs.LG])

    [http://arxiv.org/abs/2309.12716](http://arxiv.org/abs/2309.12716)

    H2O+是一种改进的混合离线和在线强化学习框架，通过综合考虑真实和模拟环境的动力学差距，同时利用有限的离线数据和不完美的模拟器进行策略学习，并在广泛的仿真和实际机器人实验中展示了卓越的性能和灵活性。

    

    在没有高精度模拟环境或大量离线数据的情况下，使用强化学习（RL）解决实际复杂任务可能相当具有挑战性。在非完美模拟环境中训练的在线RL代理可能会受到严重的模拟与现实问题。虽然离线RL方法可以绕过对模拟器的需求，但往往对离线数据集的大小和质量提出了苛刻的要求。最近出现的混合离线和在线RL提供了一个有吸引力的框架，可以同时使用有限的离线数据和不完美的模拟器进行可转移策略学习。本文提出了一种名为H2O+的新算法，该算法在桥接不同的离线和在线学习方法的同时，也考虑了真实和模拟环境之间的动力学差距。通过广泛的仿真和实际机器人实验，我们证明了H2O+在性能和灵活性上优于先进的跨域在线方法

    Solving real-world complex tasks using reinforcement learning (RL) without high-fidelity simulation environments or large amounts of offline data can be quite challenging. Online RL agents trained in imperfect simulation environments can suffer from severe sim-to-real issues. Offline RL approaches although bypass the need for simulators, often pose demanding requirements on the size and quality of the offline datasets. The recently emerged hybrid offline-and-online RL provides an attractive framework that enables joint use of limited offline data and imperfect simulator for transferable policy learning. In this paper, we develop a new algorithm, called H2O+, which offers great flexibility to bridge various choices of offline and online learning methods, while also accounting for dynamics gaps between the real and simulation environment. Through extensive simulation and real-world robotics experiments, we demonstrate superior performance and flexibility over advanced cross-domain online
    

