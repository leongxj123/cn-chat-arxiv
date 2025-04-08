# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SoftMAC: Differentiable Soft Body Simulation with Forecast-based Contact Model and Two-way Coupling with Articulated Rigid Bodies and Clothes](https://arxiv.org/abs/2312.03297) | SoftMAC提出了一个不同于以往的可微仿真框架，能够将软体、关节刚体和衣物耦合在一起，并采用基于预测的接触模型和穿透追踪算法，有效地减少了穿透现象。 |

# 详细

[^1]: SoftMAC：基于预测接触模型和与关节刚体和衣物双向耦合的可微软体仿真

    SoftMAC: Differentiable Soft Body Simulation with Forecast-based Contact Model and Two-way Coupling with Articulated Rigid Bodies and Clothes

    [https://arxiv.org/abs/2312.03297](https://arxiv.org/abs/2312.03297)

    SoftMAC提出了一个不同于以往的可微仿真框架，能够将软体、关节刚体和衣物耦合在一起，并采用基于预测的接触模型和穿透追踪算法，有效地减少了穿透现象。

    

    可微物理仿真通过基于梯度的优化，显著提高了解决机器人相关问题的效率。为在各种机器人操纵场景中应用可微仿真，一个关键挑战是将各种材料集成到统一框架中。我们提出了SoftMAC，一个可微仿真框架，将软体与关节刚体和衣物耦合在一起。SoftMAC使用基于连续力学的材料点法来模拟软体。我们提出了一种新颖的基于预测的MPM接触模型，有效减少了穿透，而不会引入其他异常现象，如不自然的反弹。为了将MPM粒子与可变形和非体积衣物网格耦合，我们还提出了一种穿透追踪算法，重建局部区域的有符号距离场。

    arXiv:2312.03297v2 Announce Type: replace-cross  Abstract: Differentiable physics simulation provides an avenue to tackle previously intractable challenges through gradient-based optimization, thereby greatly improving the efficiency of solving robotics-related problems. To apply differentiable simulation in diverse robotic manipulation scenarios, a key challenge is to integrate various materials in a unified framework. We present SoftMAC, a differentiable simulation framework that couples soft bodies with articulated rigid bodies and clothes. SoftMAC simulates soft bodies with the continuum-mechanics-based Material Point Method (MPM). We provide a novel forecast-based contact model for MPM, which effectively reduces penetration without introducing other artifacts like unnatural rebound. To couple MPM particles with deformable and non-volumetric clothes meshes, we also propose a penetration tracing algorithm that reconstructs the signed distance field in local area. Diverging from prev
    

