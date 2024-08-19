# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A PPO-based DRL Auto-Tuning Nonlinear PID Drone Controller for Robust Autonomous Flights](https://arxiv.org/abs/2404.00204) | 该项目将非线性深度强化学习（DRL）代理引入无人机控制中，取代传统线性PID控制器，实现了无缝过渡、提高响应速度和稳定性，同时结合PPO策略训练DRL代理，并利用高精度跟踪系统提高自主飞行精度。 |
| [^2] | [Isometric Motion Manifold Primitives.](http://arxiv.org/abs/2310.17072) | Isometric Motion Manifold Primitives (IMMP) is proposed to address the degradation of Motion Manifold Primitive (MMP) performance due to geometric distortion in the latent space. IMMP preserves the geometry of the manifold in the latent coordinate space using a Riemannian metric, and experimental results show that IMMP significantly outperforms existing MMP methods. |

# 详细

[^1]: 基于PPO的DRL自调PID非线性无人机控制器用于稳健自主飞行

    A PPO-based DRL Auto-Tuning Nonlinear PID Drone Controller for Robust Autonomous Flights

    [https://arxiv.org/abs/2404.00204](https://arxiv.org/abs/2404.00204)

    该项目将非线性深度强化学习（DRL）代理引入无人机控制中，取代传统线性PID控制器，实现了无缝过渡、提高响应速度和稳定性，同时结合PPO策略训练DRL代理，并利用高精度跟踪系统提高自主飞行精度。

    

    该项目旨在通过将非线性深度强化学习（DRL）代理作为传统线性比例积分微分（PID）控制器的替代品，从而彻底改变无人机飞行控制。主要目标是在手动和自主模式之间实现无缝过渡，提高响应速度和稳定性。我们在Gazebo模拟器中利用近端策略优化（PPO）强化学习策略来训练DRL代理。添加20000美元的室内Vicon跟踪系统提供<1mm的定位精度，显着提高了自主飞行精度。为了在最短的无碰撞轨迹中导航无人机，我们还建立了一个三维A*路径规划器并成功地将其实施到实际飞行中。

    arXiv:2404.00204v1 Announce Type: cross  Abstract: This project aims to revolutionize drone flight control by implementing a nonlinear Deep Reinforcement Learning (DRL) agent as a replacement for traditional linear Proportional Integral Derivative (PID) controllers. The primary objective is to seamlessly transition drones between manual and autonomous modes, enhancing responsiveness and stability. We utilize the Proximal Policy Optimization (PPO) reinforcement learning strategy within the Gazebo simulator to train the DRL agent. Adding a $20,000 indoor Vicon tracking system offers <1mm positioning accuracy, which significantly improves autonomous flight precision. To navigate the drone in the shortest collision-free trajectory, we also build a 3 dimensional A* path planner and implement it into the real flight successfully.
    
[^2]: 等距运动流形基元

    Isometric Motion Manifold Primitives. (arXiv:2310.17072v1 [cs.AI])

    [http://arxiv.org/abs/2310.17072](http://arxiv.org/abs/2310.17072)

    Isometric Motion Manifold Primitives (IMMP) is proposed to address the degradation of Motion Manifold Primitive (MMP) performance due to geometric distortion in the latent space. IMMP preserves the geometry of the manifold in the latent coordinate space using a Riemannian metric, and experimental results show that IMMP significantly outperforms existing MMP methods.

    

    运动流形基元（MMP）为给定任务生成一系列连续轨迹流形，每一个轨迹流形都能成功完成任务。它由对流形进行参数化的解码函数以及潜在坐标空间中的概率密度组成。本文首先展示了由于潜在空间中的几何扭曲，MMP的性能可能会显著降低--通过变形，我们指的是相似的运动在潜在空间中无法相邻。然后，我们提出了等距运动流形基元（IMMP），其潜在坐标空间保持了流形的几何结构。为此，我们建立和使用了一个Riemannian度量，用于运动空间（即，参数化曲线空间），我们称之为CurveGeom Riemannian度量。对于平面障碍避让运动和推动操纵任务的实验表明，IMMP明显优于现有的MMP方法。代码可在https://github.com/Gabe-YHLee/IMMP找到。

    The Motion Manifold Primitive (MMP) produces, for a given task, a continuous manifold of trajectories each of which can successfully complete the task. It consists of the decoder function that parametrizes the manifold and the probability density in the latent coordinate space. In this paper, we first show that the MMP performance can significantly degrade due to the geometric distortion in the latent space -- by distortion, we mean that similar motions are not located nearby in the latent space. We then propose {\it Isometric Motion Manifold Primitives (IMMP)} whose latent coordinate space preserves the geometry of the manifold. For this purpose, we formulate and use a Riemannian metric for the motion space (i.e., parametric curve space), which we call a {\it CurveGeom Riemannian metric}. Experiments with planar obstacle-avoiding motions and pushing manipulation tasks show that IMMP significantly outperforms existing MMP methods. Code is available at https://github.com/Gabe-YHLee/IMMP
    

