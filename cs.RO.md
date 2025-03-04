# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Calib3D: Calibrating Model Preferences for Reliable 3D Scene Understanding](https://arxiv.org/abs/2403.17010) | Calib3D是一个从不确定性估计的角度出发，对多个3D场景理解模型进行了全面评估，发现现有模型虽然准确但不可靠，从而阐明了安全关键的背景下的重要性。 |
| [^2] | [A Policy Adaptation Method for Implicit Multitask Reinforcement Learning Problems.](http://arxiv.org/abs/2308.16471) | 本研究提出了一种适用于动态运动生成任务的多任务强化学习算法，可用于适应单个运动类别中的隐式变化，并在头球任务中取得良好的适应效果。 |

# 详细

[^1]: Calib3D：校准模型偏好以实现可靠的3D场景理解

    Calib3D: Calibrating Model Preferences for Reliable 3D Scene Understanding

    [https://arxiv.org/abs/2403.17010](https://arxiv.org/abs/2403.17010)

    Calib3D是一个从不确定性估计的角度出发，对多个3D场景理解模型进行了全面评估，发现现有模型虽然准确但不可靠，从而阐明了安全关键的背景下的重要性。

    

    安全关键的3D场景理解任务需要的不仅仅是准确的预测，还需要来自3D感知模型的自信预测。本研究推出了Calib3D，这是一项开创性的工作，旨在从不确定性估计的角度基准和审查3D场景理解模型的可靠性。我们全面评估了28个最先进的模型在10个不同的3D数据集上，揭示了能够处理3D场景理解中的误差不确定性和认知不确定性的有见地的现象。我们发现，尽管现有模型取得了令人印象深刻的准确度水平，但它们经常无法提供可靠的不确定性估计 -- 这个关键的缺陷严重损害了它们在安全敏感环境中的适用性。通过对关键因素（如网络容量、LiDAR表示、光栅分辨率和3D数据增强技术）进行了广泛分析，我们直接将这些方面与模型校准相关联。

    arXiv:2403.17010v1 Announce Type: cross  Abstract: Safety-critical 3D scene understanding tasks necessitate not only accurate but also confident predictions from 3D perception models. This study introduces Calib3D, a pioneering effort to benchmark and scrutinize the reliability of 3D scene understanding models from an uncertainty estimation viewpoint. We comprehensively evaluate 28 state-of-the-art models across 10 diverse 3D datasets, uncovering insightful phenomena that cope with both the aleatoric and epistemic uncertainties in 3D scene understanding. We discover that despite achieving impressive levels of accuracy, existing models frequently fail to provide reliable uncertainty estimates -- a pitfall that critically undermines their applicability in safety-sensitive contexts. Through extensive analysis of key factors such as network capacity, LiDAR representations, rasterization resolutions, and 3D data augmentation techniques, we correlate these aspects directly with the model cal
    
[^2]: 一种适用于隐式多任务强化学习问题的策略适应方法

    A Policy Adaptation Method for Implicit Multitask Reinforcement Learning Problems. (arXiv:2308.16471v1 [cs.RO])

    [http://arxiv.org/abs/2308.16471](http://arxiv.org/abs/2308.16471)

    本研究提出了一种适用于动态运动生成任务的多任务强化学习算法，可用于适应单个运动类别中的隐式变化，并在头球任务中取得良好的适应效果。

    

    在动态运动生成任务中，包括接触和碰撞，策略参数的小改变可能导致极其不同的回报。例如，在足球中，通过稍微改变踢球位置或施加球的力或者球的摩擦力发生变化，球可以以完全不同的方向飞行。然而，很难想象在不同的方向上头球需要完全不同的技能。在本研究中，我们提出了一种多任务强化学习算法，用于在单个运动类别中适应目标或环境的隐式变化，包括不同的奖励函数或环境的物理参数。我们利用单脚机器人模型对所提出的方法进行了评估，在头球任务中取得了良好的适应效果。结果表明，所提出的方法可以适应目标位置的隐式变化或球的恢复系数的变化，而标准的领域随机化方法则不能。

    In dynamic motion generation tasks, including contact and collisions, small changes in policy parameters can lead to extremely different returns. For example, in soccer, the ball can fly in completely different directions with a similar heading motion by slightly changing the hitting position or the force applied to the ball or when the friction of the ball varies. However, it is difficult to imagine that completely different skills are needed for heading a ball in different directions. In this study, we proposed a multitask reinforcement learning algorithm for adapting a policy to implicit changes in goals or environments in a single motion category with different reward functions or physical parameters of the environment. We evaluated the proposed method on the ball heading task using a monopod robot model. The results showed that the proposed method can adapt to implicit changes in the goal positions or the coefficients of restitution of the ball, whereas the standard domain randomi
    

