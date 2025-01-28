# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Push it to the Demonstrated Limit: Multimodal Visuotactile Imitation Learning with Force Matching.](http://arxiv.org/abs/2311.01248) | 本研究利用视觉触觉传感器和模仿学习相结合，通过配对优化触觉力量曲线和简化传感器应用，对接触丰富的操作任务进行了研究。 |
| [^2] | [GenDOM: Generalizable One-shot Deformable Object Manipulation with Parameter-Aware Policy.](http://arxiv.org/abs/2309.09051) | GenDOM是一个框架，通过条件化操作策略和多样化模拟训练，使操作策略能够仅通过一个真实世界演示来处理不同的可变形物体，并通过最小化真实世界演示和模拟之间的点云格密度差异来估计新物体的可变形物体参数。 |
| [^3] | [Generalizable One-shot Rope Manipulation with Parameter-Aware Policy.](http://arxiv.org/abs/2306.09872) | GenORM通过增加可变形绳索参数和使用各种可变形绳索的模拟训练操作策略，实现利用一次真实演示处理不同可形变绳索，从而节省演示时间和提高适用性。 |
| [^4] | [Collective Intelligence for 2D Push Manipulation with Mobile Robots.](http://arxiv.org/abs/2211.15136) | 本研究利用基于软体物理模拟器的规划器和基于注意力的神经网络，实现了移动机器人2D协作推动操作中的集体智能，比传统方法具有更好的性能并具备环境自适应能力。 |

# 详细

[^1]: 将其推向展示极限：多模态视觉触觉模仿学习与力匹配

    Push it to the Demonstrated Limit: Multimodal Visuotactile Imitation Learning with Force Matching. (arXiv:2311.01248v1 [cs.RO])

    [http://arxiv.org/abs/2311.01248](http://arxiv.org/abs/2311.01248)

    本研究利用视觉触觉传感器和模仿学习相结合，通过配对优化触觉力量曲线和简化传感器应用，对接触丰富的操作任务进行了研究。

    

    光学触觉传感器已经成为机器人操作过程中获取密集接触信息的有效手段。最近引入的“透视你的皮肤”（STS）型传感器具有视觉和触觉模式，通过利用半透明表面和可控照明实现。本文研究了视觉触觉传感与模仿学习在富有接触的操作任务中的好处。首先，我们使用触觉力测量和一种新的算法，在运动示范中产生更好匹配人体示范者的力曲线。其次，我们添加了视觉/触觉STS模式切换作为控制策略输出，简化传感器的应用。最后，我们研究了多种观察配置，比较和对比了视觉/触觉数据（包括模式切换和不切换）与手腕挂载的眼在手摄像机的视觉数据的价值。我们在一个广泛的实验系列上进行实验。

    Optical tactile sensors have emerged as an effective means to acquire dense contact information during robotic manipulation. A recently-introduced `see-through-your-skin' (STS) variant of this type of sensor has both visual and tactile modes, enabled by leveraging a semi-transparent surface and controllable lighting. In this work, we investigate the benefits of pairing visuotactile sensing with imitation learning for contact-rich manipulation tasks. First, we use tactile force measurements and a novel algorithm during kinesthetic teaching to yield a force profile that better matches that of the human demonstrator. Second, we add visual/tactile STS mode switching as a control policy output, simplifying the application of the sensor. Finally, we study multiple observation configurations to compare and contrast the value of visual/tactile data (both with and without mode switching) with visual data from a wrist-mounted eye-in-hand camera. We perform an extensive series of experiments on a
    
[^2]: GenDOM：具有参数感知的泛化性一次变形物体操作

    GenDOM: Generalizable One-shot Deformable Object Manipulation with Parameter-Aware Policy. (arXiv:2309.09051v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2309.09051](http://arxiv.org/abs/2309.09051)

    GenDOM是一个框架，通过条件化操作策略和多样化模拟训练，使操作策略能够仅通过一个真实世界演示来处理不同的可变形物体，并通过最小化真实世界演示和模拟之间的点云格密度差异来估计新物体的可变形物体参数。

    

    由于在运动中存在固有的变形不确定性，以往的可变形物体操作方法（如绳子和布料）通常需要数百个真实世界演示来训练每个物体的操作策略，这限制了它们在不断变化的世界中的应用。为了解决这个问题，我们引入了GenDOM，这是一个框架，允许操作策略只需一个真实世界演示来处理不同的可变形物体。为了实现这一点，我们通过将操作策略与可变形物体参数联系起来，并使用多样化的模拟可变形物体对其进行训练，使策略能够根据不同的物体参数调整动作。在推理时，给定一个新的物体，GenDOM可以通过最小化真实世界演示和模拟之间点云的格密度差异来估计可变形物体参数，而只需一个真实世界演示。

    Due to the inherent uncertainty in their deformability during motion, previous methods in deformable object manipulation, such as rope and cloth, often required hundreds of real-world demonstrations to train a manipulation policy for each object, which hinders their applications in our ever-changing world. To address this issue, we introduce GenDOM, a framework that allows the manipulation policy to handle different deformable objects with only a single real-world demonstration. To achieve this, we augment the policy by conditioning it on deformable object parameters and training it with a diverse range of simulated deformable objects so that the policy can adjust actions based on different object parameters. At the time of inference, given a new object, GenDOM can estimate the deformable object parameters with only a single real-world demonstration by minimizing the disparity between the grid density of point clouds of real-world demonstrations and simulations in a differentiable phys
    
[^3]: 可泛化的一次性绳索操作策略及其参数感知性。

    Generalizable One-shot Rope Manipulation with Parameter-Aware Policy. (arXiv:2306.09872v1 [cs.LG])

    [http://arxiv.org/abs/2306.09872](http://arxiv.org/abs/2306.09872)

    GenORM通过增加可变形绳索参数和使用各种可变形绳索的模拟训练操作策略，实现利用一次真实演示处理不同可形变绳索，从而节省演示时间和提高适用性。

    

    以绳索在运动过程中的固有不确定性为因素，以往绳索操作方法往往需要数百次真实演示来为每个绳索训练操作策略，即使是简单的“到达目标”任务，这限制了它们在我们不断变化的世界中的应用。为了解决这个问题，我们介绍了GenORM，一个框架，它可以让操作策略通过一次真实演示就可以处理不同可形变的绳索。我们通过在策略上增加可变形绳索参数并使用各种模拟可变形绳索来训练它，使策略能够根据不同的绳索参数调整行动。在推断时，GenORM通过最小化真实演示和模拟点云的网格密度差异来估计可变形绳索参数。通过可微分物理模拟器的帮助，我们仅需要一次演示数据就可以处理不同的绳索。

    Due to the inherent uncertainty in their deformability during motion, previous methods in rope manipulation often require hundreds of real-world demonstrations to train a manipulation policy for each rope, even for simple tasks such as rope goal reaching, which hinder their applications in our ever-changing world. To address this issue, we introduce GenORM, a framework that allows the manipulation policy to handle different deformable ropes with a single real-world demonstration. To achieve this, we augment the policy by conditioning it on deformable rope parameters and training it with a diverse range of simulated deformable ropes so that the policy can adjust actions based on different rope parameters. At the time of inference, given a new rope, GenORM estimates the deformable rope parameters by minimizing the disparity between the grid density of point clouds of real-world demonstrations and simulations. With the help of a differentiable physics simulator, we require only a single r
    
[^4]: 移动机器人的2D推动操作中的集体智能

    Collective Intelligence for 2D Push Manipulation with Mobile Robots. (arXiv:2211.15136v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2211.15136](http://arxiv.org/abs/2211.15136)

    本研究利用基于软体物理模拟器的规划器和基于注意力的神经网络，实现了移动机器人2D协作推动操作中的集体智能，比传统方法具有更好的性能并具备环境自适应能力。

    

    自然系统通常表现出能够自我组织和适应变化的集体智能，但大多数人工系统缺乏这种等效性。本文探讨使用移动机器人进行2D协作推动操作的集体智能系统的可能性。我们展示了将从软体物理模拟派生的规划器提炼为基于注意力的神经网络后，我们的多机器人推动操作系统相对于基线系统具有更好的性能，并可适应外部扰动和环境变化完成任务。

    While natural systems often present collective intelligence that allows them to self-organize and adapt to changes, the equivalent is missing in most artificial systems. We explore the possibility of such a system in the context of cooperative 2D push manipulations using mobile robots. Although conventional works demonstrate potential solutions for the problem in restricted settings, they have computational and learning difficulties. More importantly, these systems do not possess the ability to adapt when facing environmental changes. In this work, we show that by distilling a planner derived from a differentiable soft-body physics simulator into an attention-based neural network, our multi-robot push manipulation system achieves better performance than baselines. In addition, our system also generalizes to configurations not seen during training and is able to adapt toward task completions when external turbulence and environmental changes are applied. Supplementary videos can be foun
    

