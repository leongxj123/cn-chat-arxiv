# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ODIN: A Single Model for 2D and 3D Perception.](http://arxiv.org/abs/2401.02416) | ODIN是一个模型，可以同时对2D RGB图像和3D点云进行分割和标记，使用变压器架构进行2D和3D视图间的信息融合。 |
| [^2] | [Push it to the Demonstrated Limit: Multimodal Visuotactile Imitation Learning with Force Matching.](http://arxiv.org/abs/2311.01248) | 本研究利用视觉触觉传感器和模仿学习相结合，通过配对优化触觉力量曲线和简化传感器应用，对接触丰富的操作任务进行了研究。 |
| [^3] | [On Convex Data-Driven Inverse Optimal Control for Nonlinear, Non-stationary and Stochastic Systems.](http://arxiv.org/abs/2306.13928) | 本文提出了一个凸数据驱动逆最优控制方案，能够有效解决非线性、非平稳和随机系统下的成本估计问题。 |

# 详细

[^1]: ODIN: 一个用于2D和3D感知的单一模型

    ODIN: A Single Model for 2D and 3D Perception. (arXiv:2401.02416v1 [cs.CV])

    [http://arxiv.org/abs/2401.02416](http://arxiv.org/abs/2401.02416)

    ODIN是一个模型，可以同时对2D RGB图像和3D点云进行分割和标记，使用变压器架构进行2D和3D视图间的信息融合。

    

    目前的先进模型在像ScanNet这样的当代3D感知基准上使用并标记依赖于数据集提供的3D点云，该点云是通过对感知到的多视角RGB-D图像进行后处理获得的。它们通常在领域内进行训练，放弃了大规模的2D预训练，并且胜过将姿态RGB-D多视角图像进行特征化的替代方案。消耗姿态图像和后处理的3D点云之间的性能差距，加剧了2D和3D感知需要不同模型架构的观点。在本文中，我们挑战这个观点，并提出ODIN（Omni-Dimensional INstance segmentation），一种能够使用变压器架构对2D RGB图像和3D点云进行分割和标记的模型，该模型通过交替的2D视图内和3D视图间信息融合来区分2D和3D特征操作，利用涉及的令牌的位置编码来捕捉2D补丁令牌和3D坐标的像素坐标。

    State-of-the-art models on contemporary 3D perception benchmarks like ScanNet consume and label dataset-provided 3D point clouds, obtained through post processing of sensed multiview RGB-D images. They are typically trained in-domain, forego large-scale 2D pre-training and outperform alternatives that featurize the posed RGB-D multiview images instead. The gap in performance between methods that consume posed images versus post-processed 3D point clouds has fueled the belief that 2D and 3D perception require distinct model architectures. In this paper, we challenge this view and propose ODIN (Omni-Dimensional INstance segmentation), a model that can segment and label both 2D RGB images and 3D point clouds, using a transformer architecture that alternates between 2D within-view and 3D cross-view information fusion. Our model differentiates 2D and 3D feature operations through the positional encodings of the tokens involved, which capture pixel coordinates for 2D patch tokens and 3D coor
    
[^2]: 将其推向展示极限：多模态视觉触觉模仿学习与力匹配

    Push it to the Demonstrated Limit: Multimodal Visuotactile Imitation Learning with Force Matching. (arXiv:2311.01248v1 [cs.RO])

    [http://arxiv.org/abs/2311.01248](http://arxiv.org/abs/2311.01248)

    本研究利用视觉触觉传感器和模仿学习相结合，通过配对优化触觉力量曲线和简化传感器应用，对接触丰富的操作任务进行了研究。

    

    光学触觉传感器已经成为机器人操作过程中获取密集接触信息的有效手段。最近引入的“透视你的皮肤”（STS）型传感器具有视觉和触觉模式，通过利用半透明表面和可控照明实现。本文研究了视觉触觉传感与模仿学习在富有接触的操作任务中的好处。首先，我们使用触觉力测量和一种新的算法，在运动示范中产生更好匹配人体示范者的力曲线。其次，我们添加了视觉/触觉STS模式切换作为控制策略输出，简化传感器的应用。最后，我们研究了多种观察配置，比较和对比了视觉/触觉数据（包括模式切换和不切换）与手腕挂载的眼在手摄像机的视觉数据的价值。我们在一个广泛的实验系列上进行实验。

    Optical tactile sensors have emerged as an effective means to acquire dense contact information during robotic manipulation. A recently-introduced `see-through-your-skin' (STS) variant of this type of sensor has both visual and tactile modes, enabled by leveraging a semi-transparent surface and controllable lighting. In this work, we investigate the benefits of pairing visuotactile sensing with imitation learning for contact-rich manipulation tasks. First, we use tactile force measurements and a novel algorithm during kinesthetic teaching to yield a force profile that better matches that of the human demonstrator. Second, we add visual/tactile STS mode switching as a control policy output, simplifying the application of the sensor. Finally, we study multiple observation configurations to compare and contrast the value of visual/tactile data (both with and without mode switching) with visual data from a wrist-mounted eye-in-hand camera. We perform an extensive series of experiments on a
    
[^3]: 针对非线性、非平稳和随机系统的凸数据驱动逆最优控制研究

    On Convex Data-Driven Inverse Optimal Control for Nonlinear, Non-stationary and Stochastic Systems. (arXiv:2306.13928v1 [math.OC])

    [http://arxiv.org/abs/2306.13928](http://arxiv.org/abs/2306.13928)

    本文提出了一个凸数据驱动逆最优控制方案，能够有效解决非线性、非平稳和随机系统下的成本估计问题。

    

    本文主要论述了一个有限时域的逆控制问题，其目的是从观测值中推断出驱动智能体行动的成本，即使这个成本是非凸和非平稳的，同时受到非线性、非平稳和随机因素的影响。在这种情况下，我们提出了一个解决方案，通过解决一个优化问题来实现成本估计，即使代理成本不是凸的，本文也能够生成凸问题。为了得出这个结果，我们还研究了一个以随机策略为决策变量的有限时域前向控制问题，并给出了最优解的显式表达式。此外，我们将我们的发现转化为算法流程，并通过虚拟实验和真实硬件实验验证了我们的方法的有效性。所有的实验结果都证实了我们方法的有效性。

    This paper is concerned with a finite-horizon inverse control problem, which has the goal of inferring, from observations, the possibly non-convex and non-stationary cost driving the actions of an agent. In this context, we present a result that enables cost estimation by solving an optimization problem that is convex even when the agent cost is not and when the underlying dynamics is nonlinear, non-stationary and stochastic. To obtain this result, we also study a finite-horizon forward control problem that has randomized policies as decision variables. For this problem, we give an explicit expression for the optimal solution. Moreover, we turn our findings into algorithmic procedures and we show the effectiveness of our approach via both in-silico and experimental validations with real hardware. All the experiments confirm the effectiveness of our approach.
    

