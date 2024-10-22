# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MindArm: Mechanized Intelligent Non-Invasive Neuro-Driven Prosthetic Arm System](https://arxiv.org/abs/2403.19992) | 提出了一种低成本技术解决方案MindArm，利用深度神经网络将大脑信号翻译成假肢运动，帮助患者执行各种活动 |
| [^2] | [UADA3D: Unsupervised Adversarial Domain Adaptation for 3D Object Detection with Sparse LiDAR and Large Domain Gaps](https://arxiv.org/abs/2403.17633) | UADA3D是一种无监督对抗领域自适应方法，能够在3D物体检测中处理稀疏LiDAR数据和大领域差距，并在自动驾驶汽车和移动机器人领域中表现出显著的改进。 |
| [^3] | [ViSaRL: Visual Reinforcement Learning Guided by Human Saliency](https://arxiv.org/abs/2403.10940) | ViSaRL提出了Visual Saliency-Guided Reinforcement Learning（受视觉显著性引导的强化学习）方法，通过学习视觉表示来显著提高RL代理在不同任务上的成功率、样本效率和泛化性能。 |
| [^4] | [Granger-Causal Hierarchical Skill Discovery.](http://arxiv.org/abs/2306.09509) | 本文介绍了一种名为HIntS的算法，使用无监督检测器，基于Granger因果性捕捉因素之间的关键事件，发现和训练一系列操作因素化环境中的因素的技能，其展示了在机器人推动任务上有2-3倍的样本效率和最终性能的提高，有效的处理了复杂问题和转移学习。 |
| [^5] | [Deep Radar Inverse Sensor Models for Dynamic Occupancy Grid Maps.](http://arxiv.org/abs/2305.12409) | 该研究提出了一种基于深度学习的雷达逆向传感器模型，用于将稀疏雷达检测映射到极坐标测量网格，并生成动态网格地图，实验结果表明该方法优于手工制作的几何ISM。与最先进的深度学习方法相比，该方法为从有限视场的雷达中学习极坐标方案的单帧测量网格的第一个方法。 |

# 详细

[^1]: MindArm: 机械智能非侵入式神经驱动假肢系统

    MindArm: Mechanized Intelligent Non-Invasive Neuro-Driven Prosthetic Arm System

    [https://arxiv.org/abs/2403.19992](https://arxiv.org/abs/2403.19992)

    提出了一种低成本技术解决方案MindArm，利用深度神经网络将大脑信号翻译成假肢运动，帮助患者执行各种活动

    

    目前，残疾或难以移动手臂的人（简称“患者”）在有效解决生理限制方面有非常有限的技术解决方案。这主要是由于两个原因：一是像以思维控制为主的假肢设备通常非常昂贵并需要昂贵的维护；二是其他解决方案需要昂贵的侵入性脑部手术，这种手术风险高，昂贵且维护困难。因此，当前的技术解决方案并不适用于具有不同财务背景的所有患者。为此，我们提出了一种低成本技术解决方案，名为MindArm，即一种机械智能非侵入式神经驱动假肢系统。我们的MindArm系统采用深度神经网络（DNN）引擎将大脑信号翻译成预期的假肢运动，从而帮助患者实施许多活动，尽管他们

    arXiv:2403.19992v1 Announce Type: new  Abstract: Currently, people with disability or difficulty to move their arms (referred to as "patients") have very limited technological solutions to efficiently address their physiological limitations. It is mainly due to two reasons: (1) the non-invasive solutions like mind-controlled prosthetic devices are typically very costly and require expensive maintenance; and (2) other solutions require costly invasive brain surgery, which is high risk to perform, expensive, and difficult to maintain. Therefore, current technological solutions are not accessible for all patients with different financial backgrounds. Toward this, we propose a low-cost technological solution called MindArm, a mechanized intelligent non-invasive neuro-driven prosthetic arm system. Our MindArm system employs a deep neural network (DNN) engine to translate brain signals into the intended prosthetic arm motion, thereby helping patients to perform many activities despite their 
    
[^2]: UADA3D：面向稀疏LiDAR和大领域差距的无监督对抗领域自适应在3D物体检测中的应用

    UADA3D: Unsupervised Adversarial Domain Adaptation for 3D Object Detection with Sparse LiDAR and Large Domain Gaps

    [https://arxiv.org/abs/2403.17633](https://arxiv.org/abs/2403.17633)

    UADA3D是一种无监督对抗领域自适应方法，能够在3D物体检测中处理稀疏LiDAR数据和大领域差距，并在自动驾驶汽车和移动机器人领域中表现出显著的改进。

    

    在这项研究中，我们解决了现有无监督领域适应方法在基于LiDAR的3D物体检测中的一个问题，这些方法主要集中在适应已建立的高密度自动驾驶数据集之间的转变。我们专注于更稀疏的点云，捕捉来自不同视角的场景：不仅来自道路上的车辆，还来自人行道上的移动机器人，遭遇着明显不同的环境条件和传感器配置。我们引入了无监督对抗领域自适应3D物体检测（UADA3D）。UADA3D不依赖于预训练的源模型或师生架构。相反，它使用对抗方法直接学习域不变特征。我们展示了它在各种适应场景中的有效性，在自动驾驶汽车和移动机器人领域均显示出显著的改进。我们的代码是开源的，很快将会提供。

    arXiv:2403.17633v1 Announce Type: cross  Abstract: In this study, we address a gap in existing unsupervised domain adaptation approaches on LiDAR-based 3D object detection, which have predominantly concentrated on adapting between established, high-density autonomous driving datasets. We focus on sparser point clouds, capturing scenarios from different perspectives: not just from vehicles on the road but also from mobile robots on sidewalks, which encounter significantly different environmental conditions and sensor configurations. We introduce Unsupervised Adversarial Domain Adaptation for 3D Object Detection (UADA3D). UADA3D does not depend on pre-trained source models or teacher-student architectures. Instead, it uses an adversarial approach to directly learn domain-invariant features. We demonstrate its efficacy in various adaptation scenarios, showing significant improvements in both self-driving car and mobile robot domains. Our code is open-source and will be available soon.
    
[^3]: ViSaRL：受人类显著性引导的视觉强化学习

    ViSaRL: Visual Reinforcement Learning Guided by Human Saliency

    [https://arxiv.org/abs/2403.10940](https://arxiv.org/abs/2403.10940)

    ViSaRL提出了Visual Saliency-Guided Reinforcement Learning（受视觉显著性引导的强化学习）方法，通过学习视觉表示来显著提高RL代理在不同任务上的成功率、样本效率和泛化性能。

    

    使用强化学习（RL）从高维像素输入培训机器人执行复杂控制任务在样本效率上是低效的，因为图像观察主要由与任务无关的信息组成。相比之下，人类能够在视觉上关注与任务相关的对象和区域。基于这一观察，我们引入了受视觉显著性引导的强化学习（ViSaRL）。使用ViSaRL学习视觉表示显着提高了RL代理在不同任务上，包括DeepMind控制基准、仿真中的机器人操作和真实机器人上的成功率、样本效率和泛化性能。我们提出了将显著性整合到基于CNN和Transformer的编码器中的方法。我们展示使用ViSaRL学习的视觉表示对各种视觉扰动，包括感知噪声和场景变化，都具有鲁棒性。ViSaRL在真实环境中成功率几乎翻了一番。

    arXiv:2403.10940v1 Announce Type: cross  Abstract: Training robots to perform complex control tasks from high-dimensional pixel input using reinforcement learning (RL) is sample-inefficient, because image observations are comprised primarily of task-irrelevant information. By contrast, humans are able to visually attend to task-relevant objects and areas. Based on this insight, we introduce Visual Saliency-Guided Reinforcement Learning (ViSaRL). Using ViSaRL to learn visual representations significantly improves the success rate, sample efficiency, and generalization of an RL agent on diverse tasks including DeepMind Control benchmark, robot manipulation in simulation and on a real robot. We present approaches for incorporating saliency into both CNN and Transformer-based encoders. We show that visual representations learned using ViSaRL are robust to various sources of visual perturbations including perceptual noise and scene variations. ViSaRL nearly doubles success rate on the real-
    
[^4]: Granger因果的分层技能发现

    Granger-Causal Hierarchical Skill Discovery. (arXiv:2306.09509v1 [cs.AI])

    [http://arxiv.org/abs/2306.09509](http://arxiv.org/abs/2306.09509)

    本文介绍了一种名为HIntS的算法，使用无监督检测器，基于Granger因果性捕捉因素之间的关键事件，发现和训练一系列操作因素化环境中的因素的技能，其展示了在机器人推动任务上有2-3倍的样本效率和最终性能的提高，有效的处理了复杂问题和转移学习。

    

    强化学习已经在学习复杂任务的策略方面显示出了有希望的结果，但往往会遭受低样本效率和有限转移的问题。本文介绍了一种名为HIntS的算法，它使用学习得到的交互检测器来发现和训练一系列技能，这些技能操作因素化环境中的因素。受Granger因果性的启发，这些无监督检测器捕捉到因素之间的关键事件，以便高效地学习有用的技能，并将这些技能转移到其他相关任务，这些任务是许多强化学习技术所面临的困境。我们在一个带有障碍物的机器人推动任务上评估了HIntS - 这是一个具有挑战性的领域，在这个领域，其他RL和HRL方法都表现不佳。学习到的技能不仅展示了使用Breakout的变体的转移，而且与可比较的强化学习基线相比，还表现出2-3倍的样本效率和最终性能的提高。HIntS一起证明了一种层次结构的技能发现方法，可以处理复杂问题。

    Reinforcement Learning (RL) has shown promising results learning policies for complex tasks, but can often suffer from low sample efficiency and limited transfer. We introduce the Hierarchy of Interaction Skills (HIntS) algorithm, which uses learned interaction detectors to discover and train a hierarchy of skills that manipulate factors in factored environments. Inspired by Granger causality, these unsupervised detectors capture key events between factors to sample efficiently learn useful skills and transfer those skills to other related tasks -- tasks where many reinforcement learning techniques struggle. We evaluate HIntS on a robotic pushing task with obstacles -- a challenging domain where other RL and HRL methods fall short. The learned skills not only demonstrate transfer using variants of Breakout, a common RL benchmark, but also show 2-3x improvement in both sample efficiency and final performance compared to comparable RL baselines. Together, HIntS demonstrates a proof of co
    
[^5]: 动态占据网格地图的深度雷达逆向传感器模型

    Deep Radar Inverse Sensor Models for Dynamic Occupancy Grid Maps. (arXiv:2305.12409v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2305.12409](http://arxiv.org/abs/2305.12409)

    该研究提出了一种基于深度学习的雷达逆向传感器模型，用于将稀疏雷达检测映射到极坐标测量网格，并生成动态网格地图，实验结果表明该方法优于手工制作的几何ISM。与最先进的深度学习方法相比，该方法为从有限视场的雷达中学习极坐标方案的单帧测量网格的第一个方法。

    

    实现自动驾驶的一个重要步骤是基于传感器输入对车辆环境进行建模。由于其众所周知的优势，雷达成为推断围绕车辆的网格单元占用状态的流行选择。为了解决雷达检测数据稀疏性和噪声问题，我们提出了一种基于深度学习的逆向传感器模型（ISM），用于学习从稀疏雷达检测到极坐标测量网格的映射。改进的基于激光雷达测量的网格用作参考。学习到的雷达测量网格与雷达多普勒速度测量相结合，进一步用于生成动态网格地图（DGM）。在实际的高速公路情景实验中表明，我们的方法优于手工制作的几何ISM。与最先进的深度学习方法相比，我们的方法是第一个从有限视场（FOV）的雷达中学习极坐标方案的单帧测量网格的方法。学习框架使学习到的ISM可以直接嵌入到现有的贝叶斯状态估计方案中，以提高环境建模的准确性。

    To implement autonomous driving, one essential step is to model the vehicle environment based on the sensor inputs. Radars, with their well-known advantages, became a popular option to infer the occupancy state of grid cells surrounding the vehicle. To tackle data sparsity and noise of radar detections, we propose a deep learning-based Inverse Sensor Model (ISM) to learn the mapping from sparse radar detections to polar measurement grids. Improved lidar-based measurement grids are used as reference. The learned radar measurement grids, combined with radar Doppler velocity measurements, are further used to generate a Dynamic Grid Map (DGM). Experiments in real-world highway scenarios show that our approach outperforms the hand-crafted geometric ISMs. In comparison to state-of-the-art deep learning methods, our approach is the first one to learn a single-frame measurement grid in the polar scheme from radars with a limited Field Of View (FOV). The learning framework makes the learned ISM
    

