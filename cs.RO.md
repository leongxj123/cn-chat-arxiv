# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DexCap: Scalable and Portable Mocap Data Collection System for Dexterous Manipulation](https://arxiv.org/abs/2403.07788) | DexCap是一个可移植的手部动作捕捉系统，结合DexIL算法从人类手部运动数据中训练机器人技能，具有精确追踪和复制人类动作的能力。 |
| [^2] | [Deep Reinforcement Learning with Dynamic Graphs for Adaptive Informative Path Planning](https://arxiv.org/abs/2402.04894) | 提出了一种利用动态图的深度强化学习方法，用于自适应信息路径规划，能够在未知的三维环境中映射出感兴趣的目标。 |
| [^3] | [DexDiffuser: Generating Dexterous Grasps with Diffusion Models](https://arxiv.org/abs/2402.02989) | DexDiffuser是一种使用扩散模型生成灵巧抓取姿势的新方法，通过对物体点云的生成、评估和优化，实现了较高的抓取成功率。 |
| [^4] | [PhotoBot: Reference-Guided Interactive Photography via Natural Language.](http://arxiv.org/abs/2401.11061) | PhotoBot是一个通过自然语言引导和机器人摄影师相互作用的自动化照片获取框架。它利用视觉语言模型和物体检测器来提供摄影建议，并通过视觉变换器计算相机的姿态调整，从而实现高质量的照片获取。 |
| [^5] | [Robust Pivoting Manipulation using Contact Implicit Bilevel Optimization.](http://arxiv.org/abs/2303.08965) | 本文使用接触隐式双层优化来规划支点操纵并增加鲁棒性，通过利用摩擦力来弥补物体和环境物理属性估计中的不准确性，以应对不确定性影响。 |

# 详细

[^1]: DexCap：用于灵巧操作的可扩展和可移植动作捕捉数据收集系统

    DexCap: Scalable and Portable Mocap Data Collection System for Dexterous Manipulation

    [https://arxiv.org/abs/2403.07788](https://arxiv.org/abs/2403.07788)

    DexCap是一个可移植的手部动作捕捉系统，结合DexIL算法从人类手部运动数据中训练机器人技能，具有精确追踪和复制人类动作的能力。

    

    从人类手部运动数据中学习是为机器人赋予类人灵巧在现实操纵任务中的潜在途径，然而，现存手部动作捕捉系统的可移植性以及将动作捕捉数据转化为有效控制策略的困难仍然存在挑战。为了应对这些问题，我们引入了DexCap，一个便携式手部动作捕捉系统，以及DexIL，一种新颖的模仿算法，可直接从人类手部动作捕捉数据训练灵巧机器人技能。DexCap基于SLAM和电磁场以及环境的3D观察，提供了对手腕和手指运动的精确、抗遮挡的跟踪。利用这一丰富的数据集，DexIL采用逆运动学和基于点云的模仿学习来复制人类动作与机器人手。除了从人类运动中学习外，DexCap还提供了一种op

    arXiv:2403.07788v1 Announce Type: cross  Abstract: Imitation learning from human hand motion data presents a promising avenue for imbuing robots with human-like dexterity in real-world manipulation tasks. Despite this potential, substantial challenges persist, particularly with the portability of existing hand motion capture (mocap) systems and the difficulty of translating mocap data into effective control policies. To tackle these issues, we introduce DexCap, a portable hand motion capture system, alongside DexIL, a novel imitation algorithm for training dexterous robot skills directly from human hand mocap data. DexCap offers precise, occlusion-resistant tracking of wrist and finger motions based on SLAM and electromagnetic field together with 3D observations of the environment. Utilizing this rich dataset, DexIL employs inverse kinematics and point cloud-based imitation learning to replicate human actions with robot hands. Beyond learning from human motion, DexCap also offers an op
    
[^2]: 利用动态图的深度强化学习进行自适应信息路径规划

    Deep Reinforcement Learning with Dynamic Graphs for Adaptive Informative Path Planning

    [https://arxiv.org/abs/2402.04894](https://arxiv.org/abs/2402.04894)

    提出了一种利用动态图的深度强化学习方法，用于自适应信息路径规划，能够在未知的三维环境中映射出感兴趣的目标。

    

    自主机器人常常被用于数据收集，因为它们高效且劳动成本低。机器人数据采集的关键任务是在初始未知环境中规划路径，以满足平台特定的资源约束，例如有限的电池寿命。在三维环境中进行自适应在线路径规划面临着很多挑战，包括大量有效动作的存在以及未知遮挡物的存在。为了解决这些问题，我们提出了一种新颖的深度强化学习方法，用于自适应重新规划机器人路径以在未知的三维环境中映射出感兴趣的目标。我们方法的关键之处在于构建动态图，将规划动作限制在机器人附近，使我们能够快速响应新发现的障碍和感兴趣的目标。对于重新规划，我们提出了一种新的奖励函数，平衡探索未知环境和利用在线收集的有关感兴趣目标的数据。

    Autonomous robots are often employed for data collection due to their efficiency and low labour costs. A key task in robotic data acquisition is planning paths through an initially unknown environment to collect observations given platform-specific resource constraints, such as limited battery life. Adaptive online path planning in 3D environments is challenging due to the large set of valid actions and the presence of unknown occlusions. To address these issues, we propose a novel deep reinforcement learning approach for adaptively replanning robot paths to map targets of interest in unknown 3D environments. A key aspect of our approach is a dynamically constructed graph that restricts planning actions local to the robot, allowing us to quickly react to newly discovered obstacles and targets of interest. For replanning, we propose a new reward function that balances between exploring the unknown environment and exploiting online-collected data about the targets of interest. Our experi
    
[^3]: DexDiffuser: 使用扩散模型生成灵巧抓取姿势

    DexDiffuser: Generating Dexterous Grasps with Diffusion Models

    [https://arxiv.org/abs/2402.02989](https://arxiv.org/abs/2402.02989)

    DexDiffuser是一种使用扩散模型生成灵巧抓取姿势的新方法，通过对物体点云的生成、评估和优化，实现了较高的抓取成功率。

    

    我们引入了DexDiffuser，一种新颖的灵巧抓取方法，能够在部分物体点云上生成、评估和优化抓取姿势。DexDiffuser包括条件扩散型抓取采样器DexSampler和灵巧抓取评估器DexEvaluator。DexSampler通过对随机抓取进行迭代去噪，生成与物体点云条件相关的高质量抓取姿势。我们还引入了两种抓取优化策略：基于评估器的扩散(Evaluator-Guided Diffusion，EGD)和基于评估器的采样优化(Evaluator-based Sampling Refinement，ESR)。我们在虚拟环境和真实世界的实验中，使用Allegro Hand进行测试，结果表明DexDiffuser相比最先进的多指抓取生成方法FFHNet，平均抓取成功率提高了21.71-22.20%。

    We introduce DexDiffuser, a novel dexterous grasping method that generates, evaluates, and refines grasps on partial object point clouds. DexDiffuser includes the conditional diffusion-based grasp sampler DexSampler and the dexterous grasp evaluator DexEvaluator. DexSampler generates high-quality grasps conditioned on object point clouds by iterative denoising of randomly sampled grasps. We also introduce two grasp refinement strategies: Evaluator-Guided Diffusion (EGD) and Evaluator-based Sampling Refinement (ESR). Our simulation and real-world experiments on the Allegro Hand consistently demonstrate that DexDiffuser outperforms the state-of-the-art multi-finger grasp generation method FFHNet with an, on average, 21.71--22.20\% higher grasp success rate.
    
[^4]: PhotoBot：通过自然语言引导的参考互动摄影

    PhotoBot: Reference-Guided Interactive Photography via Natural Language. (arXiv:2401.11061v1 [cs.CV])

    [http://arxiv.org/abs/2401.11061](http://arxiv.org/abs/2401.11061)

    PhotoBot是一个通过自然语言引导和机器人摄影师相互作用的自动化照片获取框架。它利用视觉语言模型和物体检测器来提供摄影建议，并通过视觉变换器计算相机的姿态调整，从而实现高质量的照片获取。

    

    我们介绍了一个名为PhotoBot的框架，它基于高级人类语言引导和机器人摄影师之间的相互作用，用于自动化的照片获取。我们建议通过从策展画廊中检索到的参考图片向用户传达摄影建议。我们利用视觉语言模型（VLM）和物体检测器，通过文本描述对参考图片进行特征化，并使用大型语言模型（LLM）通过基于用户语言查询的文本推理检索相关的参考图片。为了对应参考图片和观察到的场景，我们利用一个能够捕捉显著不同的图像的语义相似性的预训练特征的视觉变换器，通过解决一个透视n-点（PnP）问题来计算RGB-D相机的姿态调整。我们在配备有手腕相机的真实机械手臂上演示了我们的方法。我们的用户研究表明，由PhotoBot拍摄的照片具有良好的质量和效果。

    We introduce PhotoBot, a framework for automated photo acquisition based on an interplay between high-level human language guidance and a robot photographer. We propose to communicate photography suggestions to the user via a reference picture that is retrieved from a curated gallery. We exploit a visual language model (VLM) and an object detector to characterize reference pictures via textual descriptions and use a large language model (LLM) to retrieve relevant reference pictures based on a user's language query through text-based reasoning. To correspond the reference picture and the observed scene, we exploit pre-trained features from a vision transformer capable of capturing semantic similarity across significantly varying images. Using these features, we compute pose adjustments for an RGB-D camera by solving a Perspective-n-Point (PnP) problem. We demonstrate our approach on a real-world manipulator equipped with a wrist camera. Our user studies show that photos taken by PhotoBo
    
[^5]: 使用接触隐式双层优化实现鲁棒的支点操作

    Robust Pivoting Manipulation using Contact Implicit Bilevel Optimization. (arXiv:2303.08965v1 [cs.RO])

    [http://arxiv.org/abs/2303.08965](http://arxiv.org/abs/2303.08965)

    本文使用接触隐式双层优化来规划支点操纵并增加鲁棒性，通过利用摩擦力来弥补物体和环境物理属性估计中的不准确性，以应对不确定性影响。

    

    通用操纵要求机器人能够与新物体和环境进行交互。这个要求使得操纵变得异常具有挑战性，因为机器人必须考虑到不确定因素下的复杂摩擦相互作用及物体和环境的物理属性估计的不准确性。本文研究了支点操作规划的鲁棒优化问题，提供了如何利用摩擦力来弥补物理特性估计中的不准确性的见解。在某些假设下，导出了摩擦力提供的支点操作稳定裕度的解析表达式。然后，在接触隐式双层优化(CIBO)框架中使用该裕度来优化轨迹 ，以增强对物体多个物理参数不确定性的鲁棒性。我们在实际机器人上的实验中，对于严重干扰的参数，分析了稳定裕度，并显示了优化轨迹的改善鲁棒性。

    Generalizable manipulation requires that robots be able to interact with novel objects and environment. This requirement makes manipulation extremely challenging as a robot has to reason about complex frictional interactions with uncertainty in physical properties of the object and the environment. In this paper, we study robust optimization for planning of pivoting manipulation in the presence of uncertainties. We present insights about how friction can be exploited to compensate for inaccuracies in the estimates of the physical properties during manipulation. Under certain assumptions, we derive analytical expressions for stability margin provided by friction during pivoting manipulation. This margin is then used in a Contact Implicit Bilevel Optimization (CIBO) framework to optimize a trajectory that maximizes this stability margin to provide robustness against uncertainty in several physical parameters of the object. We present analysis of the stability margin with respect to sever
    

