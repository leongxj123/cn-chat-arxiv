# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Proprioception Is All You Need: Terrain Classification for Boreal Forests](https://arxiv.org/abs/2403.16877) | 通过引入 BorealTC 数据集，结合现有数据集，我们评估了基于卷积神经网络（CNN）和新颖的状态空间模型（SSM）-Mamba体系结构在北方森林地形分类上的表现。 |
| [^2] | [Visuo-Tactile Pretraining for Cable Plugging](https://arxiv.org/abs/2403.11898) | 本文研究了如何将触觉信息纳入模仿学习平台以在复杂任务中提高性能，通过训练机器人代理插拔USB电缆，实现了在微细操纵任务中的进展。 |
| [^3] | [Deep Bayesian Future Fusion for Self-Supervised, High-Resolution, Off-Road Mapping](https://arxiv.org/abs/2403.11876) | 该论文提出了一种深度贝叶斯未来融合的方法，通过自监督的方式实现高分辨率越野地图的制作，为长程预测提供更好的支持。 |

# 详细

[^1]: 感知力就是你所需要的：北方森林的地形分类

    Proprioception Is All You Need: Terrain Classification for Boreal Forests

    [https://arxiv.org/abs/2403.16877](https://arxiv.org/abs/2403.16877)

    通过引入 BorealTC 数据集，结合现有数据集，我们评估了基于卷积神经网络（CNN）和新颖的状态空间模型（SSM）-Mamba体系结构在北方森林地形分类上的表现。

    

    最近的领域机器人学研究强调了抵御不同类型地形的重要性。北方森林特别受到许多限制机动性的地形的影响，这些地形应该在越野自主导航中加以考虑。此外，作为地球上最大的陆地生物群落之一，北方森林是预计自主车辆将日益普及的地区。在本文中，我们通过引入BorealTC来解决这个问题，这是一个用于基于感知力的地形分类（TC）的公开可用数据集。我们的数据集记录了Husky A200的116分钟的惯性测量单元（IMU）、电机电流和轮胎里程数据，重点关注典型的北方森林地形，特别是雪、冰和淤泥壤。结合我们的数据集与另一个来自最新技术的数据集，我们在TC t

    arXiv:2403.16877v1 Announce Type: cross  Abstract: Recent works in field robotics highlighted the importance of resiliency against different types of terrains. Boreal forests, in particular, are home to many mobility-impeding terrains that should be considered for off-road autonomous navigation. Also, being one of the largest land biomes on Earth, boreal forests are an area where autonomous vehicles are expected to become increasingly common. In this paper, we address this issue by introducing BorealTC, a publicly available dataset for proprioceptive-based terrain classification (TC). Recorded with a Husky A200, our dataset contains 116 min of Inertial Measurement Unit (IMU), motor current, and wheel odometry data, focusing on typical boreal forest terrains, notably snow, ice, and silty loam. Combining our dataset with another dataset from the state-of-the-art, we evaluate both a Convolutional Neural Network (CNN) and the novel state space model (SSM)-based Mamba architecture on a TC t
    
[^2]: 视觉-触觉预训练用于插拔电缆

    Visuo-Tactile Pretraining for Cable Plugging

    [https://arxiv.org/abs/2403.11898](https://arxiv.org/abs/2403.11898)

    本文研究了如何将触觉信息纳入模仿学习平台以在复杂任务中提高性能，通过训练机器人代理插拔USB电缆，实现了在微细操纵任务中的进展。

    

    触觉信息是进行精细操纵的关键工具。作为人类，我们在很大程度上依赖触觉信息来理解周围的物体以及如何与其互动。我们不仅使用触摸来执行操纵任务，还用它来学习如何执行这些任务。因此，为了创建能够学习以人类或超人类水平完成操纵任务的机器人代理，我们需要正确地将触觉信息融入技能执行和技能学习中。本文研究了如何将触觉信息纳入模仿学习平台以提高复杂任务的性能。为此，我们着手解决插拔USB电缆的挑战，这是一项依赖于微观视觉-触觉协作的熟练操纵任务。通过将触觉信息纳入模仿学习框架，我们能够训练一个机器人代理插拔USB电缆。

    arXiv:2403.11898v1 Announce Type: cross  Abstract: Tactile information is a critical tool for fine-grain manipulation. As humans, we rely heavily on tactile information to understand objects in our environments and how to interact with them. We use touch not only to perform manipulation tasks but also to learn how to perform these tasks. Therefore, to create robotic agents that can learn to complete manipulation tasks at a human or super-human level of performance, we need to properly incorporate tactile information into both skill execution and skill learning. In this paper, we investigate how we can incorporate tactile information into imitation learning platforms to improve performance on complex tasks. To do this, we tackle the challenge of plugging in a USB cable, a dexterous manipulation task that relies on fine-grain visuo-tactile serving. By incorporating tactile information into imitation learning frameworks, we are able to train a robotic agent to plug in a USB cable - a firs
    
[^3]: 深度贝叶斯未来融合用于自监督、高分辨率、越野地图制作

    Deep Bayesian Future Fusion for Self-Supervised, High-Resolution, Off-Road Mapping

    [https://arxiv.org/abs/2403.11876](https://arxiv.org/abs/2403.11876)

    该论文提出了一种深度贝叶斯未来融合的方法，通过自监督的方式实现高分辨率越野地图的制作，为长程预测提供更好的支持。

    

    资源受限的越野车辆的传感器分辨率有限，这给可靠的越野自主性带来了巨大挑战。为了克服这一局限性，我们提出了一个基于融合未来信息（即未来融合）进行自监督的通用框架。最近的方法利用未来信息以及手工制作的启发式方法来直接监督目标下游任务（例如可穿越性估计）。然而，在本文中，我们选择了一个更为通用的发展方向 - 通过未来融合以自监督的方式时间高效地完成最高分辨率（即每像素2厘米）BEV地图，可用于任何下游任务以获得更好的长程预测。为此，首先，我们创建了一个高分辨率未来融合数据集，其中包含（RGB / 高度）原始稀疏噪音输入和基于地图的密集标签的成对数据。接下来，为了适应传感器的噪声和稀疏性

    arXiv:2403.11876v1 Announce Type: cross  Abstract: The limited sensing resolution of resource-constrained off-road vehicles poses significant challenges towards reliable off-road autonomy. To overcome this limitation, we propose a general framework based on fusing the future information (i.e. future fusion) for self-supervision. Recent approaches exploit this future information alongside the hand-crafted heuristics to directly supervise the targeted downstream tasks (e.g. traversability estimation). However, in this paper, we opt for a more general line of development - time-efficient completion of the highest resolution (i.e. 2cm per pixel) BEV map in a self-supervised manner via future fusion, which can be used for any downstream tasks for better longer range prediction. To this end, first, we create a high-resolution future-fusion dataset containing pairs of (RGB / height) raw sparse and noisy inputs and map-based dense labels. Next, to accommodate the noise and sparsity of the sens
    

