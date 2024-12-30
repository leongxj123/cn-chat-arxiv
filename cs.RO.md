# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SRLM: Human-in-Loop Interactive Social Robot Navigation with Large Language Model and Deep Reinforcement Learning](https://arxiv.org/abs/2403.15648) | SRLM 提出了一种结合了大型语言模型和深度强化学习的新型混合方法，用于人机交互式社交机器人导航，通过实时的人类语言指令推断全局规划，并在公共空间中提供多种社交服务，表现出出色的性能。 |
| [^2] | [MaxMin-RLHF: Towards Equitable Alignment of Large Language Models with Diverse Human Preferences](https://arxiv.org/abs/2402.08925) | 这项工作提出了一种公平对齐大型语言模型与多样的人类偏好的方法，通过学习混合偏好分布并使用MaxMin对齐目标来更好地表示人类偏好。 |
| [^3] | [Working Backwards: Learning to Place by Picking](https://arxiv.org/abs/2312.02352) | 通过逆向抓取过程并利用拾取和放置问题的对称性，提出了一种通过拾取的放置方法，并用自主收集的演示直接训练策略，实现在接触受限环境下物体放置任务的自主收集和泛化。 |
| [^4] | [PhotoBot: Reference-Guided Interactive Photography via Natural Language.](http://arxiv.org/abs/2401.11061) | PhotoBot是一个通过自然语言引导和机器人摄影师相互作用的自动化照片获取框架。它利用视觉语言模型和物体检测器来提供摄影建议，并通过视觉变换器计算相机的姿态调整，从而实现高质量的照片获取。 |

# 详细

[^1]: SRLM: 使用大型语言模型和深度强化学习进行人机交互式社交机器人导航

    SRLM: Human-in-Loop Interactive Social Robot Navigation with Large Language Model and Deep Reinforcement Learning

    [https://arxiv.org/abs/2403.15648](https://arxiv.org/abs/2403.15648)

    SRLM 提出了一种结合了大型语言模型和深度强化学习的新型混合方法，用于人机交互式社交机器人导航，通过实时的人类语言指令推断全局规划，并在公共空间中提供多种社交服务，表现出出色的性能。

    

    一名交互式社交机器人助手必须在复杂拥挤的空间中提供服务，根据实时的人类语言指令或反馈调整其行为。本文提出了一种名为Social Robot Planner (SRLM) 的新型混合方法，它将大型语言模型（LLM）和深度强化学习（DRL）整合起来，以在充斥着人群的公共空间中导航，并提供多种社交服务。SRLM 通过实时的人机交互指令推断全局规划，并将社交信息编码到基于LLM的大型导航模型（LNM）中，用于低层次的运动执行。此外，设计了一个基于DRL的规划器来保持基准性能，通过大型反馈模型（LFM）与LNM融合，以解决当前文本和LLM驱动的LNM的不稳定性。最后，SRLM 在广泛的实验中展示出了出色的性能。有关此工作的更多详细信息，请访问：https://sites.g

    arXiv:2403.15648v1 Announce Type: cross  Abstract: An interactive social robotic assistant must provide services in complex and crowded spaces while adapting its behavior based on real-time human language commands or feedback. In this paper, we propose a novel hybrid approach called Social Robot Planner (SRLM), which integrates Large Language Models (LLM) and Deep Reinforcement Learning (DRL) to navigate through human-filled public spaces and provide multiple social services. SRLM infers global planning from human-in-loop commands in real-time, and encodes social information into a LLM-based large navigation model (LNM) for low-level motion execution. Moreover, a DRL-based planner is designed to maintain benchmarking performance, which is blended with LNM by a large feedback model (LFM) to address the instability of current text and LLM-driven LNM. Finally, SRLM demonstrates outstanding performance in extensive experiments. More details about this work are available at: https://sites.g
    
[^2]: MaxMin-RLHF:面向具有多样的人类偏好的大型语言模型的公平对齐

    MaxMin-RLHF: Towards Equitable Alignment of Large Language Models with Diverse Human Preferences

    [https://arxiv.org/abs/2402.08925](https://arxiv.org/abs/2402.08925)

    这项工作提出了一种公平对齐大型语言模型与多样的人类偏好的方法，通过学习混合偏好分布并使用MaxMin对齐目标来更好地表示人类偏好。

    

    强化学习从人类反馈中学习(RLHF)通过使用从偏好数据中派生的单一奖励模型来对齐语言模型与人类偏好一致。然而，这种方法忽视了从多个用户收集的数据中固有的人类偏好的丰富多样性。在这项工作中，我们首先推导出了使用单一奖励RLHF进行对齐的不可能性结果，从而凸显了其无法表示多样的人类偏好。为了提供一个公平的解决方案，我们通过期望最大化算法学习了一种混合偏好分布，并提出了一种受社会选择理论中的平等原则启发的MaxMin对齐目标来更好地表示多样的人类偏好。我们阐明了我们提出的方法与分布稳健优化和通用效用RL的联系，从而突显了我们提出的方法的普适性和鲁棒性。

    arXiv:2402.08925v1 Announce Type: cross Abstract: Reinforcement Learning from Human Feedback (RLHF) aligns language models to human preferences by employing a singular reward model derived from preference data. However, such an approach overlooks the rich diversity of human preferences inherent in data collected from multiple users. In this work, we first derive an impossibility result of alignment with single reward RLHF, thereby highlighting its insufficiency in representing diverse human preferences. To provide an equitable solution to the problem, we learn a mixture of preference distributions via an expectation-maximization algorithm and propose a MaxMin alignment objective for policy learning inspired by the Egalitarian principle in social choice theory to better represent diverse human preferences. We elucidate the connection of our proposed approach to distributionally robust optimization and general utility RL, thereby highlighting the generality and robustness of our proposed
    
[^3]: 逆向学习：通过捡取学习放置

    Working Backwards: Learning to Place by Picking

    [https://arxiv.org/abs/2312.02352](https://arxiv.org/abs/2312.02352)

    通过逆向抓取过程并利用拾取和放置问题的对称性，提出了一种通过拾取的放置方法，并用自主收集的演示直接训练策略，实现在接触受限环境下物体放置任务的自主收集和泛化。

    

    我们提出了一种通过拾取（PvP）的放置方法，可以自主收集适用于一系列放置任务的现实世界演示，其中物体必须被操纵到特定的接触限制位置。通过PvP，我们通过颠倒抓取过程并利用拾取和放置问题固有的对称性，接近于机器人物体放置演示的收集。具体而言，我们从一组最初位于目标放置位置的物体的抓取序列中获得放置演示。我们的系统可以在接触受限环境中收集数百个演示，而无需人类干预，这是通过结合两个模块实现的：触觉重新抓取和用于抓取的顺从控制。我们通过行为克隆直接从视觉观察中通过自主收集的演示中训练策略。通过这样做，策略可以推广到超出训练环境范围的物体放置场景。

    arXiv:2312.02352v2 Announce Type: replace-cross  Abstract: We present placing via picking (PvP), a method to autonomously collect real-world demonstrations for a family of placing tasks in which objects must be manipulated to specific contact-constrained locations. With PvP, we approach the collection of robotic object placement demonstrations by reversing the grasping process and exploiting the inherent symmetry of the pick and place problems. Specifically, we obtain placing demonstrations from a set of grasp sequences of objects initially located at their target placement locations. Our system can collect hundreds of demonstrations in contact-constrained environments without human intervention by combining two modules: tactile regrasping and compliant control for grasps. We train a policy directly from visual observations through behavioral cloning, using the autonomously-collected demonstrations. By doing so, the policy can generalize to object placement scenarios outside of the tra
    
[^4]: PhotoBot：通过自然语言引导的参考互动摄影

    PhotoBot: Reference-Guided Interactive Photography via Natural Language. (arXiv:2401.11061v1 [cs.CV])

    [http://arxiv.org/abs/2401.11061](http://arxiv.org/abs/2401.11061)

    PhotoBot是一个通过自然语言引导和机器人摄影师相互作用的自动化照片获取框架。它利用视觉语言模型和物体检测器来提供摄影建议，并通过视觉变换器计算相机的姿态调整，从而实现高质量的照片获取。

    

    我们介绍了一个名为PhotoBot的框架，它基于高级人类语言引导和机器人摄影师之间的相互作用，用于自动化的照片获取。我们建议通过从策展画廊中检索到的参考图片向用户传达摄影建议。我们利用视觉语言模型（VLM）和物体检测器，通过文本描述对参考图片进行特征化，并使用大型语言模型（LLM）通过基于用户语言查询的文本推理检索相关的参考图片。为了对应参考图片和观察到的场景，我们利用一个能够捕捉显著不同的图像的语义相似性的预训练特征的视觉变换器，通过解决一个透视n-点（PnP）问题来计算RGB-D相机的姿态调整。我们在配备有手腕相机的真实机械手臂上演示了我们的方法。我们的用户研究表明，由PhotoBot拍摄的照片具有良好的质量和效果。

    We introduce PhotoBot, a framework for automated photo acquisition based on an interplay between high-level human language guidance and a robot photographer. We propose to communicate photography suggestions to the user via a reference picture that is retrieved from a curated gallery. We exploit a visual language model (VLM) and an object detector to characterize reference pictures via textual descriptions and use a large language model (LLM) to retrieve relevant reference pictures based on a user's language query through text-based reasoning. To correspond the reference picture and the observed scene, we exploit pre-trained features from a vision transformer capable of capturing semantic similarity across significantly varying images. Using these features, we compute pose adjustments for an RGB-D camera by solving a Perspective-n-Point (PnP) problem. We demonstrate our approach on a real-world manipulator equipped with a wrist camera. Our user studies show that photos taken by PhotoBo
    

