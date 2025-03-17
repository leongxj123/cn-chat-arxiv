# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LIX: Implicitly Infusing Spatial Geometric Prior Knowledge into Visual Semantic Segmentation for Autonomous Driving](https://arxiv.org/abs/2403.08215) | 将双编码器教师模型获得的空间几何先验知识隐式注入单编码器学生模型，通过新的logit蒸馏和特征蒸馏方法，解决自动驾驶中的视觉语义分割问题。 |
| [^2] | [CoPAL: Corrective Planning of Robot Actions with Large Language Models.](http://arxiv.org/abs/2310.07263) | 本文提出了一个具有大规模语言模型的机器人动作纠正规划系统，通过处理生成计划中的物理基础、逻辑和语义错误的再规划策略，实现了在复杂环境中的任务和动作规划。通过仿真和实际场景的验证，证明了该系统的有效性。 |
| [^3] | [Virtual Guidance as a Mid-level Representation for Navigation.](http://arxiv.org/abs/2303.02731) | 该论文介绍了一种名为“虚拟导航”的新技术，通过在智能体的相机视图上叠加彩色路径或球的形式的视觉指引，以易于理解的导航指令传达抽象的导航信息。实验结果表明，在模拟和真实环境中，虚拟导航在遵循计划路径和避开障碍物等多个指标上优于现有方法。 |

# 详细

[^1]: LIX：将空间几何先验知识隐式注入视觉语义分割，用于自动驾驶

    LIX: Implicitly Infusing Spatial Geometric Prior Knowledge into Visual Semantic Segmentation for Autonomous Driving

    [https://arxiv.org/abs/2403.08215](https://arxiv.org/abs/2403.08215)

    将双编码器教师模型获得的空间几何先验知识隐式注入单编码器学生模型，通过新的logit蒸馏和特征蒸馏方法，解决自动驾驶中的视觉语义分割问题。

    

    尽管数据融合网络在视觉语义分割中表现出色，但当缺乏空间几何数据时，双编码器变得无效。将双编码器教师模型获得的空间几何先验知识隐式注入单编码器学生模型是一个实用但不太探索的研究领域。本文深入探讨了这个主题，并采用知识蒸馏方法来解决这个问题。我们引入了Learning to Infuse "X" (LIX) 框架，在logit蒸馏和特征蒸馏方面进行了新颖贡献。我们提出了一个数学证明，强调在解耦知识蒸馏中使用单一固定权重的局限性，并引入了logit智能动态权重控制器作为解决这个问题的方法。此外，我们开发了一种自适应重新校准的特征蒸馏算法，包括两种技术。

    arXiv:2403.08215v1 Announce Type: cross  Abstract: Despite the impressive performance achieved by data-fusion networks with duplex encoders for visual semantic segmentation, they become ineffective when spatial geometric data are not available. Implicitly infusing the spatial geometric prior knowledge acquired by a duplex-encoder teacher model into a single-encoder student model is a practical, albeit less explored research avenue. This paper delves into this topic and resorts to knowledge distillation approaches to address this problem. We introduce the Learning to Infuse "X" (LIX) framework, with novel contributions in both logit distillation and feature distillation aspects. We present a mathematical proof that underscores the limitation of using a single fixed weight in decoupled knowledge distillation and introduce a logit-wise dynamic weight controller as a solution to this issue. Furthermore, we develop an adaptively-recalibrated feature distillation algorithm, including two tec
    
[^2]: CoPAL:具有大规模语言模型的机器人动作纠正规划

    CoPAL: Corrective Planning of Robot Actions with Large Language Models. (arXiv:2310.07263v1 [cs.RO])

    [http://arxiv.org/abs/2310.07263](http://arxiv.org/abs/2310.07263)

    本文提出了一个具有大规模语言模型的机器人动作纠正规划系统，通过处理生成计划中的物理基础、逻辑和语义错误的再规划策略，实现了在复杂环境中的任务和动作规划。通过仿真和实际场景的验证，证明了该系统的有效性。

    

    为了实现完全自主的机器人系统能够接管人类传统执行的任务，开放世界环境的复杂性提出了巨大的挑战。在这一背景下，本研究为应用于机器人任务和动作规划的大规模语言模型领域做出了贡献。我们提出了一个系统架构，协调多个认知层次之间的无缝交互，包括推理、规划和动作生成。其核心是一种处理生成的计划中的物理基础、逻辑和语义错误的新型再规划策略。通过在仿真环境和两个复杂的实际场景（方块世界、酒吧和比萨制作）中进行实证评估，我们展示了所提出的反馈架构的有效性，尤其是对可执行性、正确性和时间复杂性的影响。

    In the pursuit of fully autonomous robotic systems capable of taking over tasks traditionally performed by humans, the complexity of open-world environments poses a considerable challenge. Addressing this imperative, this study contributes to the field of Large Language Models (LLMs) applied to task and motion planning for robots. We propose a system architecture that orchestrates a seamless interplay between multiple cognitive levels, encompassing reasoning, planning, and motion generation. At its core lies a novel replanning strategy that handles physically grounded, logical, and semantic errors in the generated plans. We demonstrate the efficacy of the proposed feedback architecture, particularly its impact on executability, correctness, and time complexity via empirical evaluation in the context of a simulation and two intricate real-world scenarios: blocks world, barman and pizza preparation.
    
[^3]: 导航的中层表示——虚拟导航

    Virtual Guidance as a Mid-level Representation for Navigation. (arXiv:2303.02731v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.02731](http://arxiv.org/abs/2303.02731)

    该论文介绍了一种名为“虚拟导航”的新技术，通过在智能体的相机视图上叠加彩色路径或球的形式的视觉指引，以易于理解的导航指令传达抽象的导航信息。实验结果表明，在模拟和真实环境中，虚拟导航在遵循计划路径和避开障碍物等多个指标上优于现有方法。

    

    在自主导航的背景下，有效地传达抽象的导航指引给动态环境中的智能体存在挑战，特别是当导航信息是多模态的时候。为了解决这个问题，本文引入了一种名为“虚拟导航”的新技术，旨在以视觉方式呈现非视觉指令信号。这些视觉指引以彩色路径或球的形式叠加在智能体的相机视图上，作为易于理解的导航指令。我们通过在模拟和真实环境中进行实验来评估我们提出的方法。在模拟环境中，我们的虚拟导航在多项指标上优于基线混合方法，包括遵循计划路径和避开障碍物。此外，我们将虚拟导航的概念扩展到将基于文本提示的指令转换为用于真实环境实验的直观视觉格式。我们的结果验证了虚拟导航的适应性。

    In the context of autonomous navigation, effectively conveying abstract navigational cues to agents in dynamic environments poses challenges, particularly when the navigation information is multimodal. To address this issue, the paper introduces a novel technique termed "Virtual Guidance," which is designed to visually represent non-visual instructional signals. These visual cues, rendered as colored paths or spheres, are overlaid onto the agent's camera view, serving as easily comprehensible navigational instructions. We evaluate our proposed method through experiments in both simulated and real-world settings. In the simulated environments, our virtual guidance outperforms baseline hybrid approaches in several metrics, including adherence to planned routes and obstacle avoidance. Furthermore, we extend the concept of virtual guidance to transform text-prompt-based instructions into a visually intuitive format for real-world experiments. Our results validate the adaptability of virtua
    

