# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AutoRT: Embodied Foundation Models for Large Scale Orchestration of Robotic Agents.](http://arxiv.org/abs/2401.12963) | AutoRT是一个利用现有的基础模型来扩展机器人在未知场景中的部署的系统，通过利用视觉-语言模型和大型语言模型，提出多样化和新颖的指令，并有效地推理自主权和安全性的权衡。 |

# 详细

[^1]: AutoRT：大规模编排机器人代理的具身基础模型

    AutoRT: Embodied Foundation Models for Large Scale Orchestration of Robotic Agents. (arXiv:2401.12963v1 [cs.RO])

    [http://arxiv.org/abs/2401.12963](http://arxiv.org/abs/2401.12963)

    AutoRT是一个利用现有的基础模型来扩展机器人在未知场景中的部署的系统，通过利用视觉-语言模型和大型语言模型，提出多样化和新颖的指令，并有效地推理自主权和安全性的权衡。

    

    拥有语言、视觉和行动等功能的具身基础模型已经彻底改变了利用互联网规模的数据来推理有用任务的能力。然而，训练具身基础模型的一个关键挑战是缺乏基于物理世界的数据。在本文中，我们提出了AutoRT，一个利用现有的基础模型来扩展完全未知场景中操作机器人的部署的系统，只需要最少的人工监督。AutoRT利用视觉-语言模型(VLMs)实现场景理解和基础绑定，并进一步利用大型语言模型(LLMs)提出多样化和新颖的指令，供一组机器人执行。通过利用基础模型的知识来指导数据收集，AutoRT能够有效地推理自主权和安全性的权衡，同时显著扩大机器人学习的数据收集。我们演示了AutoRT向20多个机器人提议指令。

    Foundation models that incorporate language, vision, and more recently actions have revolutionized the ability to harness internet scale data to reason about useful tasks. However, one of the key challenges of training embodied foundation models is the lack of data grounded in the physical world. In this paper, we propose AutoRT, a system that leverages existing foundation models to scale up the deployment of operational robots in completely unseen scenarios with minimal human supervision. AutoRT leverages vision-language models (VLMs) for scene understanding and grounding, and further uses large language models (LLMs) for proposing diverse and novel instructions to be performed by a fleet of robots. Guiding data collection by tapping into the knowledge of foundation models enables AutoRT to effectively reason about autonomy tradeoffs and safety while significantly scaling up data collection for robot learning. We demonstrate AutoRT proposing instructions to over 20 robots across multi
    

