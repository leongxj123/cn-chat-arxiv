# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Survey for Foundation Models in Autonomous Driving](https://rss.arxiv.org/abs/2402.01105) | 本综述论文回顾了40多篇研究论文，总结了基于基础模型的自动驾驶在规划、仿真和关键任务方面的重要贡献，强调了大型语言模型的推理和翻译能力，视觉基础模型在物体检测和驾驶场景创建方面的应用，以及多模态基础模型的视觉理解和空间推理能力。 |
| [^2] | [LLM^3:Large Language Model-based Task and Motion Planning with Motion Failure Reasoning](https://arxiv.org/abs/2403.11552) | LLM^3是一个基于大型语言模型的任务和运动规划框架，利用预训练的LLM具备强大的推理和规划能力，通过接口提出符号动作序列和选择连续动作参数进行运动规划，并通过运动规划的反馈来迭代优化提议，从而简化了处理领域特定消息的设计过程。 |
| [^3] | [ComTraQ-MPC: Meta-Trained DQN-MPC Integration for Trajectory Tracking with Limited Active Localization Updates](https://arxiv.org/abs/2403.01564) | ComTraQ-MPC是一个结合了DQN和MPC的新框架，旨在优化在有限主动定位更新下的轨迹跟踪。 |

# 详细

[^1]: 自动驾驶领域基础模型综述

    A Survey for Foundation Models in Autonomous Driving

    [https://rss.arxiv.org/abs/2402.01105](https://rss.arxiv.org/abs/2402.01105)

    本综述论文回顾了40多篇研究论文，总结了基于基础模型的自动驾驶在规划、仿真和关键任务方面的重要贡献，强调了大型语言模型的推理和翻译能力，视觉基础模型在物体检测和驾驶场景创建方面的应用，以及多模态基础模型的视觉理解和空间推理能力。

    

    基于基础模型的出现，自然语言处理和计算机视觉领域发生了革命，为自动驾驶应用铺平了道路。本综述论文对40多篇研究论文进行了全面的回顾，展示了基础模型在提升自动驾驶中的作用。大型语言模型在自动驾驶的规划和仿真中发挥着重要作用，特别是通过其在推理、代码生成和翻译方面的能力。与此同时，视觉基础模型在关键任务中得到越来越广泛的应用，例如三维物体检测和跟踪，以及为仿真和测试创建逼真的驾驶场景。多模态基础模型可以整合多样的输入，展现出卓越的视觉理解和空间推理能力，对于端到端自动驾驶至关重要。本综述不仅提供了一个结构化的分类，根据模态和自动驾驶领域中的功能对基础模型进行分类，还深入研究了方法。

    The advent of foundation models has revolutionized the fields of natural language processing and computer vision, paving the way for their application in autonomous driving (AD). This survey presents a comprehensive review of more than 40 research papers, demonstrating the role of foundation models in enhancing AD. Large language models contribute to planning and simulation in AD, particularly through their proficiency in reasoning, code generation and translation. In parallel, vision foundation models are increasingly adapted for critical tasks such as 3D object detection and tracking, as well as creating realistic driving scenarios for simulation and testing. Multi-modal foundation models, integrating diverse inputs, exhibit exceptional visual understanding and spatial reasoning, crucial for end-to-end AD. This survey not only provides a structured taxonomy, categorizing foundation models based on their modalities and functionalities within the AD domain but also delves into the meth
    
[^2]: LLM^3:基于大型语言模型的任务和运动规划以及运动失败推理

    LLM^3:Large Language Model-based Task and Motion Planning with Motion Failure Reasoning

    [https://arxiv.org/abs/2403.11552](https://arxiv.org/abs/2403.11552)

    LLM^3是一个基于大型语言模型的任务和运动规划框架，利用预训练的LLM具备强大的推理和规划能力，通过接口提出符号动作序列和选择连续动作参数进行运动规划，并通过运动规划的反馈来迭代优化提议，从而简化了处理领域特定消息的设计过程。

    

    传统任务和运动规划（TAMP）方法依赖于手工设计的界面，将符号任务规划与连续运动生成连接起来。这些特定领域的、劳动密集型的模块在处理现实世界设置中出现的新任务方面有限。在这里，我们提出了LLM^3，这是一个新颖的基于大型语言模型（LLM）的TAMP框架，具有领域无关的接口。具体来说，我们利用预训练的LLM的强大推理和规划能力来提出符号动作序列，并选择连续动作参数进行运动规划。关键是，LLM^3通过提示将运动规划反馈到其中，使得LLM能够通过对运动失败进行推理来迭代地优化其提议。因此，LLM^3在任务规划和运动规划之间建立接口，减轻了处理它们之间特定领域消息的复杂设计过程。通过一系列仿真

    arXiv:2403.11552v1 Announce Type: cross  Abstract: Conventional Task and Motion Planning (TAMP) approaches rely on manually crafted interfaces connecting symbolic task planning with continuous motion generation. These domain-specific and labor-intensive modules are limited in addressing emerging tasks in real-world settings. Here, we present LLM^3, a novel Large Language Model (LLM)-based TAMP framework featuring a domain-independent interface. Specifically, we leverage the powerful reasoning and planning capabilities of pre-trained LLMs to propose symbolic action sequences and select continuous action parameters for motion planning. Crucially, LLM^3 incorporates motion planning feed- back through prompting, allowing the LLM to iteratively refine its proposals by reasoning about motion failure. Consequently, LLM^3 interfaces between task planning and motion planning, alleviating the intricate design process of handling domain- specific messages between them. Through a series of simulat
    
[^3]: ComTraQ-MPC：元训练的DQN-MPC集成用于具有有限主动定位更新的轨迹跟踪

    ComTraQ-MPC: Meta-Trained DQN-MPC Integration for Trajectory Tracking with Limited Active Localization Updates

    [https://arxiv.org/abs/2403.01564](https://arxiv.org/abs/2403.01564)

    ComTraQ-MPC是一个结合了DQN和MPC的新框架，旨在优化在有限主动定位更新下的轨迹跟踪。

    

    在局部可观察、随机环境中进行轨迹跟踪的最佳决策往往面临着一个重要挑战，即主动定位更新数量有限，这是指代理从传感器获取真实状态信息的过程。传统方法往往难以平衡资源保存、准确状态估计和精确跟踪之间的关系，导致性能次优。本文介绍了ComTraQ-MPC，这是一个结合了Deep Q-Networks (DQN)和模型预测控制(MPC)的新颖框架，旨在优化有限主动定位更新下的轨迹跟踪。元训练的DQN确保了自适应主动定位调度，同时

    arXiv:2403.01564v1 Announce Type: cross  Abstract: Optimal decision-making for trajectory tracking in partially observable, stochastic environments where the number of active localization updates -- the process by which the agent obtains its true state information from the sensors -- are limited, presents a significant challenge. Traditional methods often struggle to balance resource conservation, accurate state estimation and precise tracking, resulting in suboptimal performance. This problem is particularly pronounced in environments with large action spaces, where the need for frequent, accurate state data is paramount, yet the capacity for active localization updates is restricted by external limitations. This paper introduces ComTraQ-MPC, a novel framework that combines Deep Q-Networks (DQN) and Model Predictive Control (MPC) to optimize trajectory tracking with constrained active localization updates. The meta-trained DQN ensures adaptive active localization scheduling, while the
    

