# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RASP: A Drone-based Reconfigurable Actuation and Sensing Platform Towards Ambient Intelligent Systems](https://arxiv.org/abs/2403.12853) | 提出了RASP，一个可在25秒内自主更换传感器和执行器的模块化和可重构传感和作动平台，使无人机能快速适应各种任务，同时引入了利用大规模语言和视觉语言模型的个人助理系统架构。 |
| [^2] | [Relying on the Unreliable: The Impact of Language Models' Reluctance to Express Uncertainty.](http://arxiv.org/abs/2401.06730) | 本研究调查了语言模型在回答问题时不愿表达不确定性的影响，发现语言模型往往过于自信，导致高错误率。实验还表明用户无论是否标记了确定性都会严重依赖语言模型生成的结果。 |

# 详细

[^1]: 基于无人机的环境智能系统的可重构作动和传感平台RASP

    RASP: A Drone-based Reconfigurable Actuation and Sensing Platform Towards Ambient Intelligent Systems

    [https://arxiv.org/abs/2403.12853](https://arxiv.org/abs/2403.12853)

    提出了RASP，一个可在25秒内自主更换传感器和执行器的模块化和可重构传感和作动平台，使无人机能快速适应各种任务，同时引入了利用大规模语言和视觉语言模型的个人助理系统架构。

    

    实现消费级无人机与我们家中的吸尘机器人或日常生活中的个人智能手机一样有用，需要无人机能感知、驱动和响应可能出现的一般情况。为了实现这一愿景，我们提出了RASP，一个模块化和可重构的传感和作动平台，允许无人机在仅25秒内自主更换机载传感器和执行器，使单个无人机能够快速适应各种任务。RASP包括一个机械层，用于物理更换传感器模块，一个电气层，用于维护传感器/执行器的电源和通信线路，以及一个软件层，用于在无人机和我们平台上的任何传感器模块之间维护一个公共接口。利用最近在大型语言和视觉语言模型方面的进展，我们进一步介绍了一种利用RASP的个人助理系统的架构、实现和现实世界部署。

    arXiv:2403.12853v1 Announce Type: cross  Abstract: Realizing consumer-grade drones that are as useful as robot vacuums throughout our homes or personal smartphones in our daily lives requires drones to sense, actuate, and respond to general scenarios that may arise. Towards this vision, we propose RASP, a modular and reconfigurable sensing and actuation platform that allows drones to autonomously swap onboard sensors and actuators in only 25 seconds, allowing a single drone to quickly adapt to a diverse range of tasks. RASP consists of a mechanical layer to physically swap sensor modules, an electrical layer to maintain power and communication lines to the sensor/actuator, and a software layer to maintain a common interface between the drone and any sensor module in our platform. Leveraging recent advances in large language and visual language models, we further introduce the architecture, implementation, and real-world deployments of a personal assistant system utilizing RASP. We demo
    
[^2]: 不可靠的依赖：语言模型不愿表达不确定性的影响

    Relying on the Unreliable: The Impact of Language Models' Reluctance to Express Uncertainty. (arXiv:2401.06730v1 [cs.CL])

    [http://arxiv.org/abs/2401.06730](http://arxiv.org/abs/2401.06730)

    本研究调查了语言模型在回答问题时不愿表达不确定性的影响，发现语言模型往往过于自信，导致高错误率。实验还表明用户无论是否标记了确定性都会严重依赖语言模型生成的结果。

    

    随着自然语言成为人工智能交互的默认接口，语言模型适当地传达下游应用的不确定性变得至关重要。本研究调查了语言模型如何通过自然语言表达对其回答的置信度，以及下游用户对语言模型表达的不确定性的反应。我们调查了公开部署的模型，发现在回答问题时，即使产生了错误答案，语言模型也无法表达不确定性。虽然可以明确要求语言模型表达置信度，但它们往往过于自信，导致在置信的回答中错误率高达平均47%。我们通过人类实验测试了语言模型过度自信的风险，并证明用户无论是否标记了确定性都会严重依赖语言模型生成的结果。最后，我们研究了在RLHF对齐中使用的偏好注释数据集，并发现人类对带有不确定性的文本有偏见。我们的研究突出了这一问题。

    As natural language becomes the default interface for human-AI interaction, there is a critical need for LMs to appropriately communicate uncertainties in downstream applications. In this work, we investigate how LMs incorporate confidence about their responses via natural language and how downstream users behave in response to LM-articulated uncertainties. We examine publicly deployed models and find that LMs are unable to express uncertainties when answering questions even when they produce incorrect responses. LMs can be explicitly prompted to express confidences, but tend to be overconfident, resulting in high error rates (on average 47%) among confident responses. We test the risks of LM overconfidence by running human experiments and show that users rely heavily on LM generations, whether or not they are marked by certainty. Lastly, we investigate the preference-annotated datasets used in RLHF alignment and find that humans have a bias against texts with uncertainty. Our work hig
    

