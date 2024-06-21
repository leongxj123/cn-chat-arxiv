# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Embedding Large Language Models into Extended Reality: Opportunities and Challenges for Inclusion, Engagement, and Privacy](https://arxiv.org/abs/2402.03907) | 本研究探讨了将大型语言模型嵌入扩展现实中的机会和挑战，认为通过使用LLMs可以实现扩展现实的更包容和参与，并有望推动其在日常生活中的广泛应用。 |
| [^2] | [Toward Human-AI Alignment in Large-Scale Multi-Player Games](https://arxiv.org/abs/2402.03575) | 本研究提出了一种在大规模多人游戏中评估人工智能与人类协作的方法，通过分析人类游戏数据和训练AI代理来比较和对比人类和AI的行为差异，以识别高级行为概念。 |
| [^3] | [DOCTOR: A Multi-Disease Detection Continual Learning Framework Based on Wearable Medical Sensors.](http://arxiv.org/abs/2305.05738) | DOCTOR是一种基于可穿戴医疗传感器的多疾病检测持续学习框架，采用了多头深度神经网络和Exemplar-replay风格的CL算法。它可以不断地学习新任务，并在内存使用、电池消耗和检测复杂度方面优于传统的ML驱动疾病检测方法。 |
| [^4] | [Rethinking AI Explainability and Plausibility.](http://arxiv.org/abs/2303.17707) | 本文研究了XAI评估中最普遍的人为概念——解释合理性。虽然一直被制定为AI可解释性任务的重要评估目标，但是评估XAI的合理性有时是有害的，且无法达到模型可理解性、透明度和可信度的目的。 |

# 详细

[^1]: 将大型语言模型嵌入扩展现实中：包容性、参与度和隐私的机会和挑战

    Embedding Large Language Models into Extended Reality: Opportunities and Challenges for Inclusion, Engagement, and Privacy

    [https://arxiv.org/abs/2402.03907](https://arxiv.org/abs/2402.03907)

    本研究探讨了将大型语言模型嵌入扩展现实中的机会和挑战，认为通过使用LLMs可以实现扩展现实的更包容和参与，并有望推动其在日常生活中的广泛应用。

    

    计算机图形学、硬件、人工智能和人机交互的最新发展可能导致扩展现实设备的普及。在这篇论文中，我们提出将大型语言模型（LLMs）嵌入扩展现实中，通过将它们嵌入虚拟角色或作为叙事方式，来促进更包容的体验。我们认为这种包容性将有助于扩展现实的多样性使用。此外，我们相信LLMs的多功能对话能力将增加用户与扩展现实环境的参与度，从而帮助扩展现实更广泛地应用于日常生活中。

    Recent developments in computer graphics, hardware, artificial intelligence (AI), and human-computer interaction likely lead to extended reality (XR) devices and setups being more pervasive. While these devices and setups provide users with interactive, engaging, and immersive experiences with different sensing modalities, such as eye and hand trackers, many non-player characters are utilized in a pre-scripted way or by conventional AI techniques. In this paper, we argue for using large language models (LLMs) in XR by embedding them in virtual avatars or as narratives to facilitate more inclusive experiences through prompt engineering according to user profiles and fine-tuning the LLMs for particular purposes. We argue that such inclusion will facilitate diversity for XR use. In addition, we believe that with the versatile conversational capabilities of LLMs, users will engage more with XR environments, which might help XR be more used in everyday life. Lastly, we speculate that combin
    
[^2]: 在大规模多人游戏中实现人工智能与人类的协同

    Toward Human-AI Alignment in Large-Scale Multi-Player Games

    [https://arxiv.org/abs/2402.03575](https://arxiv.org/abs/2402.03575)

    本研究提出了一种在大规模多人游戏中评估人工智能与人类协作的方法，通过分析人类游戏数据和训练AI代理来比较和对比人类和AI的行为差异，以识别高级行为概念。

    

    在复杂的多智能体游戏中实现人工智能与人类的协同对于创建增强游戏体验的可信任的人工智能代理至关重要。我们提出了一种方法来评估这种协同，使用可解释的任务集框架，重点关注高级行为任务而非低级策略。我们的方法有三个组成部分。首先，我们分析了来自Xbox的Bleeding Edge（10万+游戏）的大量人类游戏数据，揭示了复杂任务空间中的行为模式。这个任务空间作为行为流形的基础集合，捕捉可解释的轴：战斗-逃跑、探索-利用以及单人-多人智能体。其次，我们训练一个使用生成预训练因果变换器的人工智能代理来玩Bleeding Edge，并测量其行为。第三，我们将人类和人工智能游戏映射到提出的行为流形中进行比较和对比。这样可以解释策略的差异，如，我们发现人类玩家在战斗-逃跑方面表现变化多样。

    Achieving human-AI alignment in complex multi-agent games is crucial for creating trustworthy AI agents that enhance gameplay. We propose a method to evaluate this alignment using an interpretable task-sets framework, focusing on high-level behavioral tasks instead of low-level policies. Our approach has three components. First, we analyze extensive human gameplay data from Xbox's Bleeding Edge (100K+ games), uncovering behavioral patterns in a complex task space. This task space serves as a basis set for a behavior manifold capturing interpretable axes: fight-flight, explore-exploit, and solo-multi-agent. Second, we train an AI agent to play Bleeding Edge using a Generative Pretrained Causal Transformer and measure its behavior. Third, we project human and AI gameplay to the proposed behavior manifold to compare and contrast. This allows us to interpret differences in policy as higher-level behavioral concepts, e.g., we find that while human players exhibit variability in fight-flight
    
[^3]: DOCTOR：基于可穿戴医疗传感器的多疾病检测持续学习框架

    DOCTOR: A Multi-Disease Detection Continual Learning Framework Based on Wearable Medical Sensors. (arXiv:2305.05738v1 [cs.LG])

    [http://arxiv.org/abs/2305.05738](http://arxiv.org/abs/2305.05738)

    DOCTOR是一种基于可穿戴医疗传感器的多疾病检测持续学习框架，采用了多头深度神经网络和Exemplar-replay风格的CL算法。它可以不断地学习新任务，并在内存使用、电池消耗和检测复杂度方面优于传统的ML驱动疾病检测方法。

    

    现代机器学习（ML）和边缘设备中的可穿戴医疗传感器（WMS）的进步使得智能医疗的ML驱动疾病检测成为可能。传统的ML驱动疾病检测方法依赖于为每种疾病和相应的WMS数据定制个别模型。然而，这种方法缺乏对分布变化和新任务分类的适应性。同时，为了检测每个新疾病，需要从头开始重新构建和训练模型。针对这些挑战，我们提出了基于WMS的多疾病检测持续学习框架DOCTOR。它采用了多头深度神经网络（DNN）和一种Exemplar-replay风格的CL算法。CL算法使得框架能够不断地学习新任务，其中涉及不同的数据分布、分类类别和疾病检测任务。DOCTOR在使用来自实际WMS的公共数据集进行四种常见疾病检测方面取得了最先进的性能。同时，在内存使用、电池消耗和检测复杂度方面，DOCTOR也优于基线方法。

    Modern advances in machine learning (ML) and wearable medical sensors (WMSs) in edge devices have enabled ML-driven disease detection for smart healthcare. Conventional ML-driven disease detection methods rely on customizing individual models for each disease and its corresponding WMS data. However, such methods lack adaptability to distribution shifts and new task classification classes. Also, they need to be rearchitected and retrained from scratch for each new disease. Moreover, installing multiple ML models in an edge device consumes excessive memory, drains the battery faster, and complicates the detection process. To address these challenges, we propose DOCTOR, a multi-disease detection continual learning (CL) framework based on WMSs. It employs a multi-headed deep neural network (DNN) and an exemplar-replay-style CL algorithm. The CL algorithm enables the framework to continually learn new missions where different data distributions, classification classes, and disease detection
    
[^4]: 重新思考人工智能可解释性与合理性

    Rethinking AI Explainability and Plausibility. (arXiv:2303.17707v1 [cs.AI])

    [http://arxiv.org/abs/2303.17707](http://arxiv.org/abs/2303.17707)

    本文研究了XAI评估中最普遍的人为概念——解释合理性。虽然一直被制定为AI可解释性任务的重要评估目标，但是评估XAI的合理性有时是有害的，且无法达到模型可理解性、透明度和可信度的目的。

    

    为了使可解释人工智能（XAI）算法符合人类交流规范，支持人类推理过程，并满足人类对于AI解释的需求，设定适当的评估目标至关重要。在本文中，我们研究了解释合理性，这是XAI评估中最普遍的人为概念。合理性衡量机器解释与人类解释相比的合理程度。合理性一直被传统地制定为AI可解释性任务的重要评估目标。我们反对这个想法，并展示了如何优化和评估XAI的合理性有时是有害的，且无法达到模型可理解性、透明度和可信度的目的。具体来说，评估XAI算法的合理性会规范机器解释，以表达与人类解释完全相同的内容，这偏离了人类解释的基本动机：表达自己的理解。

    Setting proper evaluation objectives for explainable artificial intelligence (XAI) is vital for making XAI algorithms follow human communication norms, support human reasoning processes, and fulfill human needs for AI explanations. In this article, we examine explanation plausibility, which is the most pervasive human-grounded concept in XAI evaluation. Plausibility measures how reasonable the machine explanation is compared to the human explanation. Plausibility has been conventionally formulated as an important evaluation objective for AI explainability tasks. We argue against this idea, and show how optimizing and evaluating XAI for plausibility is sometimes harmful, and always ineffective to achieve model understandability, transparency, and trustworthiness. Specifically, evaluating XAI algorithms for plausibility regularizes the machine explanation to express exactly the same content as human explanation, which deviates from the fundamental motivation for humans to explain: expres
    

