# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [IoTCO2: Assessing the End-To-End Carbon Footprint of Internet-of-Things-Enabled Deep Learning](https://arxiv.org/abs/2403.10984) | 介绍了一种名为\carb 的端到端建模工具，用于在物联网-启用深度学习中精确估算碳足迹，展示了与实际测量值相比最大$\pm21\%$的碳足迹差异。 |
| [^2] | [Distribution-Free Fair Federated Learning with Small Samples](https://arxiv.org/abs/2402.16158) | 本文介绍了一种用于分布无关公平学习的后处理算法FedFaiREE，适用于去中心化具有小样本的环境。 |
| [^3] | [Generative Ghosts: Anticipating Benefits and Risks of AI Afterlives](https://arxiv.org/abs/2402.01662) | 本文讨论了生成幽灵的潜在实施设计空间和其对个人和社会的实际和伦理影响，提出了研究议程以便使人们能够安全而有益地创建和与人工智能来世进行互动。 |

# 详细

[^1]: IoTCO2：评估物联网-启用深度学习的端到端碳足迹

    IoTCO2: Assessing the End-To-End Carbon Footprint of Internet-of-Things-Enabled Deep Learning

    [https://arxiv.org/abs/2403.10984](https://arxiv.org/abs/2403.10984)

    介绍了一种名为\carb 的端到端建模工具，用于在物联网-启用深度学习中精确估算碳足迹，展示了与实际测量值相比最大$\pm21\%$的碳足迹差异。

    

    为了提高隐私性和确保服务质量（QoS），深度学习（DL）模型越来越多地部署在物联网（IoT）设备上进行数据处理，极大地增加了与IoT上DL相关的碳足迹，涵盖了操作和实体方面。现有的操作能量预测器经常忽略了量化的DL模型和新兴的神经处理单元（NPUs），而实体碳足迹建模工具忽略了IoT设备中常见的非计算硬件组件，导致了物联网DL准确碳足迹建模工具的差距。本文介绍了\textit{\carb}，一种用于精确估算物联网DL中碳足迹的端到端建模工具，展示了与各种DL模型的实际测量值相比最大$\pm21\%$的碳足迹差异。此外，\carb的实际应用通过多个用户案例展示。

    arXiv:2403.10984v1 Announce Type: cross  Abstract: To improve privacy and ensure quality-of-service (QoS), deep learning (DL) models are increasingly deployed on Internet of Things (IoT) devices for data processing, significantly increasing the carbon footprint associated with DL on IoT, covering both operational and embodied aspects. Existing operational energy predictors often overlook quantized DL models and emerging neural processing units (NPUs), while embodied carbon footprint modeling tools neglect non-computing hardware components common in IoT devices, creating a gap in accurate carbon footprint modeling tools for IoT-enabled DL. This paper introduces \textit{\carb}, an end-to-end modeling tool for precise carbon footprint estimation in IoT-enabled DL, demonstrating a maximum $\pm21\%$ deviation in carbon footprint values compared to actual measurements across various DL models. Additionally, practical applications of \carb are showcased through multiple user case studies.
    
[^2]: 分布无关公平联邦学习与小样本

    Distribution-Free Fair Federated Learning with Small Samples

    [https://arxiv.org/abs/2402.16158](https://arxiv.org/abs/2402.16158)

    本文介绍了一种用于分布无关公平学习的后处理算法FedFaiREE，适用于去中心化具有小样本的环境。

    

    随着联邦学习在实际应用中变得越来越重要，因为它具有去中心化数据训练的能力，解决跨群体的公平性问题变得至关重要。然而，大多数现有的用于确保公平性的机器学习算法是为集中化数据环境设计的，通常需要大样本和分布假设，强调了迫切需要针对具有有限样本和分布无关保证的去中心化和异构系统进行公平性技术的调整。为了解决这个问题，本文介绍了FedFaiREE，这是一种专门用于去中心化环境中小样本的分布无关公平学习的后处理算法。我们的方法考虑到了去中心化环境中的独特挑战，例如客户异质性、通信成本和小样本大小。我们为bot提供严格的理论保证

    arXiv:2402.16158v1 Announce Type: cross  Abstract: As federated learning gains increasing importance in real-world applications due to its capacity for decentralized data training, addressing fairness concerns across demographic groups becomes critically important. However, most existing machine learning algorithms for ensuring fairness are designed for centralized data environments and generally require large-sample and distributional assumptions, underscoring the urgent need for fairness techniques adapted for decentralized and heterogeneous systems with finite-sample and distribution-free guarantees. To address this issue, this paper introduces FedFaiREE, a post-processing algorithm developed specifically for distribution-free fair learning in decentralized settings with small samples. Our approach accounts for unique challenges in decentralized environments, such as client heterogeneity, communication costs, and small sample sizes. We provide rigorous theoretical guarantees for bot
    
[^3]: 生成幽灵：预测人工智能来世的益处和风险

    Generative Ghosts: Anticipating Benefits and Risks of AI Afterlives

    [https://arxiv.org/abs/2402.01662](https://arxiv.org/abs/2402.01662)

    本文讨论了生成幽灵的潜在实施设计空间和其对个人和社会的实际和伦理影响，提出了研究议程以便使人们能够安全而有益地创建和与人工智能来世进行互动。

    

    随着人工智能系统在性能的广度和深度上迅速提升，它们越来越适合创建功能强大、逼真的代理人，包括基于特定人物建模的代理人的可能性。我们预计，在我们有生之年，人们可能会普遍使用定制的人工智能代理人与爱的人和/或更广大的世界进行互动。我们称之为生成幽灵，因为这些代理人将能够生成新颖的内容，而不只是复述其创作者在生前的内容。在本文中，我们首先讨论了生成幽灵潜在实施的设计空间。然后，我们讨论了生成幽灵的实际和伦理影响，包括对个人和社会的潜在积极和消极影响。基于这些考虑，我们制定了一个研究议程，旨在使人们能够安全而有益地创建和与人工智能来世进行互动。

    As AI systems quickly improve in both breadth and depth of performance, they lend themselves to creating increasingly powerful and realistic agents, including the possibility of agents modeled on specific people. We anticipate that within our lifetimes it may become common practice for people to create a custom AI agent to interact with loved ones and/or the broader world after death. We call these generative ghosts, since such agents will be capable of generating novel content rather than merely parroting content produced by their creator while living. In this paper, we first discuss the design space of potential implementations of generative ghosts. We then discuss the practical and ethical implications of generative ghosts, including potential positive and negative impacts on individuals and society. Based on these considerations, we lay out a research agenda for the AI and HCI research communities to empower people to create and interact with AI afterlives in a safe and beneficial 
    

