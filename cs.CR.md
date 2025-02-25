# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Institutional Platform for Secure Self-Service Large Language Model Exploration](https://rss.arxiv.org/abs/2402.00913) | 这个论文介绍了一个用户友好型平台，旨在使大型定制语言模型更易于使用，通过最新的多LoRA推理技术和定制适配器，实现了数据隔离、加密和身份验证的安全服务。 |
| [^2] | [Defending Jailbreak Prompts via In-Context Adversarial Game](https://arxiv.org/abs/2402.13148) | 介绍了一种通过上下文对抗游戏(ICAG)防御越狱提示的方法，能够显著降低成功率。 |
| [^3] | [SoK: Facial Deepfake Detectors.](http://arxiv.org/abs/2401.04364) | 本文对最新的面部深度伪造检测器进行了全面回顾和分析，提供了对其有效性影响因素的深入见解，并在各种攻击场景中进行了评估。 |

# 详细

[^1]: 用于安全自助大型语言模型探索的机构平台

    Institutional Platform for Secure Self-Service Large Language Model Exploration

    [https://rss.arxiv.org/abs/2402.00913](https://rss.arxiv.org/abs/2402.00913)

    这个论文介绍了一个用户友好型平台，旨在使大型定制语言模型更易于使用，通过最新的多LoRA推理技术和定制适配器，实现了数据隔离、加密和身份验证的安全服务。

    

    本文介绍了由肯塔基大学应用人工智能中心开发的用户友好型平台，旨在使大型定制语言模型（LLM）更易于使用。通过利用最近在多LoRA推理方面的进展，系统有效地适应了各类用户和项目的定制适配器。论文概述了系统的架构和关键特性，包括数据集策划、模型训练、安全推理和基于文本的特征提取。我们通过使用基于代理的方法建立了一个基于租户意识的计算网络，在安全地利用孤立资源岛的基础上形成了一个统一的系统。该平台致力于提供安全的LLM服务，强调过程和数据隔离、端到端加密以及基于角色的资源身份验证。该贡献与实现简化访问先进的AI模型和技术以支持科学发现的总体目标一致。

    This paper introduces a user-friendly platform developed by the University of Kentucky Center for Applied AI, designed to make large, customized language models (LLMs) more accessible. By capitalizing on recent advancements in multi-LoRA inference, the system efficiently accommodates custom adapters for a diverse range of users and projects. The paper outlines the system's architecture and key features, encompassing dataset curation, model training, secure inference, and text-based feature extraction.   We illustrate the establishment of a tenant-aware computational network using agent-based methods, securely utilizing islands of isolated resources as a unified system. The platform strives to deliver secure LLM services, emphasizing process and data isolation, end-to-end encryption, and role-based resource authentication. This contribution aligns with the overarching goal of enabling simplified access to cutting-edge AI models and technology in support of scientific discovery.
    
[^2]: 通过上下文对抗游戏防御越狱提示

    Defending Jailbreak Prompts via In-Context Adversarial Game

    [https://arxiv.org/abs/2402.13148](https://arxiv.org/abs/2402.13148)

    介绍了一种通过上下文对抗游戏(ICAG)防御越狱提示的方法，能够显著降低成功率。

    

    大语言模型(LLMs)展现出在不同应用领域中的显著能力。然而，对其安全性的担忧，特别是对越狱攻击的脆弱性，仍然存在。受到深度学习中对抗训练和LLM代理学习过程的启发，我们引入了上下文对抗游戏(ICAG)来防御越狱攻击，无需进行微调。ICAG利用代理学习进行对抗游戏，旨在动态扩展知识以防御越狱攻击。与依赖静态数据集的传统方法不同，ICAG采用迭代过程来增强防御和攻击代理。这一持续改进过程加强了对新生成的越狱提示的防御。我们的实证研究证实了ICAG的有效性，经由ICAG保护的LLMs在各种攻击场景中显著降低了越狱成功率。

    arXiv:2402.13148v1 Announce Type: new  Abstract: Large Language Models (LLMs) demonstrate remarkable capabilities across diverse applications. However, concerns regarding their security, particularly the vulnerability to jailbreak attacks, persist. Drawing inspiration from adversarial training in deep learning and LLM agent learning processes, we introduce the In-Context Adversarial Game (ICAG) for defending against jailbreaks without the need for fine-tuning. ICAG leverages agent learning to conduct an adversarial game, aiming to dynamically extend knowledge to defend against jailbreaks. Unlike traditional methods that rely on static datasets, ICAG employs an iterative process to enhance both the defense and attack agents. This continuous improvement process strengthens defenses against newly generated jailbreak prompts. Our empirical studies affirm ICAG's efficacy, where LLMs safeguarded by ICAG exhibit significantly reduced jailbreak success rates across various attack scenarios. Mo
    
[^3]: SoK：面部深度伪造检测器

    SoK: Facial Deepfake Detectors. (arXiv:2401.04364v1 [cs.CV])

    [http://arxiv.org/abs/2401.04364](http://arxiv.org/abs/2401.04364)

    本文对最新的面部深度伪造检测器进行了全面回顾和分析，提供了对其有效性影响因素的深入见解，并在各种攻击场景中进行了评估。

    

    深度伪造技术迅速成为对社会构成深远和严重威胁的原因之一，主要由于其易于制作和传播。这种情况加速了深度伪造检测技术的发展。然而，许多现有的检测器在验证时 heavily 依赖实验室生成的数据集，这可能无法有效地让它们应对新颖、新兴和实际的深度伪造技术。本文对最新的深度伪造检测器进行广泛全面的回顾和分析，根据几个关键标准对它们进行评估。这些标准将这些检测器分为 4 个高级组别和 13 个细粒度子组别，都遵循一个统一的标准概念框架。这种分类和框架提供了对影响检测器功效的因素的深入和实用的见解。我们对 16 个主要的检测器在各种标准的攻击场景中的普适性进行评估，包括黑盒攻击场景。

    Deepfakes have rapidly emerged as a profound and serious threat to society, primarily due to their ease of creation and dissemination. This situation has triggered an accelerated development of deepfake detection technologies. However, many existing detectors rely heavily on lab-generated datasets for validation, which may not effectively prepare them for novel, emerging, and real-world deepfake techniques. In this paper, we conduct an extensive and comprehensive review and analysis of the latest state-of-the-art deepfake detectors, evaluating them against several critical criteria. These criteria facilitate the categorization of these detectors into 4 high-level groups and 13 fine-grained sub-groups, all aligned with a unified standard conceptual framework. This classification and framework offer deep and practical insights into the factors that affect detector efficacy. We assess the generalizability of 16 leading detectors across various standard attack scenarios, including black-bo
    

