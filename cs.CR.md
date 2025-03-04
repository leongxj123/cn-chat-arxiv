# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimization-based Prompt Injection Attack to LLM-as-a-Judge](https://arxiv.org/abs/2403.17710) | 介绍了一种基于优化的提示注入攻击方法，JudgeDeceiver，针对LLM-as-a-Judge，通过自动化生成对抗序列实现了有针对性和高效的模型评估操控。 |
| [^2] | [Explainable Machine Learning-Based Security and Privacy Protection Framework for Internet of Medical Things Systems](https://arxiv.org/abs/2403.09752) | 该论文提出了面向互联网医疗物联网系统的可解释机器学习安全与隐私保护框架，旨在解决IoMT系统面临的安全挑战，包括数据敏感性、恶意攻击和异常检测。 |
| [^3] | [Syntactic Ghost: An Imperceptible General-purpose Backdoor Attacks on Pre-trained Language Models](https://arxiv.org/abs/2402.18945) | 论文提出了一种名为Syntactic Ghost的新方法，实现了对预训练语言模型进行无感知和通用的后门植入。 |
| [^4] | [SoK: Facial Deepfake Detectors.](http://arxiv.org/abs/2401.04364) | 本文对最新的面部深度伪造检测器进行了全面回顾和分析，提供了对其有效性影响因素的深入见解，并在各种攻击场景中进行了评估。 |
| [^5] | [Secure and Effective Data Appraisal for Machine Learning.](http://arxiv.org/abs/2310.02373) | 本文介绍了一种机密的数据选择和评估方法，通过创新的流程和简化的低维度操作来实现，以保护数据和模型的隐私，并在多个Transformer模型和NLP/CV基准测试中进行了评估。 |
| [^6] | [SecureFalcon: The Next Cyber Reasoning System for Cyber Security.](http://arxiv.org/abs/2307.06616) | SecureFalcon是基于FalconLLM的网络推理系统，通过微调FalconLLM来实现网络安全应用，能够识别C代码样本中的漏洞和非漏洞内容。 |

# 详细

[^1]: 基于优化的对LLM评判系统的提示注入攻击

    Optimization-based Prompt Injection Attack to LLM-as-a-Judge

    [https://arxiv.org/abs/2403.17710](https://arxiv.org/abs/2403.17710)

    介绍了一种基于优化的提示注入攻击方法，JudgeDeceiver，针对LLM-as-a-Judge，通过自动化生成对抗序列实现了有针对性和高效的模型评估操控。

    

    LLM-as-a-Judge 是一种可以使用大型语言模型（LLMs）评估文本信息的新颖解决方案。根据现有研究，LLMs在提供传统人类评估的引人注目替代方面表现出色。然而，这些系统针对提示注入攻击的鲁棒性仍然是一个未解决的问题。在这项工作中，我们引入了JudgeDeceiver，一种针对LLM-as-a-Judge量身定制的基于优化的提示注入攻击。我们的方法制定了一个精确的优化目标，用于攻击LLM-as-a-Judge的决策过程，并利用优化算法高效地自动化生成对抗序列，实现对模型评估的有针对性和有效的操作。与手工制作的提示注入攻击相比，我们的方法表现出卓越的功效，给基于LLM的判断系统当前的安全范式带来了重大挑战。

    arXiv:2403.17710v1 Announce Type: cross  Abstract: LLM-as-a-Judge is a novel solution that can assess textual information with large language models (LLMs). Based on existing research studies, LLMs demonstrate remarkable performance in providing a compelling alternative to traditional human assessment. However, the robustness of these systems against prompt injection attacks remains an open question. In this work, we introduce JudgeDeceiver, a novel optimization-based prompt injection attack tailored to LLM-as-a-Judge. Our method formulates a precise optimization objective for attacking the decision-making process of LLM-as-a-Judge and utilizes an optimization algorithm to efficiently automate the generation of adversarial sequences, achieving targeted and effective manipulation of model evaluations. Compared to handcraft prompt injection attacks, our method demonstrates superior efficacy, posing a significant challenge to the current security paradigms of LLM-based judgment systems. T
    
[^2]: 面向IoMT系统的可解释机器学习安全与隐私保护框架

    Explainable Machine Learning-Based Security and Privacy Protection Framework for Internet of Medical Things Systems

    [https://arxiv.org/abs/2403.09752](https://arxiv.org/abs/2403.09752)

    该论文提出了面向互联网医疗物联网系统的可解释机器学习安全与隐私保护框架，旨在解决IoMT系统面临的安全挑战，包括数据敏感性、恶意攻击和异常检测。

    

    互联网医疗物联网（IoMT）跨越了传统医疗边界，实现了从被动治疗向主动预防的过渡。这种创新方法通过实时健康数据收集实现早期疾病检测和个性化护理，特别在慢性病管理方面，IoMT可以自动化治疗。然而，由于处理数据的敏感性和价值，IoMT面临着严重的安全挑战，这会威胁到其用户的生命，因此吸引了恶意利益。此外，利用无线通信进行数据传输会使医疗数据暴露于被网络犯罪分子截获和篡改的风险之下。此外，由于人为错误、网络干扰或硬件故障，可能会出现异常。在这种背景下，基于机器学习（ML）的异常检测是一个有趣的解决方案，但它再次出现。

    arXiv:2403.09752v1 Announce Type: cross  Abstract: The Internet of Medical Things (IoMT) transcends traditional medical boundaries, enabling a transition from reactive treatment to proactive prevention. This innovative method revolutionizes healthcare by facilitating early disease detection and tailored care, particularly in chronic disease management, where IoMT automates treatments based on real-time health data collection. Nonetheless, its benefits are countered by significant security challenges that endanger the lives of its users due to the sensitivity and value of the processed data, thereby attracting malicious interests. Moreover, the utilization of wireless communication for data transmission exposes medical data to interception and tampering by cybercriminals. Additionally, anomalies may arise due to human errors, network interference, or hardware malfunctions. In this context, anomaly detection based on Machine Learning (ML) is an interesting solution, but it comes up again
    
[^3]: Syntactic Ghost：一种对预训练语言模型进行的无感知通用后门攻击

    Syntactic Ghost: An Imperceptible General-purpose Backdoor Attacks on Pre-trained Language Models

    [https://arxiv.org/abs/2402.18945](https://arxiv.org/abs/2402.18945)

    论文提出了一种名为Syntactic Ghost的新方法，实现了对预训练语言模型进行无感知和通用的后门植入。

    

    预训练语言模型（PLMs）被发现容易受到后门攻击，可以将漏洞转移到各种下游任务中。然而，现有的PLM后门攻击采用明显的触发器，在手动对准的情况下进行，因此在效果、隐匿性和通用性方面无法同时满足期望目标。本文提出了一种新方法，实现了不可见和通用的后门植入，称为Syntactic Ghost（简称为synGhost）。具体来说，该方法敌意地使用具有不同预定义句法结构的毒害样本作为隐蔽触发器，然后将后门植入到预训练表示空间，而不会破坏原始知识。毒害样本的输出表示在特征空间中尽可能均匀地分布，通过对比学习形成广泛的后门。此外，在亮

    arXiv:2402.18945v1 Announce Type: cross  Abstract: Pre-trained language models (PLMs) have been found susceptible to backdoor attacks, which can transfer vulnerabilities to various downstream tasks. However, existing PLM backdoors are conducted with explicit triggers under the manually aligned, thus failing to satisfy expectation goals simultaneously in terms of effectiveness, stealthiness, and universality. In this paper, we propose a novel approach to achieve invisible and general backdoor implantation, called \textbf{Syntactic Ghost} (synGhost for short). Specifically, the method hostilely manipulates poisoned samples with different predefined syntactic structures as stealth triggers and then implants the backdoor to pre-trained representation space without disturbing the primitive knowledge. The output representations of poisoned samples are distributed as uniformly as possible in the feature space via contrastive learning, forming a wide range of backdoors. Additionally, in light 
    
[^4]: SoK：面部深度伪造检测器

    SoK: Facial Deepfake Detectors. (arXiv:2401.04364v1 [cs.CV])

    [http://arxiv.org/abs/2401.04364](http://arxiv.org/abs/2401.04364)

    本文对最新的面部深度伪造检测器进行了全面回顾和分析，提供了对其有效性影响因素的深入见解，并在各种攻击场景中进行了评估。

    

    深度伪造技术迅速成为对社会构成深远和严重威胁的原因之一，主要由于其易于制作和传播。这种情况加速了深度伪造检测技术的发展。然而，许多现有的检测器在验证时 heavily 依赖实验室生成的数据集，这可能无法有效地让它们应对新颖、新兴和实际的深度伪造技术。本文对最新的深度伪造检测器进行广泛全面的回顾和分析，根据几个关键标准对它们进行评估。这些标准将这些检测器分为 4 个高级组别和 13 个细粒度子组别，都遵循一个统一的标准概念框架。这种分类和框架提供了对影响检测器功效的因素的深入和实用的见解。我们对 16 个主要的检测器在各种标准的攻击场景中的普适性进行评估，包括黑盒攻击场景。

    Deepfakes have rapidly emerged as a profound and serious threat to society, primarily due to their ease of creation and dissemination. This situation has triggered an accelerated development of deepfake detection technologies. However, many existing detectors rely heavily on lab-generated datasets for validation, which may not effectively prepare them for novel, emerging, and real-world deepfake techniques. In this paper, we conduct an extensive and comprehensive review and analysis of the latest state-of-the-art deepfake detectors, evaluating them against several critical criteria. These criteria facilitate the categorization of these detectors into 4 high-level groups and 13 fine-grained sub-groups, all aligned with a unified standard conceptual framework. This classification and framework offer deep and practical insights into the factors that affect detector efficacy. We assess the generalizability of 16 leading detectors across various standard attack scenarios, including black-bo
    
[^5]: 机器学习的安全有效数据评估

    Secure and Effective Data Appraisal for Machine Learning. (arXiv:2310.02373v1 [cs.LG])

    [http://arxiv.org/abs/2310.02373](http://arxiv.org/abs/2310.02373)

    本文介绍了一种机密的数据选择和评估方法，通过创新的流程和简化的低维度操作来实现，以保护数据和模型的隐私，并在多个Transformer模型和NLP/CV基准测试中进行了评估。

    

    一个无拘无束的数据市场需要在数据所有者和模型所有者最终交易前能够对训练数据进行私密选择和评估。为了保护数据和模型的隐私，这个过程涉及使用多方计算(MPC)来审查目标模型。尽管之前的研究认为基于MPC的Transformer模型评估过于耗费资源，本文介绍了一种创新的方法，使数据选择成为可行的。本研究的贡献包括三个关键要素：(1)使用MPC进行机密数据选择的开创性流程；(2)通过在有限的相关数据子集上训练简化的低维度MLP来复制复杂的高维度操作；(3)并发、多阶段地实现MPC。所提出的方法在一系列Transformer模型和NLP/CV基准测试中进行了评估。与直接基于MPC的评估相比

    Essential for an unfettered data market is the ability to discreetly select and evaluate training data before finalizing a transaction between the data owner and model owner. To safeguard the privacy of both data and model, this process involves scrutinizing the target model through Multi-Party Computation (MPC). While prior research has posited that the MPC-based evaluation of Transformer models is excessively resource-intensive, this paper introduces an innovative approach that renders data selection practical. The contributions of this study encompass three pivotal elements: (1) a groundbreaking pipeline for confidential data selection using MPC, (2) replicating intricate high-dimensional operations with simplified low-dimensional MLPs trained on a limited subset of pertinent data, and (3) implementing MPC in a concurrent, multi-phase manner. The proposed method is assessed across an array of Transformer models and NLP/CV benchmarks. In comparison to the direct MPC-based evaluation 
    
[^6]: SecureFalcon:下一代面向网络安全的网络推理系统

    SecureFalcon: The Next Cyber Reasoning System for Cyber Security. (arXiv:2307.06616v1 [cs.CR])

    [http://arxiv.org/abs/2307.06616](http://arxiv.org/abs/2307.06616)

    SecureFalcon是基于FalconLLM的网络推理系统，通过微调FalconLLM来实现网络安全应用，能够识别C代码样本中的漏洞和非漏洞内容。

    

    软件漏洞导致各种不利影响，如崩溃、数据丢失和安全漏洞，严重影响软件应用和系统的市场采用率。尽管传统的方法，如自动化软件测试、故障定位和修复已经得到广泛研究，但静态分析工具最常用且有固有的误报率，给开发人员的生产力带来了实质性挑战。大型语言模型（LLM）为这些持久问题提供了有希望的解决方案。其中，FalconLLM在识别复杂模式和漏洞方面显示出重要潜力，因此在软件漏洞检测中至关重要。在本文中，我们首次对FalconLLM进行了针对网络安全应用的微调，从而推出了SecureFalcon，这是基于FalconLLM的创新模型架构。SecureFalcon被训练用于区分有漏洞和无漏洞的C代码样本。

    Software vulnerabilities leading to various detriments such as crashes, data loss, and security breaches, significantly hinder the quality, affecting the market adoption of software applications and systems. Although traditional methods such as automated software testing, fault localization, and repair have been intensively studied, static analysis tools are most commonly used and have an inherent false positives rate, posing a solid challenge to developer productivity. Large Language Models (LLMs) offer a promising solution to these persistent issues. Among these, FalconLLM has shown substantial potential in identifying intricate patterns and complex vulnerabilities, hence crucial in software vulnerability detection. In this paper, for the first time, FalconLLM is being fine-tuned for cybersecurity applications, thus introducing SecureFalcon, an innovative model architecture built upon FalconLLM. SecureFalcon is trained to differentiate between vulnerable and non-vulnerable C code sam
    

