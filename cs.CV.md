# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Customizing Visual-Language Foundation Models for Multi-modal Anomaly Detection and Reasoning](https://arxiv.org/abs/2403.11083) | 该研究旨在开发一种适用于多种场景的通用异常检测模型，通过定制视觉-语言基础模型和引入多模态提示策略进行多模态异常检测和推理。 |
| [^2] | [RadCLIP: Enhancing Radiologic Image Analysis through Contrastive Language-Image Pre-training](https://arxiv.org/abs/2403.09948) | RadCLIP是一种创新的跨模态基础模型，利用对比语言图像预训练以改进放射学图像分析，包含针对体积图像分析定制的新颖3D切片池化机制，并使用丰富多样的放射学图像-文本对数据集进行训练。 |
| [^3] | [Word4Per: Zero-shot Composed Person Retrieval](https://arxiv.org/abs/2311.16515) | 提出了一个新任务：组合人员检索（CPR），旨在联合利用图像和文本信息进行目标人员检索，引入零样本组合人员检索（ZS-CPR）解决了CPR问题，提出了一个两阶段学习框架Word4Per。 |
| [^4] | [Gradient Leakage Defense with Key-Lock Module for Federated Learning.](http://arxiv.org/abs/2305.04095) | 本研究提出了一种新的联邦学习梯度泄露防御技术，使用私钥锁模块保护任意模型体系结构，并可确保无法从共享的梯度中重建私有训练数据。 |

# 详细

[^1]: 为多模态异常检测和推理定制视觉-语言基础模型

    Customizing Visual-Language Foundation Models for Multi-modal Anomaly Detection and Reasoning

    [https://arxiv.org/abs/2403.11083](https://arxiv.org/abs/2403.11083)

    该研究旨在开发一种适用于多种场景的通用异常检测模型，通过定制视觉-语言基础模型和引入多模态提示策略进行多模态异常检测和推理。

    

    异常检测在各种工业场景中十分重要，包括生产线上异常模式的识别和用于质量控制的制造缺陷检测。本研究旨在开发一种适用于多种场景的通用异常检测模型。为实现这一目标，我们将拥有广泛知识和强大推理能力的通用视觉-语言基础模型定制为异常检测器和推理器。具体来说，我们引入了一种多模态提示策略，将领域专家的领域知识作为条件引导模型。我们的方法考虑多模态提示类型，包括任务描述、类别上下文、正常规则和参考图像。另外，我们将多模态输入表示统一为2D图像格式，使其能够

    arXiv:2403.11083v1 Announce Type: cross  Abstract: Anomaly detection is vital in various industrial scenarios, including the identification of unusual patterns in production lines and the detection of manufacturing defects for quality control. Existing techniques tend to be specialized in individual scenarios and lack generalization capacities. In this study, we aim to develop a generic anomaly detection model applicable across multiple scenarios. To achieve this, we customize generic visual-language foundation models that possess extensive knowledge and robust reasoning abilities into anomaly detectors and reasoners. Specifically, we introduce a multi-modal prompting strategy that incorporates domain knowledge from experts as conditions to guide the models. Our approach considers multi-modal prompt types, including task descriptions, class context, normality rules, and reference images. In addition, we unify the input representation of multi-modality into a 2D image format, enabling m
    
[^2]: RadCLIP: 通过对比语言图像预训练增强放射学图像分析

    RadCLIP: Enhancing Radiologic Image Analysis through Contrastive Language-Image Pre-training

    [https://arxiv.org/abs/2403.09948](https://arxiv.org/abs/2403.09948)

    RadCLIP是一种创新的跨模态基础模型，利用对比语言图像预训练以改进放射学图像分析，包含针对体积图像分析定制的新颖3D切片池化机制，并使用丰富多样的放射学图像-文本对数据集进行训练。

    

    arXiv:2403.09948v1 公告类型: 跨领域  摘要: 人工智能（AI）与放射学的整合标志着医学诊断领域的变革时代。视觉基础模型已被采用来增强放射学图像分析。然而，放射学图像的独特复杂性，包括对2D和3D放射学数据的解读，带来了现有模型无法充分应对的挑战，因为这些模型是在通用非医学图像上训练的。为了弥合这一差距，并充分利用医学成像所需的诊断精度，我们引入了RadCLIP：一种开创性的跨模态基础模型，利用对比语言图像预训练（CLIP）来改进放射学图像分析。RadCLIP包含一种新颖的3D切片池化机制，专为体积图像分析定制，使用了丰富多样的放射学图像-文本对数据集进行训练。我们的评估表明，RadCLIP能有效地对齐放射学图像

    arXiv:2403.09948v1 Announce Type: cross  Abstract: The integration of artificial intelligence (AI) with radiology has marked a transformative era in medical diagnostics. Vision foundation models have been adopted to enhance radiologic imaging analysis. However, the distinct complexities of radiological imaging, including the interpretation of 2D and 3D radiological data, pose unique challenges that existing models, trained on general non-medical images, fail to address adequately. To bridge this gap and capitalize on the diagnostic precision required in medical imaging, we introduce RadCLIP: a pioneering cross-modal foundational model that harnesses Contrastive Language-Image Pre-training (CLIP) to refine radiologic image analysis. RadCLIP incorporates a novel 3D slice pooling mechanism tailored for volumetric image analysis and is trained using a comprehensive and diverse dataset of radiologic image-text pairs. Our evaluations demonstrate that RadCLIP effectively aligns radiological i
    
[^3]: Word4Per: Zero-shot组合人员检索

    Word4Per: Zero-shot Composed Person Retrieval

    [https://arxiv.org/abs/2311.16515](https://arxiv.org/abs/2311.16515)

    提出了一个新任务：组合人员检索（CPR），旨在联合利用图像和文本信息进行目标人员检索，引入零样本组合人员检索（ZS-CPR）解决了CPR问题，提出了一个两阶段学习框架Word4Per。

    

    寻找特定人员具有极大的社会效益和安全价值，通常涉及视觉和文本信息的结合。本文提出了一个全新的任务，称为组合人员检索（CPR），旨在联合利用图像和文本信息进行目标人员检索。然而，监督CPR需要昂贵的手动注释数据集，而目前没有可用资源。为了解决这个问题，我们首先引入了零样本组合人员检索（ZS-CPR），利用现有的领域相关数据解决了CPR问题而不需要昂贵的注释。其次，为了学习ZS-CPR模型，我们提出了一个两阶段学习框架，即Word4Per，其中包含一个轻量级的文本反转网络。

    arXiv:2311.16515v2 Announce Type: replace-cross  Abstract: Searching for specific person has great social benefits and security value, and it often involves a combination of visual and textual information. Conventional person retrieval methods, whether image-based or text-based, usually fall short in effectively harnessing both types of information, leading to the loss of accuracy. In this paper, a whole new task called Composed Person Retrieval (CPR) is proposed to jointly utilize both image and text information for target person retrieval. However, the supervised CPR requires very costly manual annotation dataset, while there are currently no available resources. To mitigate this issue, we firstly introduce the Zero-shot Composed Person Retrieval (ZS-CPR), which leverages existing domain-related data to resolve the CPR problem without expensive annotations. Secondly, to learn ZS-CPR model, we propose a two-stage learning framework, Word4Per, where a lightweight Textual Inversion Netw
    
[^4]: 基于密钥锁模块的联邦学习梯度泄露防御

    Gradient Leakage Defense with Key-Lock Module for Federated Learning. (arXiv:2305.04095v1 [cs.LG])

    [http://arxiv.org/abs/2305.04095](http://arxiv.org/abs/2305.04095)

    本研究提出了一种新的联邦学习梯度泄露防御技术，使用私钥锁模块保护任意模型体系结构，并可确保无法从共享的梯度中重建私有训练数据。

    

    联邦学习是一种广泛采用的隐私保护机器学习方法，其中私有数据保持本地，允许安全计算和本地模型梯度与第三方参数服务器之间的交换。然而，最近的研究发现，通过共享的梯度可能会危及隐私并恢复敏感信息。本研究提供了详细的分析和对梯度泄漏问题的新视角。这些理论工作导致了一种新的梯度泄露防御技术，使用私钥锁模块保护任意模型体系结构。只有锁定的梯度被传输到参数服务器进行全局模型聚合。我们提出的学习方法对梯度泄露攻击具有抵抗力，并且所设计和训练的密钥锁模块可以确保，没有密钥锁模块的私有信息：a) 无法从共享的梯度中重建私有训练数据。

    Federated Learning (FL) is a widely adopted privacy-preserving machine learning approach where private data remains local, enabling secure computations and the exchange of local model gradients between local clients and third-party parameter servers. However, recent findings reveal that privacy may be compromised and sensitive information potentially recovered from shared gradients. In this study, we offer detailed analysis and a novel perspective on understanding the gradient leakage problem. These theoretical works lead to a new gradient leakage defense technique that secures arbitrary model architectures using a private key-lock module. Only the locked gradient is transmitted to the parameter server for global model aggregation. Our proposed learning method is resistant to gradient leakage attacks, and the key-lock module is designed and trained to ensure that, without the private information of the key-lock module: a) reconstructing private training data from the shared gradient is
    

