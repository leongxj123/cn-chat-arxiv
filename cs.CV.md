# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Customizing Visual-Language Foundation Models for Multi-modal Anomaly Detection and Reasoning](https://arxiv.org/abs/2403.11083) | 该研究旨在开发一种适用于多种场景的通用异常检测模型，通过定制视觉-语言基础模型和引入多模态提示策略进行多模态异常检测和推理。 |
| [^2] | [FreeA: Human-object Interaction Detection using Free Annotation Labels](https://arxiv.org/abs/2403.01840) | 提出了一种新颖的自适应语言驱动的HOI检测方法FreeA，无需标记，利用了CLIP来生成潜在的HOI标签，并在两个基准数据集上取得了最先进的性能。 |

# 详细

[^1]: 为多模态异常检测和推理定制视觉-语言基础模型

    Customizing Visual-Language Foundation Models for Multi-modal Anomaly Detection and Reasoning

    [https://arxiv.org/abs/2403.11083](https://arxiv.org/abs/2403.11083)

    该研究旨在开发一种适用于多种场景的通用异常检测模型，通过定制视觉-语言基础模型和引入多模态提示策略进行多模态异常检测和推理。

    

    异常检测在各种工业场景中十分重要，包括生产线上异常模式的识别和用于质量控制的制造缺陷检测。本研究旨在开发一种适用于多种场景的通用异常检测模型。为实现这一目标，我们将拥有广泛知识和强大推理能力的通用视觉-语言基础模型定制为异常检测器和推理器。具体来说，我们引入了一种多模态提示策略，将领域专家的领域知识作为条件引导模型。我们的方法考虑多模态提示类型，包括任务描述、类别上下文、正常规则和参考图像。另外，我们将多模态输入表示统一为2D图像格式，使其能够

    arXiv:2403.11083v1 Announce Type: cross  Abstract: Anomaly detection is vital in various industrial scenarios, including the identification of unusual patterns in production lines and the detection of manufacturing defects for quality control. Existing techniques tend to be specialized in individual scenarios and lack generalization capacities. In this study, we aim to develop a generic anomaly detection model applicable across multiple scenarios. To achieve this, we customize generic visual-language foundation models that possess extensive knowledge and robust reasoning abilities into anomaly detectors and reasoners. Specifically, we introduce a multi-modal prompting strategy that incorporates domain knowledge from experts as conditions to guide the models. Our approach considers multi-modal prompt types, including task descriptions, class context, normality rules, and reference images. In addition, we unify the input representation of multi-modality into a 2D image format, enabling m
    
[^2]: 使用自由注释标签进行人-物互动检测的FreeA方法

    FreeA: Human-object Interaction Detection using Free Annotation Labels

    [https://arxiv.org/abs/2403.01840](https://arxiv.org/abs/2403.01840)

    提出了一种新颖的自适应语言驱动的HOI检测方法FreeA，无需标记，利用了CLIP来生成潜在的HOI标签，并在两个基准数据集上取得了最先进的性能。

    

    最近的人-物互动（HOI）检测方法依赖于劳动力成本高昂，并需要全面注释的图像数据集。本文提出了一种新颖的自适应语言驱动的HOI检测方法FreeA，这种方法利用了CLIP的适应性来生成潜在的HOI标签，无需标记。具体而言，FreeA将人-物对的图像特征与HOI文本模板进行匹配，并开发了基于先验知识的掩模方法来抑制不太可能的交互作用。此外，FreeA利用了提出的交互相关性匹配方法来增强与指定动作相关的动作的可能性，进一步完善生成的HOI标签。在两个基准数据集上的实验证明，FreeA在弱监督HOI模型中实现了最先进的性能。我们的方法在HICO-DET上的平均精度（mAP）提高了+8.58，在V-COCO上提高了+1.23。

    arXiv:2403.01840v1 Announce Type: cross  Abstract: Recent human-object interaction (HOI) detection approaches rely on high cost of manpower and require comprehensive annotated image datasets. In this paper, we propose a novel self-adaption language-driven HOI detection method, termed as FreeA, without labeling by leveraging the adaptability of CLIP to generate latent HOI labels. To be specific, FreeA matches image features of human-object pairs with HOI text templates, and a priori knowledge-based mask method is developed to suppress improbable interactions. In addition, FreeA utilizes the proposed interaction correlation matching method to enhance the likelihood of actions related to a specified action, further refine the generated HOI labels. Experiments on two benchmark datasets show that FreeA achieves state-of-the-art performance among weakly supervised HOI models. Our approach is +8.58 mean Average Precision (mAP) on HICO-DET and +1.23 mAP on V-COCO more accurate in localizing an
    

