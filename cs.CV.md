# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Griffon v2: Advancing Multimodal Perception with High-Resolution Scaling and Visual-Language Co-Referring](https://arxiv.org/abs/2403.09333) | Griffon v2通过引入高分辨率缩放和视觉-语言共指，提升了多模态感知能力，尤其是对于小对象的感知能力。 |
| [^2] | [SARDet-100K: Towards Open-Source Benchmark and ToolKit for Large-Scale SAR Object Detection](https://arxiv.org/abs/2403.06534) | SARDet-100K是第一个COCO级别的大规模多类别SAR物体检测数据集，为研究提供了大规模且多样化的数据集，揭示了SAR物体检测中预训练模型显著差异的关键挑战。 |
| [^3] | [Codebook-enabled Generative End-to-end Semantic Communication Powered by Transformer](https://arxiv.org/abs/2402.16868) | 本文提出了一个强大的码书辅助图像语义通信系统，通过联合构建语义编解码器和码书、设计向量-索引变换器来实现图像生成，并且借助高质量码书帮助Transformer，提高系统对抗信道噪声的鲁棒性。 |

# 详细

[^1]: Griffon v2：通过高分辨率缩放和视觉-语言共指推进多模态感知

    Griffon v2: Advancing Multimodal Perception with High-Resolution Scaling and Visual-Language Co-Referring

    [https://arxiv.org/abs/2403.09333](https://arxiv.org/abs/2403.09333)

    Griffon v2通过引入高分辨率缩放和视觉-语言共指，提升了多模态感知能力，尤其是对于小对象的感知能力。

    

    大型视觉语言模型已经实现了细粒度对象感知，但图像分辨率的限制仍然是超越复杂和密集场景中特定任务专家表现的重要障碍。为了解决这一问题，我们引入了一个统一的高分辨率通用模型，Griffon v2，实现了具有视觉和文本提示的灵活对象引用。为了有效提高图像分辨率，我们设计了一个简单轻量级的下采样投影仪，以克服大型语言模型中输入标记的约束。这种设计固有地保留了完整的上下文和细节，并显著提高了多模态感知能力，特别是对于小对象。在此基础上，我们进一步为模型配置了视觉-语言共指。

    arXiv:2403.09333v1 Announce Type: cross  Abstract: Large Vision Language Models have achieved fine-grained object perception, but the limitation of image resolution remains a significant obstacle to surpass the performance of task-specific experts in complex and dense scenarios. Such limitation further restricts the model's potential to achieve nuanced visual and language referring in domains such as GUI Agents, Counting and \etc. To address this issue, we introduce a unified high-resolution generalist model, Griffon v2, enabling flexible object referring with visual and textual prompts. To efficiently scaling up image resolution, we design a simple and lightweight down-sampling projector to overcome the input tokens constraint in Large Language Models. This design inherently preserves the complete contexts and fine details, and significantly improves multimodal perception ability especially for small objects. Building upon this, we further equip the model with visual-language co-refer
    
[^2]: SARDet-100K: 面向大规模合成孔径雷达 SAR 物体检测的开源基准和工具包

    SARDet-100K: Towards Open-Source Benchmark and ToolKit for Large-Scale SAR Object Detection

    [https://arxiv.org/abs/2403.06534](https://arxiv.org/abs/2403.06534)

    SARDet-100K是第一个COCO级别的大规模多类别SAR物体检测数据集，为研究提供了大规模且多样化的数据集，揭示了SAR物体检测中预训练模型显著差异的关键挑战。

    

    面向合成孔径雷达（SAR）物体检测近来备受关注，因其不可替代的全天候成像能力。然而，这一研究领域面临着有限的公共数据集（主要包含 <2K 张图像，且仅包含单类别物体）和源代码不可访问的挑战。为解决这些问题，我们建立了一个新的基准数据集和一个针对大规模 SAR 物体检测的开源方法。我们的数据集 SARDet-100K 结果是对 10 个现有 SAR 检测数据集进行深入调研、收集和标准化的产物，为研究提供了一个大规模且多样化的数据集。据我们所知，SARDet-100K 是有史以来第一个达到 COCO 水平的大规模多类别 SAR 物体检测数据集。凭借这一高质量数据集，我们进行了全面实验，并揭示了 SAR 物体检测中一个关键挑战：预训练模型的显著差异。

    arXiv:2403.06534v1 Announce Type: cross  Abstract: Synthetic Aperture Radar (SAR) object detection has gained significant attention recently due to its irreplaceable all-weather imaging capabilities. However, this research field suffers from both limited public datasets (mostly comprising <2K images with only mono-category objects) and inaccessible source code. To tackle these challenges, we establish a new benchmark dataset and an open-source method for large-scale SAR object detection. Our dataset, SARDet-100K, is a result of intense surveying, collecting, and standardizing 10 existing SAR detection datasets, providing a large-scale and diverse dataset for research purposes. To the best of our knowledge, SARDet-100K is the first COCO-level large-scale multi-class SAR object detection dataset ever created. With this high-quality dataset, we conducted comprehensive experiments and uncovered a crucial challenge in SAR object detection: the substantial disparities between the pretraining
    
[^3]: 由Transformer驱动的端到端语义通信的码书生成方法

    Codebook-enabled Generative End-to-end Semantic Communication Powered by Transformer

    [https://arxiv.org/abs/2402.16868](https://arxiv.org/abs/2402.16868)

    本文提出了一个强大的码书辅助图像语义通信系统，通过联合构建语义编解码器和码书、设计向量-索引变换器来实现图像生成，并且借助高质量码书帮助Transformer，提高系统对抗信道噪声的鲁棒性。

    

    基于码书的生成式语义通信引起了越来越多的关注，因为当码书在发送者和接收者之间共享时，只需要传输索引。然而，由于码向量之间的语义关系未必与对应码索引的距离相关，码书启用的语义通信系统性能容易受到信道噪声的影响。因此，如何提高系统对抗噪声的鲁棒性需要仔细设计。本文提出了一个强大的码书辅助图像语义通信系统，其中首先联合构建语义编解码器和码书，然后设计了向量-索引变换器，根据码书引导以消除信道噪声的影响，并实现图像生成。由于高质量码书对Transformer的辅助，接收端生成的图像效果优于...

    arXiv:2402.16868v1 Announce Type: cross  Abstract: Codebook-based generative semantic communication attracts increasing attention, since only indices are required to be transmitted when the codebook is shared between transmitter and receiver. However, due to the fact that the semantic relations among code vectors are not necessarily related to the distance of the corresponding code indices, the performance of the codebook-enabled semantic communication system is susceptible to the channel noise. Thus, how to improve the system robustness against the noise requires careful design. This paper proposes a robust codebook-assisted image semantic communication system, where semantic codec and codebook are first jointly constructed, and then vector-to-index transformer is designed guided by the codebook to eliminate the effects of channel noise, and achieve image generation. Thanks to the assistance of the high-quality codebook to the Transformer, the generated images at the receiver outperfo
    

