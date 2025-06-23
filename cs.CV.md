# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Hybrid SNN-ANN Network for Event-based Object Detection with Spatial and Temporal Attention](https://arxiv.org/abs/2403.10173) | 提出了一种用于基于事件的对象检测的混合SNN-ANN网络，包括了新颖的基于注意力的桥接模块，能够有效捕捉稀疏的空间和时间关系，以提高任务性能。 |
| [^2] | [ICC: Quantifying Image Caption Concreteness for Multimodal Dataset Curation](https://arxiv.org/abs/2403.01306) | 提出一种新的度量标准，图像描述具体性，用于评估标题文本的具体性和相关性，以帮助在多模态学习中隔离提供最强信号的最具体样本。 |

# 详细

[^1]: 一种用于基于事件的对象检测的混合SNN-ANN网络，具有空间和时间注意力机制

    A Hybrid SNN-ANN Network for Event-based Object Detection with Spatial and Temporal Attention

    [https://arxiv.org/abs/2403.10173](https://arxiv.org/abs/2403.10173)

    提出了一种用于基于事件的对象检测的混合SNN-ANN网络，包括了新颖的基于注意力的桥接模块，能够有效捕捉稀疏的空间和时间关系，以提高任务性能。

    

    事件相机提供高时间分辨率和动态范围，几乎没有运动模糊，非常适合对象检测任务。尖峰神经网络（SNN）与事件驱动感知数据天生匹配，在神经形态硬件上能够实现超低功耗和低延迟推断，而人工神经网络（ANN）则展示出更稳定的训练动态和更快的收敛速度，从而具有更好的任务性能。混合SNN-ANN方法是一种有前途的替代方案，能够利用SNN和ANN体系结构的优势。在这项工作中，我们引入了第一个基于混合注意力的SNN-ANN骨干网络，用于使用事件相机进行对象检测。我们提出了一种新颖的基于注意力的SNN-ANN桥接模块，从SNN层中捕捉稀疏的空间和时间关系，并将其转换为密集特征图，供骨干网络的ANN部分使用。实验结果表明，我们提出的m

    arXiv:2403.10173v1 Announce Type: cross  Abstract: Event cameras offer high temporal resolution and dynamic range with minimal motion blur, making them promising for object detection tasks. While Spiking Neural Networks (SNNs) are a natural match for event-based sensory data and enable ultra-energy efficient and low latency inference on neuromorphic hardware, Artificial Neural Networks (ANNs) tend to display more stable training dynamics and faster convergence resulting in greater task performance. Hybrid SNN-ANN approaches are a promising alternative, enabling to leverage the strengths of both SNN and ANN architectures. In this work, we introduce the first Hybrid Attention-based SNN-ANN backbone for object detection using event cameras. We propose a novel Attention-based SNN-ANN bridge module to capture sparse spatial and temporal relations from the SNN layer and convert them into dense feature maps for the ANN part of the backbone. Experimental results demonstrate that our proposed m
    
[^2]: ICC：用于多模态数据集筛选的图像描述具体性量化

    ICC: Quantifying Image Caption Concreteness for Multimodal Dataset Curation

    [https://arxiv.org/abs/2403.01306](https://arxiv.org/abs/2403.01306)

    提出一种新的度量标准，图像描述具体性，用于评估标题文本的具体性和相关性，以帮助在多模态学习中隔离提供最强信号的最具体样本。

    

    arXiv:2403.01306v1 公告类型：新摘要：针对配对文本-图像数据的Web规模训练在多模态学习中变得越来越重要，但挑战在野外数据集的高噪声特性。标准数据过滤方法成功去除了不匹配的文本-图像对，但允许语义相关但非常抽象或主观的文本。这些方法缺乏细粒度的能力来隔离提供在嘈杂数据集中学习最强信号的最具体样本。在这项工作中，我们提出了一种新的度量标准，图像描述具体性，评估没有图像参考的标题文本以衡量其具体性和相关性，以供在多模态学习中使用。我们的方法利用了衡量视觉-语义信息损失的强基础模型来进行评估。我们证明了这与人类对单词和句子级文本具体性的评估高度相关。此外，我们展示了...

    arXiv:2403.01306v1 Announce Type: new  Abstract: Web-scale training on paired text-image data is becoming increasingly central to multimodal learning, but is challenged by the highly noisy nature of datasets in the wild. Standard data filtering approaches succeed in removing mismatched text-image pairs, but permit semantically related but highly abstract or subjective text. These approaches lack the fine-grained ability to isolate the most concrete samples that provide the strongest signal for learning in a noisy dataset. In this work, we propose a new metric, image caption concreteness, that evaluates caption text without an image reference to measure its concreteness and relevancy for use in multimodal learning. Our approach leverages strong foundation models for measuring visual-semantic information loss in multimodal representations. We demonstrate that this strongly correlates with human evaluation of concreteness in both single-word and sentence-level texts. Moreover, we show tha
    

