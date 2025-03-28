# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning](https://arxiv.org/abs/2403.18886) | 提出了一种名为SEMA的新型微调方法，旨在通过自我扩展预训练模型与模块化适配，实现持续学习过程中的最小遗忘，解决先前针对静态模型架构情况下存在的过多参数分配或适应性不足等问题。 |
| [^2] | [Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures.](http://arxiv.org/abs/2307.15220) | 通过观看手术视频讲座，我们提出了一种新方法，SurgVLP，通过利用手术视频讲座中的语音和视觉信息进行多模态表示学习，并解决了手术相关语言挑战。 |

# 详细

[^1]: 使用混合适配器进行预训练模型的自我扩展以实现持续学习

    Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning

    [https://arxiv.org/abs/2403.18886](https://arxiv.org/abs/2403.18886)

    提出了一种名为SEMA的新型微调方法，旨在通过自我扩展预训练模型与模块化适配，实现持续学习过程中的最小遗忘，解决先前针对静态模型架构情况下存在的过多参数分配或适应性不足等问题。

    

    持续学习旨在从连续到达的数据流中学习，最大限度地减少先前学到的知识的遗忘。本文提出了一种名为SEMA的新型微调方法，称为自我扩展预训练模型与模块化适配，自动决定...（摘要未完整）

    arXiv:2403.18886v1 Announce Type: new  Abstract: Continual learning aims to learn from a stream of continuously arriving data with minimum forgetting of previously learned knowledge. While previous works have explored the effectiveness of leveraging the generalizable knowledge from pre-trained models in continual learning, existing parameter-efficient fine-tuning approaches focus on the use of a predetermined or task-wise set of adapters or prompts. However, these approaches still suffer from forgetting due to task interference on jointly used parameters or restricted flexibility. The reliance on a static model architecture may lead to the allocation of excessive parameters that are not essential or, conversely, inadequate adaptation for downstream tasks, given that the scale and distribution of incoming data are unpredictable in continual learning. We propose Self-Expansion of pre-trained models with Modularized Adaptation (SEMA), a novel fine-tuning approach which automatically decid
    
[^2]: 通过观看数百个手术视频讲座学习多模态表示

    Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures. (arXiv:2307.15220v1 [cs.CV])

    [http://arxiv.org/abs/2307.15220](http://arxiv.org/abs/2307.15220)

    通过观看手术视频讲座，我们提出了一种新方法，SurgVLP，通过利用手术视频讲座中的语音和视觉信息进行多模态表示学习，并解决了手术相关语言挑战。

    

    最近在外科计算机视觉应用方面的进展主要依靠完全监督方法，主要使用视觉数据。这些方法依赖于手动注释的手术视频来预测一组固定的对象类别，限制了它们在未见手术程序和后续任务上的通用性。在这项工作中，我们提出了一个观点，即通过开放的手术电子学习平台提供的手术视频讲座可以为多模态表示学习提供有效的监督信号，而无需依赖手动注释。我们通过使用多个互补的自动语音识别系统生成文本转录来解决手术视频讲座中存在的手术相关语言挑战。然后，我们提出了一种新的方法，SurgVLP - 手术视觉语言预训练，用于多模态表示学习。SurgVLP构建了一种新的对比学习目标，将视频剪辑嵌入与相应的文本嵌入对齐。

    Recent advancements in surgical computer vision applications have been driven by fully-supervised methods, primarily using only visual data. These methods rely on manually annotated surgical videos to predict a fixed set of object categories, limiting their generalizability to unseen surgical procedures and downstream tasks. In this work, we put forward the idea that the surgical video lectures available through open surgical e-learning platforms can provide effective supervisory signals for multi-modal representation learning without relying on manual annotations. We address the surgery-specific linguistic challenges present in surgical video lectures by employing multiple complementary automatic speech recognition systems to generate text transcriptions. We then present a novel method, SurgVLP - Surgical Vision Language Pre-training, for multi-modal representation learning. SurgVLP constructs a new contrastive learning objective to align video clip embeddings with the corresponding m
    

