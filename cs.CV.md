# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Severity Controlled Text-to-Image Generative Model Bias Manipulation](https://arxiv.org/abs/2404.02530) | 本文揭示了文本到图像生成模型对偏见操纵的敏感性，并提出了一种通过定量控制模型偏见来操纵输出严重性的技术，从而实现精确提示工程生成新颖图像的方法。 |
| [^2] | [Counterfactual contrastive learning: robust representations via causal image synthesis](https://arxiv.org/abs/2403.09605) | 本研究提出了CF-SimCLR，一种反事实对照学习方法，利用近似反事实推断创造正样本，大大提高了模型对采集偏移的稳健性，并在多个数据集上取得了较高的下游性能。 |
| [^3] | [Robust Multimodal Learning with Missing Modalities via Parameter-Efficient Adaptation.](http://arxiv.org/abs/2310.03986) | 通过低秩适应和中间特征的调制，我们提出了针对预训练多模态网络的参数高效适应程序，以实现对缺失模态的鲁棒性，并在某些情况下胜过独立的专门网络。 |
| [^4] | [Bengali Document Layout Analysis -- A YOLOV8 Based Ensembling Approach.](http://arxiv.org/abs/2309.00848) | 本文提出了一种基于YOLOv8模型和创新的后处理技术的孟加拉文档布局分析方法，通过数据增强和两阶段预测策略实现了准确的元素分割。该方法优于单个基础架构，并解决了BaDLAD数据集中的问题，有助于提高OCR和文档理解能力。 |
| [^5] | [Sparkles: Unlocking Chats Across Multiple Images for Multimodal Instruction-Following Models.](http://arxiv.org/abs/2308.16463) | Sparkles是一个多模态指令跟踪模型，通过整合文本和图像实现多图对话。我们引入了SparklesDialogue数据集和SparklesEval基准来支持训练和评估。实验证实了SparklesChat在理解多图对话方面的有效性。 |
| [^6] | [Overcoming the Stability Gap in Continual Learning.](http://arxiv.org/abs/2306.01904) | 本论文研究了如何克服连续学习中的稳定性差距，并通过发现一种显著减少这种差距的方法，在大规模类别增量学习实验中大幅减少了网络更新的次数。 |

# 详细

[^1]: 严重控制的文本到图像生成模型偏见操纵

    Severity Controlled Text-to-Image Generative Model Bias Manipulation

    [https://arxiv.org/abs/2404.02530](https://arxiv.org/abs/2404.02530)

    本文揭示了文本到图像生成模型对偏见操纵的敏感性，并提出了一种通过定量控制模型偏见来操纵输出严重性的技术，从而实现精确提示工程生成新颖图像的方法。

    

    文本到图像（T2I）生成模型正在广泛流行，尤其是在公共领域。然而，它们固有的偏见和潜在的恶意操纵还未被充分探讨。本文揭示了T2I模型对此类操纵的易感性，并首次提出了通过针对嵌入式语言模型动态且高效地利用模型偏见的新可能性。通过利用向量代数的数学基础，我们的技术实现了对模型偏见通过严重性的输出操纵的可扩展和方便控制。作为副产品，该控制还允许一种精确的提示工程，以生成通常不太可能通过常规文本提示生成的图像。我们还展示了我们的操纵技术在平衡生成类别频率方面的建设应用 - 如在模型去偏。我们的技术不需要训练，并且也以后门的形式构建。

    arXiv:2404.02530v1 Announce Type: cross  Abstract: Text-to-image (T2I) generative models are gaining wide popularity, especially in public domains. However, their intrinsic bias and potential malicious manipulations remain under-explored. Charting the susceptibility of T2I models to such manipulation, we first expose the new possibility of a dynamic and computationally efficient exploitation of model bias by targeting the embedded language models. By leveraging mathematical foundations of vector algebra, our technique enables a scalable and convenient control over the severity of output manipulation through model bias. As a by-product, this control also allows a form of precise prompt engineering to generate images which are generally implausible with regular text prompts. We also demonstrate a constructive application of our manipulation for balancing the frequency of generated classes - as in model debiasing. Our technique does not require training and is also framed as a backdoor at
    
[^2]: 反事实对照学习：通过因果图像合成获得稳健表示

    Counterfactual contrastive learning: robust representations via causal image synthesis

    [https://arxiv.org/abs/2403.09605](https://arxiv.org/abs/2403.09605)

    本研究提出了CF-SimCLR，一种反事实对照学习方法，利用近似反事实推断创造正样本，大大提高了模型对采集偏移的稳健性，并在多个数据集上取得了较高的下游性能。

    

    对比预训练已被广泛认为能够提高下游任务性能和模型泛化能力，特别是在有限标签设置中。然而，它对增强管道的选择敏感。正样本应保留语义信息同时破坏域特定信息。标准增强管道通过预定义的光度变换模拟域特定变化，但如果我们能够模拟真实的领域变化呢？在这项工作中，我们展示了如何利用最近在反事实图像生成方面的进展来实现这一目的。我们提出了CF-SimCLR，一种反事实对照学习方法，它利用近似反事实推断进行正样本创建。对胸部X光和乳腺X光等五个数据集的全面评估表明，CF-SimCLR显著提高了对获取偏移的稳健性，在两种数据集上的下游性能更好。

    arXiv:2403.09605v1 Announce Type: cross  Abstract: Contrastive pretraining is well-known to improve downstream task performance and model generalisation, especially in limited label settings. However, it is sensitive to the choice of augmentation pipeline. Positive pairs should preserve semantic information while destroying domain-specific information. Standard augmentation pipelines emulate domain-specific changes with pre-defined photometric transformations, but what if we could simulate realistic domain changes instead? In this work, we show how to utilise recent progress in counterfactual image generation to this effect. We propose CF-SimCLR, a counterfactual contrastive learning approach which leverages approximate counterfactual inference for positive pair creation. Comprehensive evaluation across five datasets, on chest radiography and mammography, demonstrates that CF-SimCLR substantially improves robustness to acquisition shift with higher downstream performance on both in- an
    
[^3]: 通过参数高效适应，实现对缺失模态的鲁棒多模态学习

    Robust Multimodal Learning with Missing Modalities via Parameter-Efficient Adaptation. (arXiv:2310.03986v1 [cs.CV])

    [http://arxiv.org/abs/2310.03986](http://arxiv.org/abs/2310.03986)

    通过低秩适应和中间特征的调制，我们提出了针对预训练多模态网络的参数高效适应程序，以实现对缺失模态的鲁棒性，并在某些情况下胜过独立的专门网络。

    

    多模态学习旨在利用多个数据源来提高下游任务的整体性能。在一些相关的模态中观察到，如果在测试时间缺少一个或多个模态，现有的多模态网络的性能会显著下降。为了实现对缺失模态的鲁棒性，我们提出了预训练的多模态网络的简单和参数高效的适应程序。特别地，我们利用低秩适应和中间特征的调制来补偿缺失的模态。我们证明，这种适应可以部分弥补由于缺失模态而导致的性能下降，并在某些情况下胜过针对可用模态组合进行训练的独立的、专门的网络。所提出的适应所需的参数非常少（例如，少于）

    Multimodal learning seeks to utilize data from multiple sources to improve the overall performance of downstream tasks. It is desirable for redundancies in the data to make multimodal systems robust to missing or corrupted observations in some correlated modalities. However, we observe that the performance of several existing multimodal networks significantly deteriorates if one or multiple modalities are absent at test time. To enable robustness to missing modalities, we propose simple and parameter-efficient adaptation procedures for pretrained multimodal networks. In particular, we exploit low-rank adaptation and modulation of intermediate features to compensate for the missing modalities. We demonstrate that such adaptation can partially bridge performance drop due to missing modalities and outperform independent, dedicated networks trained for the available modality combinations in some cases. The proposed adaptation requires extremely small number of parameters (e.g., fewer than 
    
[^4]: 孟加拉文档布局分析-一种基于YOLOv8的集成方法

    Bengali Document Layout Analysis -- A YOLOV8 Based Ensembling Approach. (arXiv:2309.00848v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2309.00848](http://arxiv.org/abs/2309.00848)

    本文提出了一种基于YOLOv8模型和创新的后处理技术的孟加拉文档布局分析方法，通过数据增强和两阶段预测策略实现了准确的元素分割。该方法优于单个基础架构，并解决了BaDLAD数据集中的问题，有助于提高OCR和文档理解能力。

    

    本文侧重于利用YOLOv8模型和创新的后处理技术提升孟加拉文档布局分析（DLA）。我们通过数据增强以应对孟加拉复杂文字独特的挑战，经过严格的验证集评估，对完整数据集进行微调，实现准确的元素分割的两阶段预测策略。我们的集成模型结合后处理性能优于单个基础架构，解决了BaDLAD数据集中的问题。通过利用这种方法，我们旨在推动孟加拉文档分析的发展，提高OCR和文档理解能力，同时BaDLAD作为基础资源有助于未来的研究。此外，我们的实验为将新策略纳入现有解决方案提供了关键见解。

    This paper focuses on enhancing Bengali Document Layout Analysis (DLA) using the YOLOv8 model and innovative post-processing techniques. We tackle challenges unique to the complex Bengali script by employing data augmentation for model robustness. After meticulous validation set evaluation, we fine-tune our approach on the complete dataset, leading to a two-stage prediction strategy for accurate element segmentation. Our ensemble model, combined with post-processing, outperforms individual base architectures, addressing issues identified in the BaDLAD dataset. By leveraging this approach, we aim to advance Bengali document analysis, contributing to improved OCR and document comprehension and BaDLAD serves as a foundational resource for this endeavor, aiding future research in the field. Furthermore, our experiments provided key insights to incorporate new strategies into the established solution.
    
[^5]: Sparkles: 解锁多图聊天以实现多模态指令跟踪模型

    Sparkles: Unlocking Chats Across Multiple Images for Multimodal Instruction-Following Models. (arXiv:2308.16463v1 [cs.CV])

    [http://arxiv.org/abs/2308.16463](http://arxiv.org/abs/2308.16463)

    Sparkles是一个多模态指令跟踪模型，通过整合文本和图像实现多图对话。我们引入了SparklesDialogue数据集和SparklesEval基准来支持训练和评估。实验证实了SparklesChat在理解多图对话方面的有效性。

    

    当使用指令跟踪数据来进行微调时，大型语言模型在各种任务上展现出了强大的零-shot性能。多模态指令跟踪模型通过整合文本和图像进一步扩展了这些能力。然而，现有的模型（如MiniGPT-4）在涉及多个图像的情况下保持对话连贯性面临挑战。一个主要原因是缺乏一个专门针对这一关键应用的数据集。为了弥合这些差距，我们提出了SparklesChat，一个用于多图对话的多模态指令跟踪模型。为了支持训练，我们引入了SparklesDialogue，这是第一个专为单词级交错多图像和文本交互而定制的机器生成对话数据集。此外，我们构建了SparklesEval，一个借助GPT辅助的基准，用于定量评估模型在多个图像和对话轮次中的对话能力。我们的实验验证了SparklesChat在理解多图对话方面的有效性。

    Large language models exhibit enhanced zero-shot performance on various tasks when fine-tuned with instruction-following data. Multimodal instruction-following models extend these capabilities by integrating both text and images. However, existing models such as MiniGPT-4 face challenges in maintaining dialogue coherence in scenarios involving multiple images. A primary reason is the lack of a specialized dataset for this critical application. To bridge these gaps, we present SparklesChat, a multimodal instruction-following model for open-ended dialogues across multiple images. To support the training, we introduce SparklesDialogue, the first machine-generated dialogue dataset tailored for word-level interleaved multi-image and text interactions. Furthermore, we construct SparklesEval, a GPT-assisted benchmark for quantitatively assessing a model's conversational competence across multiple images and dialogue turns. Our experiments validate the effectiveness of SparklesChat in understa
    
[^6]: 克服连续学习中的稳定性差距

    Overcoming the Stability Gap in Continual Learning. (arXiv:2306.01904v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2306.01904](http://arxiv.org/abs/2306.01904)

    本论文研究了如何克服连续学习中的稳定性差距，并通过发现一种显著减少这种差距的方法，在大规模类别增量学习实验中大幅减少了网络更新的次数。

    

    在许多实际应用中，随着数据集大小的增长，深度神经网络往往需要从头开始重新训练。考虑到重新训练的计算开销，人们认为连续学习可以使网络更新更加高效。实现这一目标的障碍是稳定性差距，即在更新新数据时，先前学习的数据性能会下降，然后才得以恢复。解决这个问题可以减少网络更新的次数，提高计算效率。我们研究了如何缓解稳定性差距，并测试了多种假设以了解其产生原因。这使我们发现了一种显著减少稳定性差距的方法。在大规模的增量类别学习实验中，我们能够显著减少连续学习所需的网络更新次数。我们的工作有可能推动连续学习在实际应用中的最新进展。

    In many real-world applications, deep neural networks are retrained from scratch as a dataset grows in size. Given the computational expense for retraining networks, it has been argued that continual learning could make updating networks more efficient. An obstacle to achieving this goal is the stability gap, which refers to an observation that when updating on new data, performance on previously learned data degrades before recovering. Addressing this problem would enable learning new data with fewer network updates, resulting in increased computational efficiency. We study how to mitigate the stability gap. We test a variety of hypotheses to understand why the stability gap occurs. This leads us to discover a method that vastly reduces this gap. In large-scale class incremental learning experiments, we are able to significantly reduce the number of network updates needed for continual learning. Our work has the potential to advance the state-of-the-art in continual learning for real-
    

