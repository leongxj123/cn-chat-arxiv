# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Joint chest X-ray diagnosis and clinical visual attention prediction with multi-stage cooperative learning: enhancing interpretability](https://arxiv.org/abs/2403.16970) | 该论文引入了一种新的深度学习框架，用于联合疾病诊断和胸部X光扫描对应视觉显著性图的预测，通过设计新颖的双编码器多任务UNet并利用多尺度特征融合分类器来提高计算辅助诊断的可解释性和质量。 |
| [^2] | [Learning from Reduced Labels for Long-Tailed Data](https://arxiv.org/abs/2403.16469) | 提出了一种名为Reduced Label的新型弱监督标签设置，能够高效地学习长尾数据，避免了尾部样本监督信息的下降，降低了标签成本 |
| [^3] | [Neuro-Symbolic Video Search](https://arxiv.org/abs/2403.11021) | 提出了一种神经网络符号视频搜索系统，该系统利用视觉-语言模型进行语义理解，并通过状态机和时间逻辑公式对事件的长期演变进行推理，从而实现高效的场景识别。 |

# 详细

[^1]: 联合胸部X光诊断和临床视觉注意力预测的多阶段协作学习：增强可解释性

    Joint chest X-ray diagnosis and clinical visual attention prediction with multi-stage cooperative learning: enhancing interpretability

    [https://arxiv.org/abs/2403.16970](https://arxiv.org/abs/2403.16970)

    该论文引入了一种新的深度学习框架，用于联合疾病诊断和胸部X光扫描对应视觉显著性图的预测，通过设计新颖的双编码器多任务UNet并利用多尺度特征融合分类器来提高计算辅助诊断的可解释性和质量。

    

    随着深度学习成为计算辅助诊断的最新技术，自动决策的可解释性对临床部署至关重要。尽管在这一领域提出了各种方法，但在放射学筛查过程中临床医生的视觉注意力图为提供重要洞察提供了独特的资产，并有可能提高计算辅助诊断的质量。通过这篇论文，我们引入了一种新颖的深度学习框架，用于联合疾病诊断和胸部X光扫描对应视觉显著性图的预测。具体来说，我们设计了一种新颖的双编码器多任务UNet，利用了DenseNet201主干和基于残差和膨胀激励块的编码器来提取用于显著性图预测的多样特征，并使用多尺度特征融合分类器进行疾病分类。

    arXiv:2403.16970v1 Announce Type: cross  Abstract: As deep learning has become the state-of-the-art for computer-assisted diagnosis, interpretability of the automatic decisions is crucial for clinical deployment. While various methods were proposed in this domain, visual attention maps of clinicians during radiological screening offer a unique asset to provide important insights and can potentially enhance the quality of computer-assisted diagnosis. With this paper, we introduce a novel deep-learning framework for joint disease diagnosis and prediction of corresponding visual saliency maps for chest X-ray scans. Specifically, we designed a novel dual-encoder multi-task UNet, which leverages both a DenseNet201 backbone and a Residual and Squeeze-and-Excitation block-based encoder to extract diverse features for saliency map prediction, and a multi-scale feature-fusion classifier to perform disease classification. To tackle the issue of asynchronous training schedules of individual tasks
    
[^2]: 学习从减少标签的长尾数据中

    Learning from Reduced Labels for Long-Tailed Data

    [https://arxiv.org/abs/2403.16469](https://arxiv.org/abs/2403.16469)

    提出了一种名为Reduced Label的新型弱监督标签设置，能够高效地学习长尾数据，避免了尾部样本监督信息的下降，降低了标签成本

    

    长尾数据在现实世界的分类任务中普遍存在，并且严重依赖监督信息，这使得注释过程异常耗时且费力。然而，尽管减少标注成本是缓解标签成本的常见方法，但现有的弱监督学习方法很难充分保留尾部样本的监督信息，导致尾部类别的准确率下降。为了缓解这一问题，我们提出了一种名为Reduced Label的新型弱监督标签设置。所提出的标签设置不仅避免了尾部样本的监督信息下降，还减少了与长尾数据相关的标签成本。此外，我们提出了一个简单直观且高效的无偏框架，具有强大的理论保证，可以从这些Reduced Labels中学习。在包括Imag在内的基准数据集上进行了广泛的实验

    arXiv:2403.16469v1 Announce Type: new  Abstract: Long-tailed data is prevalent in real-world classification tasks and heavily relies on supervised information, which makes the annotation process exceptionally labor-intensive and time-consuming. Unfortunately, despite being a common approach to mitigate labeling costs, existing weakly supervised learning methods struggle to adequately preserve supervised information for tail samples, resulting in a decline in accuracy for the tail classes. To alleviate this problem, we introduce a novel weakly supervised labeling setting called Reduced Label. The proposed labeling setting not only avoids the decline of supervised information for the tail samples, but also decreases the labeling costs associated with long-tailed data. Additionally, we propose an straightforward and highly efficient unbiased framework with strong theoretical guarantees to learn from these Reduced Labels. Extensive experiments conducted on benchmark datasets including Imag
    
[^3]: 神经符号视频搜索

    Neuro-Symbolic Video Search

    [https://arxiv.org/abs/2403.11021](https://arxiv.org/abs/2403.11021)

    提出了一种神经网络符号视频搜索系统，该系统利用视觉-语言模型进行语义理解，并通过状态机和时间逻辑公式对事件的长期演变进行推理，从而实现高效的场景识别。

    

    近年来视频数据生产的空前激增需求高效的工具，以从视频中提取有意义的帧供下游任务使用。 长期时间推理是帧检索系统的一个关键要求。 虽然 VideoLLaMA 和 ViCLIP 等最先进的基础模型在短期语义理解方面表现优异，但它们在跨帧的长期推理方面却令人惊讶地失败。 这种失败的一个关键原因是它们将逐帧感知和时间推理交织成单个深度网络。 因此，解耦但共同设计语义理解和时间推理对于高效的场景识别是至关重要的。 我们提出了一种系统，利用视觉-语言模型对单个帧进行语义理解，但有效地通过使用状态机和时间逻辑（TL）公式对事件的长期演变进行推理，这些公式在本质上捕捉了记忆。

    arXiv:2403.11021v1 Announce Type: cross  Abstract: The unprecedented surge in video data production in recent years necessitates efficient tools to extract meaningful frames from videos for downstream tasks. Long-term temporal reasoning is a key desideratum for frame retrieval systems. While state-of-the-art foundation models, like VideoLLaMA and ViCLIP, are proficient in short-term semantic understanding, they surprisingly fail at long-term reasoning across frames. A key reason for this failure is that they intertwine per-frame perception and temporal reasoning into a single deep network. Hence, decoupling but co-designing semantic understanding and temporal reasoning is essential for efficient scene identification. We propose a system that leverages vision-language models for semantic understanding of individual frames but effectively reasons about the long-term evolution of events using state machines and temporal logic (TL) formulae that inherently capture memory. Our TL-based reas
    

