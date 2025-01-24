# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TT-BLIP: Enhancing Fake News Detection Using BLIP and Tri-Transformer](https://arxiv.org/abs/2403.12481) | TT-BLIP模型通过使用BLIP和Tri-Transformer技术，结合文本和图像的多模态信息提取，采用Multimodal Tri-Transformer融合特征，实现了增强的综合表征和改进的多模态数据分析。 |
| [^2] | [Explicitly Disentangled Representations in Object-Centric Learning.](http://arxiv.org/abs/2401.10148) | 这篇论文提出了一种在物体中心化学习中明确解开形状和纹理成分的方法，通过将潜在空间划分为两个不重叠的子集，使得模型更加稳定和有效。 |

# 详细

[^1]: TT-BLIP：使用BLIP和Tri-Transformer增强假新闻检测

    TT-BLIP: Enhancing Fake News Detection Using BLIP and Tri-Transformer

    [https://arxiv.org/abs/2403.12481](https://arxiv.org/abs/2403.12481)

    TT-BLIP模型通过使用BLIP和Tri-Transformer技术，结合文本和图像的多模态信息提取，采用Multimodal Tri-Transformer融合特征，实现了增强的综合表征和改进的多模态数据分析。

    

    arXiv:2403.12481v1 公告类型：新   摘要：检测假新闻受到了极大关注。许多先前的方法将独立编码的单模态数据进行串联，忽略了综合多模态信息的好处。此外，对于文本和图像缺乏专门的特征提取进一步限制了这些方法。本文介绍了一种名为TT-BLIP的端到端模型，该模型对三种类型的信息应用了引导式语言-图像预训练用于统一的视觉-语言理解和生成（BLIP）：BERT 和 BLIP\textsubscript{Txt} 用于文本，ResNet 和 BLIP\textsubscript{Img} 用于图像，以及用于多模态信息的双向 BLIP 编码器。多模态三角变换器使用三种类型的多头注意机制融合三模态特征，确保了增强表示和改进的多模态数据分析。实验使用了两个假新闻数据集，微博和Gossipcop。 结果表明，

    arXiv:2403.12481v1 Announce Type: new  Abstract: Detecting fake news has received a lot of attention. Many previous methods concatenate independently encoded unimodal data, ignoring the benefits of integrated multimodal information. Also, the absence of specialized feature extraction for text and images further limits these methods. This paper introduces an end-to-end model called TT-BLIP that applies the bootstrapping language-image pretraining for unified vision-language understanding and generation (BLIP) for three types of information: BERT and BLIP\textsubscript{Txt} for text, ResNet and BLIP\textsubscript{Img} for images, and bidirectional BLIP encoders for multimodal information. The Multimodal Tri-Transformer fuses tri-modal features using three types of multi-head attention mechanisms, ensuring integrated modalities for enhanced representations and improved multimodal data analysis. The experiments are performed using two fake news datasets, Weibo and Gossipcop. The results in
    
[^2]: 在物体中心化学习中明确解开的表示

    Explicitly Disentangled Representations in Object-Centric Learning. (arXiv:2401.10148v1 [cs.CV])

    [http://arxiv.org/abs/2401.10148](http://arxiv.org/abs/2401.10148)

    这篇论文提出了一种在物体中心化学习中明确解开形状和纹理成分的方法，通过将潜在空间划分为两个不重叠的子集，使得模型更加稳定和有效。

    

    从原始视觉数据中提取结构化表示是机器学习中一个重要且长期存在的挑战。最近，无监督学习物体中心化表示的技术引起了越来越多的关注。在这个背景下，增强潜在特征的稳定性可以提高下游任务训练的效率和效果。在这个方向上一个有希望的步骤是解开导致数据变化的因素。先前，不变卡槽注意实现了从其他特征中解开位置、尺度和方向。扩展这一方法，我们着重于分离形状和纹理组成部分。特别地，我们提出了一种新颖的架构，将物体中心化模型中的形状和纹理成分偏置为潜在空间维度的两个不重叠子集。这些子集是先验已知的，因此在训练过程之前。在一系列物体中心化测试中进行的实验揭示了...

    Extracting structured representations from raw visual data is an important and long-standing challenge in machine learning. Recently, techniques for unsupervised learning of object-centric representations have raised growing interest. In this context, enhancing the robustness of the latent features can improve the efficiency and effectiveness of the training of downstream tasks. A promising step in this direction is to disentangle the factors that cause variation in the data. Previously, Invariant Slot Attention disentangled position, scale, and orientation from the remaining features. Extending this approach, we focus on separating the shape and texture components. In particular, we propose a novel architecture that biases object-centric models toward disentangling shape and texture components into two non-overlapping subsets of the latent space dimensions. These subsets are known a priori, hence before the training process. Experiments on a range of object-centric benchmarks reveal t
    

