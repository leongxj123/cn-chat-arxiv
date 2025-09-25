# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CLIP Can Understand Depth](https://arxiv.org/abs/2402.03251) | 本文研究了将CLIP用于单目深度估计的问题，通过联合训练反卷积解码器和可学习嵌入矩阵，使得CLIP能够理解深度，该方法在深度估计任务上取得了令人印象深刻的性能，并优于之前的方法。 |

# 详细

[^1]: CLIP可以理解深度

    CLIP Can Understand Depth

    [https://arxiv.org/abs/2402.03251](https://arxiv.org/abs/2402.03251)

    本文研究了将CLIP用于单目深度估计的问题，通过联合训练反卷积解码器和可学习嵌入矩阵，使得CLIP能够理解深度，该方法在深度估计任务上取得了令人印象深刻的性能，并优于之前的方法。

    

    最近关于将CLIP推广到单目深度估计的研究表明，在网络爬取的数据上预训练的CLIP在图像块和与深度相关的提示之间得到适当相似性是低效的。在本文中，我们适应CLIP用于有意义的密集预测单目深度估计，而无需微调其原始的视觉-语言对齐。通过联合训练一个紧凑的反卷积解码器和一个名为mirror的小型可学习嵌入矩阵作为其文本编码器的静态提示，CLIP能够理解深度。通过这种方法，我们的模型在NYU Depth v2和KITTI数据集上展现出了令人印象深刻的性能，与几个先前的仅视觉模型相匹配，而且胜过了每个基于CLIP的深度估计模型。关于时间深度一致性和空间连续性的实验证明，我们提出的框架能够有效地优化CLIP的先验知识。此外，对于时滞研究进行了消融实验。

    Recent studies on generalizing CLIP for monocular depth estimation reveal that CLIP pre-trained on web-crawled data is inefficient for deriving proper similarities between image patches and depth-related prompts. In this paper, we adapt CLIP for meaningful quality of monocular depth estimation with dense prediction, without fine-tuning its original vision-language alignment. By jointly training a compact deconvolutional decoder with a tiny learnable embedding matrix named mirror, as a static prompt for its text encoder, CLIP is enabled to understand depth. With this approach, our model exhibits impressive performance matching several previous state-of-the-art vision-only models on the NYU Depth v2 and KITTI datasets, outperforming every CLIP-based depth estimation model with a large margin. Experiments on temporal depth consistency and spatial continuity demonstrate that the prior knowledge of CLIP can be effectively refined by our proposed framework. Furthermore, an ablation study on 
    

