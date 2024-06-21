# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [EquiAV: Leveraging Equivariance for Audio-Visual Contrastive Learning](https://arxiv.org/abs/2403.09502) | 这项研究提出了一种利用等变性进行音频-视觉对比学习的新框架，通过一个共享的基于注意力的变换预测器来实现特征聚合和嵌入表示，有效提供了强大的监督，且计算开销最小。 |
| [^2] | [A Survey on Image-text Multimodal Models.](http://arxiv.org/abs/2309.15857) | 图像-文本多模型综述论文全面回顾了其发展历程和当前状态，提出了新的分类方法，并阐明了该领域的挑战和潜在研究方向。 |

# 详细

[^1]: EquiAV: 利用等变性进行音频-视觉对比学习

    EquiAV: Leveraging Equivariance for Audio-Visual Contrastive Learning

    [https://arxiv.org/abs/2403.09502](https://arxiv.org/abs/2403.09502)

    这项研究提出了一种利用等变性进行音频-视觉对比学习的新框架，通过一个共享的基于注意力的变换预测器来实现特征聚合和嵌入表示，有效提供了强大的监督，且计算开销最小。

    

    自我监督的音频-视觉表示学习最近取得了重大进展，展示出捕捉丰富综合表示的潜力。然而，尽管数据增强在许多学习方法中已经得到验证，音频-视觉学习仍然很难充分利用这些优势，因为增强可能会轻易破坏输入对之间的对应关系。为了解决这一限制，我们引入了EquiAV，一种利用等变性进行音频-视觉对比学习的新框架。我们的方法从扩展等变性开始进行音频-视觉学习，通过一个共享的基于注意力的变换预测器来促进。它使得来自不同增强的特征能够聚合到一个代表性的嵌入中，提供强大的监督。值得注意的是，这是在最小计算开销的情况下实现的。大量消融研究和定性结果验证了我们方法的有效性。

    arXiv:2403.09502v1 Announce Type: cross  Abstract: Recent advancements in self-supervised audio-visual representation learning have demonstrated its potential to capture rich and comprehensive representations. However, despite the advantages of data augmentation verified in many learning methods, audio-visual learning has struggled to fully harness these benefits, as augmentations can easily disrupt the correspondence between input pairs. To address this limitation, we introduce EquiAV, a novel framework that leverages equivariance for audio-visual contrastive learning. Our approach begins with extending equivariance to audio-visual learning, facilitated by a shared attention-based transformation predictor. It enables the aggregation of features from diverse augmentations into a representative embedding, providing robust supervision. Notably, this is achieved with minimal computational overhead. Extensive ablation studies and qualitative results verify the effectiveness of our method. 
    
[^2]: 图像-文本多模型综述论文

    A Survey on Image-text Multimodal Models. (arXiv:2309.15857v1 [cs.CL])

    [http://arxiv.org/abs/2309.15857](http://arxiv.org/abs/2309.15857)

    图像-文本多模型综述论文全面回顾了其发展历程和当前状态，提出了新的分类方法，并阐明了该领域的挑战和潜在研究方向。

    

    在人工智能不断发展的背景下，图像和文本信息的融合成为一个至关重要的领域，导致了图像-文本多模型的出现。本论文全面回顾了图像-文本多模型的发展历程和当前状态，探讨了它们的应用价值、挑战和潜在研究方向。首先，我们重新审视了这些模型的基本概念和发展里程碑，引入了一种新的分类方法，将它们的发展分为三个不同的阶段，基于它们被引入的时间和对学科的影响。此外，基于任务在学术领域中的重要性和普及性，我们提出了将与图像-文本多模型相关的任务划分为五个主要类型的分类方法，阐明了每个类别内的最新进展和关键技术。尽管这些模型取得了显著的成就，但仍面临着许多挑战。

    Amidst the evolving landscape of artificial intelligence, the convergence of visual and textual information has surfaced as a crucial frontier, leading to the advent of image-text multimodal models. This paper provides a comprehensive review of the evolution and current state of image-text multimodal models, exploring their application value, challenges, and potential research trajectories. Initially, we revisit the basic concepts and developmental milestones of these models, introducing a novel classification that segments their evolution into three distinct phases, based on their time of introduction and subsequent impact on the discipline. Furthermore, based on the tasks' significance and prevalence in the academic landscape, we propose a categorization of the tasks associated with image-text multimodal models into five major types, elucidating the recent progress and key technologies within each category. Despite the remarkable accomplishments of these models, numerous challenges a
    

