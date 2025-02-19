# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Domain Adaptation based Interpretable Image Emotion Recognition using Facial Expression Recognition](https://arxiv.org/abs/2011.08388) | 本论文提出了一种基于领域自适应的图像情绪识别方法，通过提出面部情绪识别系统并将其适应为图像情绪识别系统，解决了预训练模型和数据集不足的挑战。同时提出了一种新颖的解释性方法，用于解释情绪识别中关键的视觉特征。 |
| [^2] | [Do humans and machines have the same eyes? Human-machine perceptual differences on image classification.](http://arxiv.org/abs/2304.08733) | 本文研究通过图像分类探究了人机感知差异，发现即使准确率相似，人类和机器的答案分布也可能不同，并提出了一种后期人机合作来提高任务表现。 |

# 详细

[^1]: 基于领域自适应的可解释图像情绪识别，并利用面部表情识别

    Domain Adaptation based Interpretable Image Emotion Recognition using Facial Expression Recognition

    [https://arxiv.org/abs/2011.08388](https://arxiv.org/abs/2011.08388)

    本论文提出了一种基于领域自适应的图像情绪识别方法，通过提出面部情绪识别系统并将其适应为图像情绪识别系统，解决了预训练模型和数据集不足的挑战。同时提出了一种新颖的解释性方法，用于解释情绪识别中关键的视觉特征。

    

    本文提出了一种领域自适应技术，用于识别包含面部和非面部物体以及非人类组件的通用图像中的情绪。它解决了图像情绪识别（IER）中预训练模型和良好注释数据集的不足挑战。首先，提出了一种基于深度学习的面部情绪识别（FER）系统，将给定的面部图像分类为离散情绪类别。然后，提出了一种图像识别系统，将提出的FER系统适应于利用领域自适应识别图像所传达的情绪。它将通用图像分类为“快乐”，“悲伤”，“仇恨”和“愤怒”类别。还提出了一种新颖的解释性方法，称为分而治之的Shap（DnCShap），用于解释情绪识别中高度相关的视觉特征。

    A domain adaptation technique has been proposed in this paper to identify the emotions in generic images containing facial & non-facial objects and non-human components. It addresses the challenge of the insufficient availability of pre-trained models and well-annotated datasets for image emotion recognition (IER). It starts with proposing a facial emotion recognition (FER) system and then moves on to adapting it for image emotion recognition. First, a deep-learning-based FER system has been proposed that classifies a given facial image into discrete emotion classes. Further, an image recognition system has been proposed that adapts the proposed FER system to recognize the emotions portrayed by images using domain adaptation. It classifies the generic images into 'happy,' 'sad,' 'hate,' and 'anger' classes. A novel interpretability approach, Divide and Conquer based Shap (DnCShap), has also been proposed to interpret the highly relevant visual features for emotion recognition. The prop
    
[^2]: 人类和机器有相同的眼睛吗？基于图像分类的人机感知差异研究

    Do humans and machines have the same eyes? Human-machine perceptual differences on image classification. (arXiv:2304.08733v1 [cs.CV])

    [http://arxiv.org/abs/2304.08733](http://arxiv.org/abs/2304.08733)

    本文研究通过图像分类探究了人机感知差异，发现即使准确率相似，人类和机器的答案分布也可能不同，并提出了一种后期人机合作来提高任务表现。

    

    训练良好的计算机视觉模型通常通过模仿从训练标签中学到的人类行为来解决视觉任务。近期视觉研究的大部分努力集中在使用标准化基准来测量模型任务性能。然而，了解人与机器之间的感知差异方面的工作还很有限。为了填补这一空白，我们的研究首先量化并分析了两种来源错误的统计分布。然后我们通过难度级别对任务进行排序，探讨人类与机器专业知识的差异。即使人类和机器的整体准确性相似，答案的分布也可能会有所不同。利用人类和机器之间的感知差异，我们通过实证研究表明了一种后期人机合作，其表现比单独的人或机器更好。

    Trained computer vision models are assumed to solve vision tasks by imitating human behavior learned from training labels. Most efforts in recent vision research focus on measuring the model task performance using standardized benchmarks. Limited work has been done to understand the perceptual difference between humans and machines. To fill this gap, our study first quantifies and analyzes the statistical distributions of mistakes from the two sources. We then explore human vs. machine expertise after ranking tasks by difficulty levels. Even when humans and machines have similar overall accuracies, the distribution of answers may vary. Leveraging the perceptual difference between humans and machines, we empirically demonstrate a post-hoc human-machine collaboration that outperforms humans or machines alone.
    

