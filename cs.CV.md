# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [OVOR: OnePrompt with Virtual Outlier Regularization for Rehearsal-Free Class-Incremental Learning](https://arxiv.org/abs/2402.04129) | 这项研究提出了一种新的正则化方法，利用虚拟异常值来改善无需回顾的类增量学习过程中不同任务间的类别混淆问题，并且消除了额外的提示查询和组合计算开销。 |
| [^2] | [Adversarial Defenses via Vector Quantization.](http://arxiv.org/abs/2305.13651) | 该论文提出了两种基于矢量量化的新对抗性防御方法，能够在高维空间中提供理论保证和实验上的表现优势。 |
| [^3] | [Don't Get Me Wrong: How to Apply Deep Visual Interpretations to Time Series.](http://arxiv.org/abs/2203.07861) | 该论文提出了一个针对时间序列分类和分割任务的框架，通过六个度量来评估基于梯度、传播或干扰的事后可视化解释方法。实验结果表明，这些方法对于时间序列的解释具有较高的可信度和有效性。 |

# 详细

[^1]: OVOR：一种使用虚拟异常值正则化的OnePrompt方法，实现无需回顾的类增量学习

    OVOR: OnePrompt with Virtual Outlier Regularization for Rehearsal-Free Class-Incremental Learning

    [https://arxiv.org/abs/2402.04129](https://arxiv.org/abs/2402.04129)

    这项研究提出了一种新的正则化方法，利用虚拟异常值来改善无需回顾的类增量学习过程中不同任务间的类别混淆问题，并且消除了额外的提示查询和组合计算开销。

    

    最近的研究表明，利用大规模预训练模型和可学习的提示，在无需回顾的类增量学习（CIL）设置中可以实现比著名的基于回顾的方法更好的性能。无需回顾的CIL方法在区分不同任务的类别时遇到困难，因为它们并未一同训练。在这项研究中，我们提出了一种基于虚拟异常值的正则化方法，通过紧缩分类器的决策边界，减轻不同任务间类别的混淆。最近的基于提示的方法通常需要一个存储各任务特定提示的集合，以防止新任务的知识覆盖先前任务的知识，从而导致额外的查询和组合适当提示的计算开销。我们在论文中揭示，可以消除这种额外开销而不牺牲准确性。我们演示了简化的基于提示的方法可以达到与先前最新状态-of-the-art方法相当的结果。

    Recent works have shown that by using large pre-trained models along with learnable prompts, rehearsal-free methods for class-incremental learning (CIL) settings can achieve superior performance to prominent rehearsal-based ones. Rehearsal-free CIL methods struggle with distinguishing classes from different tasks, as those are not trained together. In this work we propose a regularization method based on virtual outliers to tighten decision boundaries of the classifier, such that confusion of classes among different tasks is mitigated. Recent prompt-based methods often require a pool of task-specific prompts, in order to prevent overwriting knowledge of previous tasks with that of the new task, leading to extra computation in querying and composing an appropriate prompt from the pool. This additional cost can be eliminated, without sacrificing accuracy, as we reveal in the paper. We illustrate that a simplified prompt-based method can achieve results comparable to previous state-of-the
    
[^2]: 基于矢量量化的对抗防御

    Adversarial Defenses via Vector Quantization. (arXiv:2305.13651v1 [cs.LG])

    [http://arxiv.org/abs/2305.13651](http://arxiv.org/abs/2305.13651)

    该论文提出了两种基于矢量量化的新对抗性防御方法，能够在高维空间中提供理论保证和实验上的表现优势。

    

    在随机离散化的基础上，我们在高维空间中利用矢量量化开发了两种新的对抗性防御方法，分别称为pRD和swRD。这些方法不仅在证明准确度方面提供了理论保证，而且通过大量实验表明，它们的表现与当前对抗防御技术相当甚至更优秀。这些方法可以扩展到一种版本，允许对目标分类器进行进一步训练，并展示出进一步改进的性能。

    Building upon Randomized Discretization, we develop two novel adversarial defenses against white-box PGD attacks, utilizing vector quantization in higher dimensional spaces. These methods, termed pRD and swRD, not only offer a theoretical guarantee in terms of certified accuracy, they are also shown, via abundant experiments, to perform comparably or even superior to the current art of adversarial defenses. These methods can be extended to a version that allows further training of the target classifier and demonstrates further improved performance.
    
[^3]: 不要误会我：如何将深度视觉解释应用于时间序列

    Don't Get Me Wrong: How to Apply Deep Visual Interpretations to Time Series. (arXiv:2203.07861v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2203.07861](http://arxiv.org/abs/2203.07861)

    该论文提出了一个针对时间序列分类和分割任务的框架，通过六个度量来评估基于梯度、传播或干扰的事后可视化解释方法。实验结果表明，这些方法对于时间序列的解释具有较高的可信度和有效性。

    

    在许多应用中，正确解释和理解深度学习模型非常重要。针对图像和自然语言处理的解释性视觉解释方法允许领域专家验证和理解几乎任何深度学习模型。然而，当推广到任意时间序列时，它们在本质上更加复杂和多样化。一个可视化解释是否解释了有效的推理或捕捉了实际特征是难以判断的。因此，我们需要客观评估来获得可信的质量指标，而不是盲目信任。我们提出了一个框架，包括六个正交度量，用于针对时间序列分类和分割任务的基于梯度、传播或干扰的事后视觉解释方法。实验研究包括了常见的时间序列神经网络架构和九种可视化解释方法。我们使用UCR r等多样的数据集评估了这些可视化解释方法。

    The correct interpretation and understanding of deep learning models are essential in many applications. Explanatory visual interpretation approaches for image, and natural language processing allow domain experts to validate and understand almost any deep learning model. However, they fall short when generalizing to arbitrary time series, which is inherently less intuitive and more diverse. Whether a visualization explains valid reasoning or captures the actual features is difficult to judge. Hence, instead of blind trust, we need an objective evaluation to obtain trustworthy quality metrics. We propose a framework of six orthogonal metrics for gradient-, propagation- or perturbation-based post-hoc visual interpretation methods for time series classification and segmentation tasks. An experimental study includes popular neural network architectures for time series and nine visual interpretation methods. We evaluate the visual interpretation methods with diverse datasets from the UCR r
    

