# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Are Vision Language Models Texture or Shape Biased and Can We Steer Them?](https://arxiv.org/abs/2403.09193) | 本文研究了广泛应用的视觉语言模型中的纹理与形状偏见，发现这些模型通常比视觉编码器更偏向形状，暗示视觉偏见在一定程度上会受到文本的调节 |
| [^2] | [Data augmentation with automated machine learning: approaches and performance comparison with classical data augmentation methods](https://arxiv.org/abs/2403.08352) | 自动化机器学习的数据增强方法旨在自动化数据增强过程，为改善机器学习模型泛化性能提供了更高效的方式。 |
| [^3] | [$\sigma$-zero: Gradient-based Optimization of $\ell_0$-norm Adversarial Examples](https://arxiv.org/abs/2402.01879) | 该论文提出了一种新的基于梯度的$\ell_0$范数攻击方法$\sigma$-zero，其利用了$\ell_0$范数的可微近似和自适应投影运算符，能够在非凸和非可微的约束下优化，从而评估深度网络对稀疏$\ell_0$范数攻击的鲁棒性。 |
| [^4] | [DoraemonGPT: Toward Understanding Dynamic Scenes with Large Language Models](https://arxiv.org/abs/2401.08392) | DoraemonGPT是一个由LLMs驱动的系统，旨在处理动态视频任务，通过将视频转换为符号记忆来进行空间-时间查询和推理，并取得简洁的中间结果。 |
| [^5] | [Deep Augmentation: Self-Supervised Learning with Transformations in Activation Space](https://arxiv.org/abs/2303.14537) | 深度增强是一种利用dropout或PCA在神经网络中转换目标层的方法，有效改善性能和泛化能力。在对比学习任务中，在Transformers、ResNets和图神经网络等基础模型上，通过深度增强实现了显著的性能提升，但在监督问题上效果相反。 |
| [^6] | [Digital Divides in Scene Recognition: Uncovering Socioeconomic Biases in Deep Learning Systems.](http://arxiv.org/abs/2401.13097) | 该研究研究了深度学习系统中的社会经济偏见对场景识别的影响，发现了预训练的卷积神经网络在低社会经济地位的家庭照片中显示出更低的分类准确度和分类置信度，并更容易分配具有冒犯性的标签。 |

# 详细

[^1]: 视觉语言模型是纹理偏见还是形状偏见，我们可以引导它们吗？

    Are Vision Language Models Texture or Shape Biased and Can We Steer Them?

    [https://arxiv.org/abs/2403.09193](https://arxiv.org/abs/2403.09193)

    本文研究了广泛应用的视觉语言模型中的纹理与形状偏见，发现这些模型通常比视觉编码器更偏向形状，暗示视觉偏见在一定程度上会受到文本的调节

    

    arXiv:2403.09193v1 公告类型: 跨领域 摘要: 视觉语言模型（VLMs）在短短几年内彻底改变了计算机视觉模型的格局，开启了一系列新的应用，从零样本图像分类到图像字幕生成，再到视觉问答。与纯视觉模型不同，它们提供了通过语言提示访问视觉内容的直观方式。这种模型的广泛适用性引发我们思考它们是否也与人类视觉一致 - 具体来说，它们在多模态融合中有多大程度地采用了人类引导的视觉偏见，或者它们是否只是从纯视觉模型中继承了偏见。其中一个重要的视觉偏见是纹理与形状偏见，即局部信息的主导地位。在本文中，我们研究了一系列流行的VLMs中的这种偏见。有趣的是，我们发现VLMs通常比它们的视觉编码器更偏向于形状，这表明视觉偏见在一定程度上通过文本进行调节。

    arXiv:2403.09193v1 Announce Type: cross  Abstract: Vision language models (VLMs) have drastically changed the computer vision model landscape in only a few years, opening an exciting array of new applications from zero-shot image classification, over to image captioning, and visual question answering. Unlike pure vision models, they offer an intuitive way to access visual content through language prompting. The wide applicability of such models encourages us to ask whether they also align with human vision - specifically, how far they adopt human-induced visual biases through multimodal fusion, or whether they simply inherit biases from pure vision models. One important visual bias is the texture vs. shape bias, or the dominance of local over global information. In this paper, we study this bias in a wide range of popular VLMs. Interestingly, we find that VLMs are often more shape-biased than their vision encoders, indicating that visual biases are modulated to some extent through text
    
[^2]: 利用自动化机器学习的数据增强方法及与传统数据增强方法性能比较

    Data augmentation with automated machine learning: approaches and performance comparison with classical data augmentation methods

    [https://arxiv.org/abs/2403.08352](https://arxiv.org/abs/2403.08352)

    自动化机器学习的数据增强方法旨在自动化数据增强过程，为改善机器学习模型泛化性能提供了更高效的方式。

    

    数据增强被认为是常用于提高机器学习模型泛化性能的最重要的正则化技术。它主要涉及应用适当的数据转换操作，以创建具有所需属性的新数据样本。尽管其有效性，这一过程通常具有挑战性，因为手动创建和测试不同候选增强及其超参数需耗费大量时间。自动化数据增强方法旨在自动化这一过程。最先进的方法通常依赖于自动化机器学习（AutoML）原则。本研究提供了基于AutoML的数据增强技术的全面调查。我们讨论了使用AutoML实现数据增强的各种方法，包括数据操作、数据集成和数据合成技术。我们详细讨论了技术

    arXiv:2403.08352v1 Announce Type: cross  Abstract: Data augmentation is arguably the most important regularization technique commonly used to improve generalization performance of machine learning models. It primarily involves the application of appropriate data transformation operations to create new data samples with desired properties. Despite its effectiveness, the process is often challenging because of the time-consuming trial and error procedures for creating and testing different candidate augmentations and their hyperparameters manually. Automated data augmentation methods aim to automate the process. State-of-the-art approaches typically rely on automated machine learning (AutoML) principles. This work presents a comprehensive survey of AutoML-based data augmentation techniques. We discuss various approaches for accomplishing data augmentation with AutoML, including data manipulation, data integration and data synthesis techniques. We present extensive discussion of technique
    
[^3]: $\sigma$-zero: 基于梯度的$\ell_0$-范数对抗样本优化

    $\sigma$-zero: Gradient-based Optimization of $\ell_0$-norm Adversarial Examples

    [https://arxiv.org/abs/2402.01879](https://arxiv.org/abs/2402.01879)

    该论文提出了一种新的基于梯度的$\ell_0$范数攻击方法$\sigma$-zero，其利用了$\ell_0$范数的可微近似和自适应投影运算符，能够在非凸和非可微的约束下优化，从而评估深度网络对稀疏$\ell_0$范数攻击的鲁棒性。

    

    评估深度网络对基于梯度攻击的对抗鲁棒性是具有挑战性的。虽然大多数攻击考虑$\ell_2$和$\ell_\infty$范数约束来制造输入扰动，但只有少数研究了稀疏的$\ell_1$和$\ell_0$范数攻击。特别是，由于在非凸且非可微约束上进行优化的固有复杂性，$\ell_0$范数攻击是研究最少的。然而，使用这些攻击评估对抗鲁棒性可以揭示在更传统的$\ell_2$和$\ell_\infty$范数攻击中未能测试出的弱点。在这项工作中，我们提出了一种新颖的$\ell_0$范数攻击，称为$\sigma$-zero，它利用了$\ell_0$范数的一个特殊可微近似来促进基于梯度的优化，并利用自适应投影运算符动态调整损失最小化和扰动稀疏性之间的权衡。通过在MNIST、CIFAR10和ImageNet数据集上进行广泛评估，包括...

    Evaluating the adversarial robustness of deep networks to gradient-based attacks is challenging. While most attacks consider $\ell_2$- and $\ell_\infty$-norm constraints to craft input perturbations, only a few investigate sparse $\ell_1$- and $\ell_0$-norm attacks. In particular, $\ell_0$-norm attacks remain the least studied due to the inherent complexity of optimizing over a non-convex and non-differentiable constraint. However, evaluating adversarial robustness under these attacks could reveal weaknesses otherwise left untested with more conventional $\ell_2$- and $\ell_\infty$-norm attacks. In this work, we propose a novel $\ell_0$-norm attack, called $\sigma$-zero, which leverages an ad hoc differentiable approximation of the $\ell_0$ norm to facilitate gradient-based optimization, and an adaptive projection operator to dynamically adjust the trade-off between loss minimization and perturbation sparsity. Extensive evaluations using MNIST, CIFAR10, and ImageNet datasets, involving
    
[^4]: DoraemonGPT：朝向理解具有大语言模型的动态场景迈进

    DoraemonGPT: Toward Understanding Dynamic Scenes with Large Language Models

    [https://arxiv.org/abs/2401.08392](https://arxiv.org/abs/2401.08392)

    DoraemonGPT是一个由LLMs驱动的系统，旨在处理动态视频任务，通过将视频转换为符号记忆来进行空间-时间查询和推理，并取得简洁的中间结果。

    

    最近由LLM驱动的视觉代理主要集中于解决基于图像的任务，这限制了它们理解动态场景的能力，使其远离像引导学生进行实验室实验和识别错误这样的真实应用。考虑到视频模态更好地反映了真实世界场景的不断变化性质，我们设计了DoraemonGPT，这是一个由LLM驱动的综合概念简洁系统，用于处理动态视频任务。给定一个带有问题/任务的视频，DoraemonGPT首先将输入视频转换为存储与任务相关属性的符号存储器。这种结构化表示允许通过精心设计的子任务工具进行空间-时间查询和推理，从而产生简洁的中间结果。鉴于LLM在涉及专业领域（例如分析实验中潜在的科学原理）时具有有限的内部知识，我们引入了

    arXiv:2401.08392v2 Announce Type: replace-cross  Abstract: Recent LLM-driven visual agents mainly focus on solving image-based tasks, which limits their ability to understand dynamic scenes, making it far from real-life applications like guiding students in laboratory experiments and identifying their mistakes. Considering the video modality better reflects the ever-changing nature of real-world scenarios, we devise DoraemonGPT, a comprehensive and conceptually elegant system driven by LLMs to handle dynamic video tasks. Given a video with a question/task, DoraemonGPT begins by converting the input video into a symbolic memory that stores task-related attributes. This structured representation allows for spatial-temporal querying and reasoning by well-designed sub-task tools, resulting in concise intermediate results. Recognizing that LLMs have limited internal knowledge when it comes to specialized domains (e.g., analyzing the scientific principles underlying experiments), we incorpor
    
[^5]: 深度增强：在激活空间中使用自监督学习进行数据增强

    Deep Augmentation: Self-Supervised Learning with Transformations in Activation Space

    [https://arxiv.org/abs/2303.14537](https://arxiv.org/abs/2303.14537)

    深度增强是一种利用dropout或PCA在神经网络中转换目标层的方法，有效改善性能和泛化能力。在对比学习任务中，在Transformers、ResNets和图神经网络等基础模型上，通过深度增强实现了显著的性能提升，但在监督问题上效果相反。

    

    我们提出了一种称为深度增强的方法，通过使用辍学或PCA来转换神经网络中的目标层，以提高性能和泛化能力。我们通过在自然语言处理、计算机视觉和图学习中的对比学习任务上进行大量实验来展示深度增强。 我们观察到在对比学习的基础模型中，如Transformers、ResNets和图神经网络上深度增强能够带来显著的性能提升，但在相应的监督问题上观察到相反的效果。 我们的分析表明，深度增强减轻了层之间的相互适应，即"崩溃"形式的问题。 我们利用这一观察结果制定了一种选择目标层的方法；特别是，我们的实验表明，用深度增强定位更深层次的层要优于增强输入数据。 这种方法的简单网络和模态无关性使其

    arXiv:2303.14537v2 Announce Type: replace-cross  Abstract: We introduce Deep Augmentation, an approach to implicit data augmentation using dropout or PCA to transform a targeted layer within a neural network to improve performance and generalization. We demonstrate Deep Augmentation through extensive experiments on contrastive learning tasks in NLP, computer vision, and graph learning. We observe substantial performance gains with Transformers, ResNets, and Graph Neural Networks as the underlying models in contrastive learning, but observe inverse effects on the corresponding supervised problems. Our analysis suggests that Deep Augmentation alleviates co-adaption between layers, a form of "collapse." We use this observation to formulate a method for selecting which layer to target; in particular, our experimentation reveals that targeting deeper layers with Deep Augmentation outperforms augmenting the input data. The simple network- and modality-agnostic nature of this approach enables
    
[^6]: 场景识别中的数字鸿沟：揭示深度学习系统中的社会经济偏见

    Digital Divides in Scene Recognition: Uncovering Socioeconomic Biases in Deep Learning Systems. (arXiv:2401.13097v1 [cs.CV])

    [http://arxiv.org/abs/2401.13097](http://arxiv.org/abs/2401.13097)

    该研究研究了深度学习系统中的社会经济偏见对场景识别的影响，发现了预训练的卷积神经网络在低社会经济地位的家庭照片中显示出更低的分类准确度和分类置信度，并更容易分配具有冒犯性的标签。

    

    基于计算机的场景理解影响了从城市规划到自动驾驶的领域，然而我们对这些技术在社会差异中的表现了解甚少。我们研究了深度卷积神经网络（dCNNs）在场景分类中的偏见，使用了来自全球和美国的近百万张图片，包括用户提交的家庭照片和Airbnb的房源照片。我们运用了统计模型，对家庭收入、人类发展指数（HDI）等社会经济指标以及公开数据来源（CIA和美国人口普查）的人口统计因素对dCNNs的表现影响进行了量化。我们的分析发现了显著的社会经济偏见，预训练的dCNNs表现出更低的分类准确度、更低的分类置信度，以及更高的倾向性在低社会经济地位的家庭（例如“废墟”，“贫民窟”）的图片中分配具有冒犯性的标签。这种趋势是持续的。

    Computer-based scene understanding has influenced fields ranging from urban planning to autonomous vehicle performance, yet little is known about how well these technologies work across social differences. We investigate the biases of deep convolutional neural networks (dCNNs) in scene classification, using nearly one million images from global and US sources, including user-submitted home photographs and Airbnb listings. We applied statistical models to quantify the impact of socioeconomic indicators such as family income, Human Development Index (HDI), and demographic factors from public data sources (CIA and US Census) on dCNN performance. Our analyses revealed significant socioeconomic bias, where pretrained dCNNs demonstrated lower classification accuracy, lower classification confidence, and a higher tendency to assign labels that could be offensive when applied to homes (e.g., "ruin", "slum"), especially in images from homes with lower socioeconomic status (SES). This trend is c
    

