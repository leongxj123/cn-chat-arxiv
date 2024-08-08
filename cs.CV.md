# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DenseNets Reloaded: Paradigm Shift Beyond ResNets and ViTs](https://arxiv.org/abs/2403.19588) | 本文重新探讨了密集连接卷积网络（DenseNets），揭示了其相对于ResNet风格架构的被低估有效性，通过改进架构、块设计和训练方法，使得DenseNets可以与现代架构竞争，并在ImageNet-1K上接近最新技术水平。 |
| [^2] | [Trajectory Regularization Enhances Self-Supervised Geometric Representation](https://arxiv.org/abs/2403.14973) | 引入新的pose-estimation基准用于评估SSL几何表示，提出无监督轨迹正则化损失来增强SSL几何表示，在姿势估计性能上取得10-20%的提升，并提高了4%的性能和泛化能力。 |
| [^3] | [Meta-Prompting for Automating Zero-shot Visual Recognition with LLMs](https://arxiv.org/abs/2403.11755) | 提出了Meta-Prompting for Visual Recognition (MPVR)方法，通过仅需少量信息即可自动化零样本识别中的提示生成过程。 |
| [^4] | [Curriculum Learning for ab initio Deep Learned Refractive Optics](https://arxiv.org/abs/2302.01089) | 通过Curriculum Learning，提出了一种DeepLens设计方法，能够从随机初始化表面从头开始学习复合透镜的光学设计，克服了对良好初始设计的需求，并实现了自动设计经典成像透镜和大视场延伸景深计算透镜。 |
| [^5] | [A Comprehensive Augmentation Framework for Anomaly Detection.](http://arxiv.org/abs/2308.15068) | 本文提出了一种用于异常检测的综合增强框架，该框架通过选择性地利用适当的组合，分析并压缩模拟异常的关键特征，与基于重构的方法相结合，并采用分割训练策略，能够在MVTec异常检测数据集上优于以前的最先进方法。 |
| [^6] | [Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning.](http://arxiv.org/abs/2305.18403) | 本论文提出了一种名为LoRAPrune的框架，可以高效微调和部署大型预训练模型，通过利用低秩自适应的值和梯度来设计PEFT感知的剪枝准则，并提出了一个迭代剪枝过程来去除冗余参数，实验结果表明与最先进的方法相比，可以显著降低模型大小和推理时间，同时保持竞争性的准确性。 |
| [^7] | [Distill Gold from Massive Ores: Efficient Dataset Distillation via Critical Samples Selection.](http://arxiv.org/abs/2305.18381) | 研究提出了一种基于选择最有价值的样本的方法，以扩展现有的蒸馏算法，从而更好地利用训练样本，显著降低训练成本，拓展对更大更多元化数据集的数据集蒸馏，并持续提高性能。 |
| [^8] | [The Face of Populism: Examining Differences in Facial Emotional Expressions of Political Leaders Using Machine Learning.](http://arxiv.org/abs/2304.09914) | 本文使用机器学习的算法分析了来自15个不同国家的220个政治领袖的YouTube视频，总结了政治领袖面部情感表达的差异。 |
| [^9] | [ASR: Attention-alike Structural Re-parameterization.](http://arxiv.org/abs/2304.06345) | 该论文提出的ASR技术是一种新颖的深度学习技术，通过等效参数转换实现不同网络体系结构之间的互转。和现有的SRP方法相比，ASR可以成功考虑自注意模块，实现推理期间的性能提升，并在工业和实际应用中具有巨大潜力。 |
| [^10] | [Fingerprinting Generative Adversarial Networks.](http://arxiv.org/abs/2106.11760) | 本文提出了一种保护GAN知识产权的指纹识别方案，通过生成指纹样本并嵌入到分类器中进行版权验证，解决了前一种对分类模型的指纹识别方法在简单转移至GAN时遇到的隐蔽性和鲁棒性瓶颈，具有实际保护现代GAN模型的可行性。 |

# 详细

[^1]: DenseNets重生：超越ResNets和ViTs的范式转变

    DenseNets Reloaded: Paradigm Shift Beyond ResNets and ViTs

    [https://arxiv.org/abs/2403.19588](https://arxiv.org/abs/2403.19588)

    本文重新探讨了密集连接卷积网络（DenseNets），揭示了其相对于ResNet风格架构的被低估有效性，通过改进架构、块设计和训练方法，使得DenseNets可以与现代架构竞争，并在ImageNet-1K上接近最新技术水平。

    

    本文复苏了密集连接卷积网络（DenseNets），揭示了它们相对于主导的ResNet风格架构被低估的有效性。我们认为DenseNets的潜力被忽视，是因为未曾触及的训练方法和传统设计元素未能完全展现其能力。我们的初步研究表明，通过连接的密集连接是强大的，表明DenseNets可以被重新激活以与现代架构竞争。我们系统地改进了次优组件 - 架构调整、块重新设计和改进的训练配方，以扩展DenseNets并提高内存效率，同时保持连接快捷方式。我们的模型采用简单的架构元素，最终超越了Swin Transformer、ConvNeXt和DeiT-III - 残差学习谱系中的关键架构。此外，我们的模型在ImageNet-1K上展现出接近最新技术水平的性能，竞争wi

    arXiv:2403.19588v1 Announce Type: cross  Abstract: This paper revives Densely Connected Convolutional Networks (DenseNets) and reveals the underrated effectiveness over predominant ResNet-style architectures. We believe DenseNets' potential was overlooked due to untouched training methods and traditional design elements not fully revealing their capabilities. Our pilot study shows dense connections through concatenation are strong, demonstrating that DenseNets can be revitalized to compete with modern architectures. We methodically refine suboptimal components - architectural adjustments, block redesign, and improved training recipes towards widening DenseNets and boosting memory efficiency while keeping concatenation shortcuts. Our models, employing simple architectural elements, ultimately surpass Swin Transformer, ConvNeXt, and DeiT-III - key architectures in the residual learning lineage. Furthermore, our models exhibit near state-of-the-art performance on ImageNet-1K, competing wi
    
[^2]: 轨迹正则化增强自监督几何表示

    Trajectory Regularization Enhances Self-Supervised Geometric Representation

    [https://arxiv.org/abs/2403.14973](https://arxiv.org/abs/2403.14973)

    引入新的pose-estimation基准用于评估SSL几何表示，提出无监督轨迹正则化损失来增强SSL几何表示，在姿势估计性能上取得10-20%的提升，并提高了4%的性能和泛化能力。

    

    自监督学习（SSL）已被证明能够有效地学习高质量的表示，用于各种下游任务，主要关注语义任务。然而，在几何任务中的应用仍未得到充分探索，部分原因是缺乏用于评估几何表示的标准化评估方法。为了填补这一空白，我们引入了一个新的姿势估计基准，用于评估SSL几何表示，该基准要求在没有语义或姿势标签的情况下进行训练，并在语义和几何下游任务中表现出熟练度。在这个基准上，我们研究了如何增强SSL几何表示而不牺牲语义分类准确性。我们发现利用中间层表示可以将姿势估计性能提高10-20％。此外，我们引入了一种无监督轨迹正则化损失，该损失额外提高了4％的性能，并提高了在越界上的泛化能力。

    arXiv:2403.14973v1 Announce Type: cross  Abstract: Self-supervised learning (SSL) has proven effective in learning high-quality representations for various downstream tasks, with a primary focus on semantic tasks. However, its application in geometric tasks remains underexplored, partially due to the absence of a standardized evaluation method for geometric representations. To address this gap, we introduce a new pose-estimation benchmark for assessing SSL geometric representations, which demands training without semantic or pose labels and achieving proficiency in both semantic and geometric downstream tasks. On this benchmark, we study enhancing SSL geometric representations without sacrificing semantic classification accuracy. We find that leveraging mid-layer representations improves pose-estimation performance by 10-20%. Further, we introduce an unsupervised trajectory-regularization loss, which improves performance by an additional 4% and improves generalization ability on out-of
    
[^3]: 使用元提示自动化LLMs进行零样本视觉识别

    Meta-Prompting for Automating Zero-shot Visual Recognition with LLMs

    [https://arxiv.org/abs/2403.11755](https://arxiv.org/abs/2403.11755)

    提出了Meta-Prompting for Visual Recognition (MPVR)方法，通过仅需少量信息即可自动化零样本识别中的提示生成过程。

    

    大型语言模型（LLM）生成的类别特定提示的提示集成已经被证明是增强视觉语言模型（VLMs）零样本识别能力的有效方法。为了获得这些类别特定提示，现有方法依赖于手工为LLMs设计提示，以生成下游任务的VLM提示。然而，这需要手动编写这些任务特定提示，而且它们可能无法涵盖与感兴趣类别相关的各种视觉概念和任务特定风格。为了有效地将人类排除在循环之外，并完全自动化零样本识别的提示生成过程，我们提出了用于视觉识别的元提示（MPVR）。仅以目标任务的少量自然语言描述形式以及一系列相关类别标签作为输入，MPVR自动产生一个多样化的类别提示集。

    arXiv:2403.11755v1 Announce Type: cross  Abstract: Prompt ensembling of Large Language Model (LLM) generated category-specific prompts has emerged as an effective method to enhance zero-shot recognition ability of Vision-Language Models (VLMs). To obtain these category-specific prompts, the present methods rely on hand-crafting the prompts to the LLMs for generating VLM prompts for the downstream tasks. However, this requires manually composing these task-specific prompts and still, they might not cover the diverse set of visual concepts and task-specific styles associated with the categories of interest. To effectively take humans out of the loop and completely automate the prompt generation process for zero-shot recognition, we propose Meta-Prompting for Visual Recognition (MPVR). Taking as input only minimal information about the target task, in the form of its short natural language description, and a list of associated class labels, MPVR automatically produces a diverse set of cat
    
[^4]: Curriculum Learning用于从头开始的深度学习折射光学

    Curriculum Learning for ab initio Deep Learned Refractive Optics

    [https://arxiv.org/abs/2302.01089](https://arxiv.org/abs/2302.01089)

    通过Curriculum Learning，提出了一种DeepLens设计方法，能够从随机初始化表面从头开始学习复合透镜的光学设计，克服了对良好初始设计的需求，并实现了自动设计经典成像透镜和大视场延伸景深计算透镜。

    

    深度光学优化最近被提出作为使用输出图像作为目标的计算成像系统设计的新范式。然而，它目前被限制于单个元素（如衍射光学元件（DOE）或金属透镜）构成的简单光学系统，或者从良好的初始设计微调复合透镜。在这里，我们提出了一种基于Curriculum Learning的DeepLens设计方法，能够从随机初始化表面从头开始学习复合透镜的光学设计，而无需人为干预，因此克服了对良好初始设计的需求。我们通过完全自动设计经典成像透镜和一个类似手机风格的大视场延伸景深计算透镜，验证了我们方法的有效性，具有高度非球面曲面和短后焦距。

    arXiv:2302.01089v3 Announce Type: replace-cross  Abstract: Deep optical optimization has recently emerged as a new paradigm for designing computational imaging systems using only the output image as the objective. However, it has been limited to either simple optical systems consisting of a single element such as a diffractive optical element (DOE) or metalens, or the fine-tuning of compound lenses from good initial designs. Here we present a DeepLens design method based on curriculum learning, which is able to learn optical designs of compound lenses ab initio from randomly initialized surfaces without human intervention, therefore overcoming the need for a good initial design. We demonstrate the effectiveness of our approach by fully automatically designing both classical imaging lenses and a large field-of-view extended depth-of-field computational lens in a cellphone-style form factor, with highly aspheric surfaces and a short back focal length.
    
[^5]: 一种用于异常检测的综合增强框架

    A Comprehensive Augmentation Framework for Anomaly Detection. (arXiv:2308.15068v1 [cs.AI])

    [http://arxiv.org/abs/2308.15068](http://arxiv.org/abs/2308.15068)

    本文提出了一种用于异常检测的综合增强框架，该框架通过选择性地利用适当的组合，分析并压缩模拟异常的关键特征，与基于重构的方法相结合，并采用分割训练策略，能够在MVTec异常检测数据集上优于以前的最先进方法。

    

    数据增强方法通常被整合到异常检测模型的训练中。以往的方法主要集中在复制真实世界的异常或增加多样性，而没有考虑到异常的标准在不同类别之间存在差异，这可能导致训练分布的偏差。本文分析了对重构网络训练有贡献的模拟异常的关键特征，并将其压缩成几种方法，从而通过选择性地使用适当的组合来创建一个综合框架。此外，将这个框架与基于重构的方法相结合，并同时提出了一种分割训练策略，既减轻过拟合问题，又避免对重构过程引入干扰。在MVTec异常检测数据集上进行的评估表明，我们的方法在性能上优于以前的最先进方法，特别是在目标相关指标方面。

    Data augmentation methods are commonly integrated into the training of anomaly detection models. Previous approaches have primarily focused on replicating real-world anomalies or enhancing diversity, without considering that the standard of anomaly varies across different classes, potentially leading to a biased training distribution.This paper analyzes crucial traits of simulated anomalies that contribute to the training of reconstructive networks and condenses them into several methods, thus creating a comprehensive framework by selectively utilizing appropriate combinations.Furthermore, we integrate this framework with a reconstruction-based approach and concurrently propose a split training strategy that alleviates the issue of overfitting while avoiding introducing interference to the reconstruction process. The evaluations conducted on the MVTec anomaly detection dataset demonstrate that our method outperforms the previous state-of-the-art approach, particularly in terms of objec
    
[^6]: 剪枝与低秩参数高效微调相遇

    Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning. (arXiv:2305.18403v1 [cs.LG])

    [http://arxiv.org/abs/2305.18403](http://arxiv.org/abs/2305.18403)

    本论文提出了一种名为LoRAPrune的框架，可以高效微调和部署大型预训练模型，通过利用低秩自适应的值和梯度来设计PEFT感知的剪枝准则，并提出了一个迭代剪枝过程来去除冗余参数，实验结果表明与最先进的方法相比，可以显著降低模型大小和推理时间，同时保持竞争性的准确性。

    

    大型预训练模型（LPM）在各种任务中表现出卓越的性能。虽然出现了参数高效微调（PEFT）来便宜地微调这些大型模型用于下游任务，但它们的部署仍然受到巨大的模型规模和计算成本的制约。神经网络剪枝通过删除冗余参数来提供模型压缩的解决方案，但大多数现有方法依赖于计算参数梯度。然而，对于LPM而言，获得梯度是计算上禁止的，这需要探索替代方法。为此，我们提出了一种用于LPM高效微调和部署的统一框架，称为LoRAPrune。我们首先设计了一个PEFT感知的剪枝准则，该准则利用了低秩自适应（LoRA）的值和梯度，而不是预训练参数的梯度进行重要性评估。然后，我们提出了一个迭代剪枝过程来基于剪枝准则去除冗余参数。各种基准数据集上的实验结果表明，与最先进的方法相比，我们的框架可以显著降低模型大小和推理时间，同时保持竞争性的准确性。

    Large pre-trained models (LPMs), such as LLaMA and ViT-G, have shown exceptional performance across various tasks. Although parameter-efficient fine-tuning (PEFT) has emerged to cheaply fine-tune these large models on downstream tasks, their deployment is still hindered by the vast model scale and computational costs. Neural network pruning offers a solution for model compression by removing redundant parameters, but most existing methods rely on computing parameter gradients. However, obtaining the gradients is computationally prohibitive for LPMs, which necessitates the exploration of alternative approaches. To this end, we propose a unified framework for efficient fine-tuning and deployment of LPMs, termed LoRAPrune. We first design a PEFT-aware pruning criterion, which utilizes the values and gradients of Low-Rank Adaption (LoRA), rather than the gradients of pre-trained parameters for importance estimation. We then propose an iterative pruning procedure to remove redundant paramet
    
[^7]: 从大型矿石中提炼黄金: 基于关键样本选择的高效数据集蒸馏

    Distill Gold from Massive Ores: Efficient Dataset Distillation via Critical Samples Selection. (arXiv:2305.18381v1 [cs.LG])

    [http://arxiv.org/abs/2305.18381](http://arxiv.org/abs/2305.18381)

    研究提出了一种基于选择最有价值的样本的方法，以扩展现有的蒸馏算法，从而更好地利用训练样本，显著降低训练成本，拓展对更大更多元化数据集的数据集蒸馏，并持续提高性能。

    

    数据效率学习近年来备受关注，特别是对于拥有大量多模型的现在，数据集蒸馏可以成为有效的解决方案。然而，数据集蒸馏过程本身仍然非常低效。在本文中，我们首先引用信息理论来建模蒸馏问题，观察到数据集蒸馏中存在严重的数据冗余，我们提出了一种方法来扩展现有的蒸馏算法，以便通过选择最有价值的样本来更好地利用这些训练样本。我们进一步对样本选择进行全面分析，并验证了其优化过程。这种新策略能够显著减少训练成本，扩大现有算法范围以对更庞大和多元化的数据集进行数据集蒸馏，例如，在某些情况下，只需要0.04％的训练数据就足以保持可比的蒸馏效果。此外，我们的策略能够持续提高性能，其贡献可能为蒸馏过程的动力学开辟新的分析方法。

    Data-efficient learning has drawn significant attention, especially given the current trend of large multi-modal models, where dataset distillation can be an effective solution. However, the dataset distillation process itself is still very inefficient. In this work, we model the distillation problem with reference to information theory. Observing that severe data redundancy exists in dataset distillation, we argue to put more emphasis on the utility of the training samples. We propose a family of methods to exploit the most valuable samples, which is validated by our comprehensive analysis of the optimal data selection. The new strategy significantly reduces the training cost and extends a variety of existing distillation algorithms to larger and more diversified datasets, e.g. in some cases only 0.04% training data is sufficient for comparable distillation performance. Moreover, our strategy consistently enhances the performance, which may open up new analyses on the dynamics of dist
    
[^8]: 柿子政治的面孔：使用机器学习比较政治领袖面部情感表达的差异

    The Face of Populism: Examining Differences in Facial Emotional Expressions of Political Leaders Using Machine Learning. (arXiv:2304.09914v1 [cs.CY])

    [http://arxiv.org/abs/2304.09914](http://arxiv.org/abs/2304.09914)

    本文使用机器学习的算法分析了来自15个不同国家的220个政治领袖的YouTube视频，总结了政治领袖面部情感表达的差异。

    

    网络媒体已经彻底改变了政治信息在全球范围内的传播和消费方式，这种转变促使政治人物采取新的策略来捕捉和保持选民的注意力。这些策略往往依赖于情感说服和吸引。随着虚拟空间中视觉内容越来越普遍，很多政治沟通也被标志着唤起情感的视频内容和图像。本文提供了一种新的分析方法。我们将基于现有训练好的卷积神经网络架构提供的Python库fer，应用一种基于深度学习的计算机视觉算法，对描绘来自15个不同国家的政治领袖的220个YouTube视频样本进行分析。该算法返回情绪分数，每一帧都代表6种情绪状态（愤怒，厌恶，恐惧，快乐，悲伤和惊讶）和一个中性表情。

    Online media has revolutionized the way political information is disseminated and consumed on a global scale, and this shift has compelled political figures to adopt new strategies of capturing and retaining voter attention. These strategies often rely on emotional persuasion and appeal, and as visual content becomes increasingly prevalent in virtual space, much of political communication too has come to be marked by evocative video content and imagery. The present paper offers a novel approach to analyzing material of this kind. We apply a deep-learning-based computer-vision algorithm to a sample of 220 YouTube videos depicting political leaders from 15 different countries, which is based on an existing trained convolutional neural network architecture provided by the Python library fer. The algorithm returns emotion scores representing the relative presence of 6 emotional states (anger, disgust, fear, happiness, sadness, and surprise) and a neutral expression for each frame of the pr
    
[^9]: ASR: 像注意力一样的结构再参数化

    ASR: Attention-alike Structural Re-parameterization. (arXiv:2304.06345v1 [cs.CV])

    [http://arxiv.org/abs/2304.06345](http://arxiv.org/abs/2304.06345)

    该论文提出的ASR技术是一种新颖的深度学习技术，通过等效参数转换实现不同网络体系结构之间的互转。和现有的SRP方法相比，ASR可以成功考虑自注意模块，实现推理期间的性能提升，并在工业和实际应用中具有巨大潜力。

    

    结构再参数化（SRP）技术是一种新颖的深度学习技术，通过等效参数转换实现不同网络体系结构之间的互转。该技术使得在推理过程中通过这些转换减少性能提升的新增代价，例如参数大小和推理时间，因此SRP在工业和实际应用中具有巨大潜力。现有的SRP方法已成功考虑了许多常用的架构，例如归一化、池化方法、多分支卷积等。然而，广泛使用的自注意模块由于在推理期间通常以乘法方式作用于骨干网络并且模块的输出在推理时依赖于输入，所以无法直接实现SRP，而这限制了SRP的应用场景。在本文中，我们从统计角度进行了广泛的实验，并发现...

    The structural re-parameterization (SRP) technique is a novel deep learning technique that achieves interconversion between different network architectures through equivalent parameter transformations. This technique enables the mitigation of the extra costs for performance improvement during training, such as parameter size and inference time, through these transformations during inference, and therefore SRP has great potential for industrial and practical applications. The existing SRP methods have successfully considered many commonly used architectures, such as normalizations, pooling methods, multi-branch convolution. However, the widely used self-attention modules cannot be directly implemented by SRP due to these modules usually act on the backbone network in a multiplicative manner and the modules' output is input-dependent during inference, which limits the application scenarios of SRP. In this paper, we conduct extensive experiments from a statistical perspective and discover
    
[^10]: 生成对抗网络的指纹识别技术

    Fingerprinting Generative Adversarial Networks. (arXiv:2106.11760v3 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2106.11760](http://arxiv.org/abs/2106.11760)

    本文提出了一种保护GAN知识产权的指纹识别方案，通过生成指纹样本并嵌入到分类器中进行版权验证，解决了前一种对分类模型的指纹识别方法在简单转移至GAN时遇到的隐蔽性和鲁棒性瓶颈，具有实际保护现代GAN模型的可行性。

    

    生成对抗网络（GANs）已经广泛应用于各种应用场景。由于商业GAN的生产需要大量的计算和人力资源，因此迫切需要版权保护。本文提出了一种用于保护GAN知识产权的指纹识别方案。我们突破了前一种对分类模型的指纹识别方法在简单转移至GAN时所遇到的隐蔽性和鲁棒性瓶颈。具体来说，我们创造性地从目标GAN和分类器构建一个复合深度学习模型。然后，我们从这个复合模型中产生指纹样本，并将其嵌入到分类器中，以进行有效的版权验证。这种方案启发了一些具体的方法，以实际保护现代GAN模型。理论分析证明了这些方法可以满足知识产权保护所需要的不同安全要求。我们还进行了实验来证明该方案的功效。

    Generative Adversarial Networks (GANs) have been widely used in various application scenarios. Since the production of a commercial GAN requires substantial computational and human resources, the copyright protection of GANs is urgently needed. In this paper, we present the first fingerprinting scheme for the Intellectual Property (IP) protection of GANs. We break through the stealthiness and robustness bottlenecks suffered by previous fingerprinting methods for classification models being naively transferred to GANs. Specifically, we innovatively construct a composite deep learning model from the target GAN and a classifier. Then we generate fingerprint samples from this composite model, and embed them in the classifier for effective ownership verification. This scheme inspires some concrete methodologies to practically protect the modern GAN models. Theoretical analysis proves that these methods can satisfy different security requirements necessary for IP protection. We also conduct 
    

