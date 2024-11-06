# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Attention-based Class-Conditioned Alignment for Multi-Source Domain Adaptive Object Detection](https://arxiv.org/abs/2403.09918) | 提出了一种基于注意力的类别条件对齐方案，用于多源领域自适应目标检测，在跨领域对齐每个对象类别的实例。 |
| [^2] | [Optimizing Negative Prompts for Enhanced Aesthetics and Fidelity in Text-To-Image Generation](https://arxiv.org/abs/2403.07605) | 提出NegOpt方法，通过监督微调和强化学习优化负面提示的生成，显著提高图像生成质量，超越其他方法并构建了负面提示数据集。 |
| [^3] | [Multi-modal preference alignment remedies regression of visual instruction tuning on language model](https://arxiv.org/abs/2402.10884) | 通过收集轻量级VQA偏好数据集并使用Direct Preference Optimization，我们能够在语言模型的指导能力上取得显著提升，在小规模数据下比其他方法实现了更高的分数。 |
| [^4] | [Robustly overfitting latents for flexible neural image compression](https://arxiv.org/abs/2401.17789) | 这项研究提出了一种鲁棒的过拟合潜变量方法来改进神经图像压缩模型，通过使用SGA+，可以显著提高性能并减少对超参数选择的敏感性。 |
| [^5] | [S-JEA: Stacked Joint Embedding Architectures for Self-Supervised Visual Representation Learning.](http://arxiv.org/abs/2305.11701) | 本文提出了使用堆叠联合嵌入结构学习高度可分离的分层语义表示，显示出更明显的语义概念子类，并且与传统方法相似。 |
| [^6] | [Diversifying Deep Ensembles: A Saliency Map Approach for Enhanced OOD Detection, Calibration, and Accuracy.](http://arxiv.org/abs/2305.11616) | 这项研究提出了一种使用显著性图来促进深度集成多样性的方法，用于改善OOD检测、校准和准确性，能够优于传统的集成技术，并在OpenOOD基准测试上证明了其有效性。 |
| [^7] | [Provably Learning Diverse Features in Multi-View Data with Midpoint Mixup.](http://arxiv.org/abs/2210.13512) | 本文提出了一种基于中点 Mixup 的多视角数据学习方法，相比于传统经验风险最小化方法能够更好地学习每个类别的所有特征，具有更好的泛化和鲁棒性。 |

# 详细

[^1]: 基于注意力的多源领域自适应目标检测的类别条件对齐

    Attention-based Class-Conditioned Alignment for Multi-Source Domain Adaptive Object Detection

    [https://arxiv.org/abs/2403.09918](https://arxiv.org/abs/2403.09918)

    提出了一种基于注意力的类别条件对齐方案，用于多源领域自适应目标检测，在跨领域对齐每个对象类别的实例。

    

    目标检测（OD）的领域自适应方法致力于通过促进源域和目标域之间的特征对齐来缓解分布转移的影响。多源领域自适应（MSDA）允许利用多个带注释的源数据集和未标记的目标数据来提高检测模型的准确性和鲁棒性。大多数最先进的OD MSDA方法以一种与类别无关的方式执行特征对齐。最近提出的基于原型的方法提出了一种按类别对齐的方法，但由于嘈杂的伪标签而导致错误积累，这可能会对不平衡数据的自适应产生负面影响。为克服这些限制，我们提出了一种基于注意力的类别条件对齐方案，用于MSDA，该方案在跨领域对齐每个对象类别的实例。

    arXiv:2403.09918v1 Announce Type: cross  Abstract: Domain adaptation methods for object detection (OD) strive to mitigate the impact of distribution shifts by promoting feature alignment across source and target domains. Multi-source domain adaptation (MSDA) allows leveraging multiple annotated source datasets, and unlabeled target data to improve the accuracy and robustness of the detection model. Most state-of-the-art MSDA methods for OD perform feature alignment in a class-agnostic manner. This is challenging since the objects have unique modal information due to variations in object appearance across domains. A recent prototype-based approach proposed a class-wise alignment, yet it suffers from error accumulation due to noisy pseudo-labels which can negatively affect adaptation with imbalanced data. To overcome these limitations, we propose an attention-based class-conditioned alignment scheme for MSDA that aligns instances of each object category across domains. In particular, an 
    
[^2]: 优化负面提示以增强文本到图像生成中的美学和保真度

    Optimizing Negative Prompts for Enhanced Aesthetics and Fidelity in Text-To-Image Generation

    [https://arxiv.org/abs/2403.07605](https://arxiv.org/abs/2403.07605)

    提出NegOpt方法，通过监督微调和强化学习优化负面提示的生成，显著提高图像生成质量，超越其他方法并构建了负面提示数据集。

    

    在文本到图像生成中，使用描述不良图像特征的负面提示可以显著提高图像质量。然而，生成良好的负面提示是一项手工而繁琐的工作。为了解决这个问题，我们提出了NegOpt，一种新颖的方法，通过监督微调和强化学习来优化负面提示生成，从而增强图像生成。我们的综合方法相对于其他方法大幅提高了25%的Inception Score，并超越了来自测试集的标准负面提示。此外，使用NegOpt，我们可以有选择地优化对我们最重要的指标。最后，我们构建了负面提示数据集Negative Prompts DB。

    arXiv:2403.07605v1 Announce Type: cross  Abstract: In text-to-image generation, using negative prompts, which describe undesirable image characteristics, can significantly boost image quality. However, producing good negative prompts is manual and tedious. To address this, we propose NegOpt, a novel method for optimizing negative prompt generation toward enhanced image generation, using supervised fine-tuning and reinforcement learning. Our combined approach results in a substantial increase of 25% in Inception Score compared to other approaches and surpasses ground-truth negative prompts from the test set. Furthermore, with NegOpt we can preferentially optimize the metrics most important to us. Finally, we construct Negative Prompts DB, a dataset of negative prompts.
    
[^3]: 多模式偏好对齐修复了语言模型在视觉指令调整上的回归

    Multi-modal preference alignment remedies regression of visual instruction tuning on language model

    [https://arxiv.org/abs/2402.10884](https://arxiv.org/abs/2402.10884)

    通过收集轻量级VQA偏好数据集并使用Direct Preference Optimization，我们能够在语言模型的指导能力上取得显著提升，在小规模数据下比其他方法实现了更高的分数。

    

    在实际应用中，多模式大型语言模型（MLLMs）被期望能够支持图像和文本模态的交换式多轮查询。然而，当前使用视觉问题回答（VQA）数据集训练的MLLMs可能会出现退化，因为VQA数据集缺乏原始文本指令数据集的多样性和复杂性，后者是底层语言模型训练的数据集。为了解决这一具有挑战性的退化问题，我们首先收集了一个轻量级（6k条记录）的VQA偏好数据集，其中答案由Gemini以细粒度方式注释了5个质量指标，然后研究了标准的监督微调、拒绝抽样、直接偏好优化（DPO）和SteerLM。我们的研究结果表明，通过DPO，我们能够超越语言模型的指导能力，实现了6.73的MT-Bench分数，而Vicuna的6.57和LLaVA的5.99，尽管数据规模较小。

    arXiv:2402.10884v1 Announce Type: cross  Abstract: In production, multi-modal large language models (MLLMs) are expected to support multi-turn queries of interchanging image and text modalities. However, the current MLLMs trained with visual-question-answering (VQA) datasets could suffer from degradation, as VQA datasets lack the diversity and complexity of the original text instruction datasets which the underlying language model had been trained with. To address this challenging degradation, we first collect a lightweight (6k entries) VQA preference dataset where answers were annotated by Gemini for 5 quality metrics in a granular fashion, and investigate standard Supervised Fine-tuning, rejection sampling, Direct Preference Optimization (DPO), and SteerLM. Our findings indicate that the with DPO we are able to surpass instruction-following capabilities of the language model, achieving a 6.73 score on MT-Bench, compared to Vicuna's 6.57 and LLaVA's 5.99 despite small data scale. This
    
[^4]: 弹性神经图像压缩中的鲁棒过拟合潜变量

    Robustly overfitting latents for flexible neural image compression

    [https://arxiv.org/abs/2401.17789](https://arxiv.org/abs/2401.17789)

    这项研究提出了一种鲁棒的过拟合潜变量方法来改进神经图像压缩模型，通过使用SGA+，可以显著提高性能并减少对超参数选择的敏感性。

    

    神经图像压缩取得了很大的进展。最先进的模型基于变分自编码器，胜过了传统模型。神经压缩模型学会将图像编码为量化的潜变量表示，然后将其高效地发送给解码器，解码器再将量化的潜变量解码为重建图像。虽然这些模型在实践中取得了成功，但由于优化不完美以及编码器和解码器容量的限制，它们导致了次优结果。最近的研究表明，如何利用随机Gumbel退火（SGA）来改进预训练的神经图像压缩模型的潜变量。我们通过引入SGA+扩展了这个想法，SGA+包含了三种不同的方法，这些方法都建立在SGA的基础上。此外，我们对我们提出的方法进行了详细分析，展示了它们如何改进性能，并且证明它们对超参数选择不敏感。此外，我们还展示了如何将每个方法扩展到三个而不是两个。

    Neural image compression has made a great deal of progress. State-of-the-art models are based on variational autoencoders and are outperforming classical models. Neural compression models learn to encode an image into a quantized latent representation that can be efficiently sent to the decoder, which decodes the quantized latent into a reconstructed image. While these models have proven successful in practice, they lead to sub-optimal results due to imperfect optimization and limitations in the encoder and decoder capacity. Recent work shows how to use stochastic Gumbel annealing (SGA) to refine the latents of pre-trained neural image compression models. We extend this idea by introducing SGA+, which contains three different methods that build upon SGA. Further, we give a detailed analysis of our proposed methods, show how they improve performance, and show that they are less sensitive to hyperparameter choices. Besides, we show how each method can be extended to three- instead of two
    
[^5]: S-JEA: 堆叠联合嵌入结构用于自监督视觉表示学习

    S-JEA: Stacked Joint Embedding Architectures for Self-Supervised Visual Representation Learning. (arXiv:2305.11701v1 [cs.CV])

    [http://arxiv.org/abs/2305.11701](http://arxiv.org/abs/2305.11701)

    本文提出了使用堆叠联合嵌入结构学习高度可分离的分层语义表示，显示出更明显的语义概念子类，并且与传统方法相似。

    

    自监督学习作为学习图像表示的基本范式，近年来已经在各种任务中展示了高度的实证成功。然而，大多数自监督学习方法未能学习到捕获分层语义概念的嵌入，这些概念是可分离和可解释的。本文旨在通过堆叠联合嵌入结构（JEA）来学习高度可分离的分层语义表示，其中较高级别的JEA使用较低级别JEA的表示结果作为输入。这导致表示空间表现出更明显的语义概念子类（如车辆的型号和颜色）在较高级别的JEA中。我们实验性地展示了堆叠JEA的表示与传统JEA相似，并展示了表示空间以验证语义分层结构。

    The recent emergence of Self-Supervised Learning (SSL) as a fundamental paradigm for learning image representations has, and continues to, demonstrate high empirical success in a variety of tasks. However, most SSL approaches fail to learn embeddings that capture hierarchical semantic concepts that are separable and interpretable. In this work, we aim to learn highly separable semantic hierarchical representations by stacking Joint Embedding Architectures (JEA) where higher-level JEAs are input with representations of lower-level JEA. This results in a representation space that exhibits distinct sub-categories of semantic concepts (e.g., model and colour of vehicles) in higher-level JEAs. We empirically show that representations from stacked JEA perform on a similar level as traditional JEA with comparative parameter counts and visualise the representation spaces to validate the semantic hierarchies.
    
[^6]: 多样化深度集成：一种使用显著性图的方法以增强OOD检测、校准和准确性

    Diversifying Deep Ensembles: A Saliency Map Approach for Enhanced OOD Detection, Calibration, and Accuracy. (arXiv:2305.11616v1 [cs.CV])

    [http://arxiv.org/abs/2305.11616](http://arxiv.org/abs/2305.11616)

    这项研究提出了一种使用显著性图来促进深度集成多样性的方法，用于改善OOD检测、校准和准确性，能够优于传统的集成技术，并在OpenOOD基准测试上证明了其有效性。

    

    深度集成在分类和 OOD 检测方面取得了最先进的成果；然而，由于集成中学习的模式的同质性，它们的效果仍然有限。为了克服这一挑战，本研究引入了一种促进集成成员之间多样性的新方法，该方法利用显著性图。通过整合显著性图多样化，我们的方法在多个分类和OOD检测任务中优于传统的集成技术，同时也提高了校准性。在已建立的OpenOOD基准测试上的实验凸显了我们的方法在实际应用中的潜力。

    Deep ensembles achieved state-of-the-art results in classification and out-of-distribution (OOD) detection; however, their effectiveness remains limited due to the homogeneity of learned patterns within the ensemble. To overcome this challenge, our study introduces a novel approach that promotes diversity among ensemble members by leveraging saliency maps. By incorporating saliency map diversification, our method outperforms conventional ensemble techniques in multiple classification and OOD detection tasks, while also improving calibration. Experiments on well-established OpenOOD benchmarks highlight the potential of our method in practical applications.
    
[^7]: 用中点 Mixup 在多视角数据中可证明学习多元特征

    Provably Learning Diverse Features in Multi-View Data with Midpoint Mixup. (arXiv:2210.13512v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.13512](http://arxiv.org/abs/2210.13512)

    本文提出了一种基于中点 Mixup 的多视角数据学习方法，相比于传统经验风险最小化方法能够更好地学习每个类别的所有特征，具有更好的泛化和鲁棒性。

    

    Mixup 是一种数据增强技术，依赖于使用数据点和标签的随机凸组合进行训练。近年来，Mixup 已成为训练最先进的图像分类模型的标准基元，因为它在泛化和鲁棒性方面比经验风险最小化有明显的优势。在这项工作中，我们试图从特征学习的角度解释一些这种成功。我们关注的分类问题是，每个类别可能具有多个相关特征（或视图），可用于正确预测类别。我们的主要理论结果表明，在具有每类两个特征的一类非平凡数据分布中，使用经验风险最小化训练 2 层卷积网络可能会导致几乎所有类别只学习一个特征，而使用 Mixup 的特定实例进行训练可以成功地学习每个类别的两个特征。

    Mixup is a data augmentation technique that relies on training using random convex combinations of data points and their labels. In recent years, Mixup has become a standard primitive used in the training of state-of-the-art image classification models due to its demonstrated benefits over empirical risk minimization with regards to generalization and robustness. In this work, we try to explain some of this success from a feature learning perspective. We focus our attention on classification problems in which each class may have multiple associated features (or views) that can be used to predict the class correctly. Our main theoretical results demonstrate that, for a non-trivial class of data distributions with two features per class, training a 2-layer convolutional network using empirical risk minimization can lead to learning only one feature for almost all classes while training with a specific instantiation of Mixup succeeds in learning both features for every class. We also show
    

