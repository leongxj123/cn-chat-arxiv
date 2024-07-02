# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Training morphological neural networks with gradient descent: some theoretical insights](https://arxiv.org/abs/2403.12975) | 形态神经网络的训练存在挑战，本文通过使用基于梯度下降的优化算法，探讨了基于微分方法和反向传播对形态网络的潜力和局限性，提供了关于初始化和学习率的理论指导。 |
| [^2] | [Unleashing the Power of Meta-tuning for Few-shot Generalization Through Sparse Interpolated Experts](https://arxiv.org/abs/2403.08477) | 本文提出了一种名为Sparse MetA-Tuning（SMAT）的方法，通过灵感来自稀疏专家混合方法，成功克服了域外任务敏感性，实现了增强视觉基础模型转移能力的目标。 |
| [^3] | [Challenging Forgets: Unveiling the Worst-Case Forget Sets in Machine Unlearning](https://arxiv.org/abs/2403.07362) | 该论文从对抗的角度提出了一种新的机器遗忘评估方法，通过确定最具挑战性的数据子集，即最坏情况遗忘集，来增强对影响擦除的挑战。 |
| [^4] | [Recovering the Pre-Fine-Tuning Weights of Generative Models](https://arxiv.org/abs/2402.10208) | 该论文提出了一种恢复生成模型预微调权重的方法，通过少量低秩微调模型可以恢复准确的预微调权重，利用这个新漏洞攻击大规模模型。 |
| [^5] | [AdaTreeFormer: Few Shot Domain Adaptation for Tree Counting from a Single High-Resolution Image](https://arxiv.org/abs/2402.02956) | AdaTreeFormer是一种从源领域学习并适应只有有限数量标注树木的目标领域的框架，利用一个共享的编码器和分层特征提取方案，实现了树木计数的少样本领域自适应。 |
| [^6] | [Exploring the Design Space of Diffusion Autoencoders for Face Morphing.](http://arxiv.org/abs/2310.09484) | 这项研究探索了面向人脸变形的扩散自编码器的设计空间，研究了采样算法、逆向DDIM求解器和部分采样的方法。 |
| [^7] | [Fine-tuning can cripple your foundation model; preserving features may be the solution.](http://arxiv.org/abs/2308.13320) | 在微调过程中，基础模型可能会遗忘概念，我们提出了一种名为LDIFS的方法，用于解决这个问题，该方法在实验证明效果显著。 |
| [^8] | [Adaptive Regularization for Class-Incremental Learning.](http://arxiv.org/abs/2303.13113) | 本文研究了适应性正则化在类增量学习中的应用，通过根据任务复杂度动态调整正则化强度，在学习新类别同时防止遗忘之前学习的类别。实验表明适应性正则化可以实现更加准确和不易遗忘的视觉增量学习。 |
| [^9] | [Patch-Token Aligned Bayesian Prompt Learning for Vision-Language Models.](http://arxiv.org/abs/2303.09100) | 本文提出了一种基于贝叶斯概率的视觉语言模型提示学习方法，通过将提示标记推向忠实捕捉标签特定的视觉概念，而不是过度拟合训练类别，解决了现有提示工程的问题。在各种视觉语言任务上的广泛实验表明，该方法优于现有的最先进模型。 |

# 详细

[^1]: 用梯度下降训练形态神经网络：一些理论见解

    Training morphological neural networks with gradient descent: some theoretical insights

    [https://arxiv.org/abs/2403.12975](https://arxiv.org/abs/2403.12975)

    形态神经网络的训练存在挑战，本文通过使用基于梯度下降的优化算法，探讨了基于微分方法和反向传播对形态网络的潜力和局限性，提供了关于初始化和学习率的理论指导。

    

    形态神经网络或层可以成为提升数学形态学进展的强大工具，无论是在理论方面，如完整格算子的表示，还是在图像处理流程的开发方面。然而，当这些架构包含多层形态学时，至少在使用基于梯度下降的优化算法的流行机器学习框架内，这些网络很难进行训练。在本文中，我们探讨了基于微分方法和反向传播应用于形态网络的潜力和局限性，考虑到Bouligand导数的非光滑优化概念。我们提供了见解和首个理论指南，特别是关于初始化和学习率。

    arXiv:2403.12975v1 Announce Type: cross  Abstract: Morphological neural networks, or layers, can be a powerful tool to boost the progress in mathematical morphology, either on theoretical aspects such as the representation of complete lattice operators, or in the development of image processing pipelines. However, these architectures turn out to be difficult to train when they count more than a few morphological layers, at least within popular machine learning frameworks which use gradient descent based optimization algorithms. In this paper we investigate the potential and limitations of differentiation based approaches and back-propagation applied to morphological networks, in light of the non-smooth optimization concept of Bouligand derivative. We provide insights and first theoretical guidelines, in particular regarding initialization and learning rates.
    
[^2]: 通过稀疏插值专家释放元调整的力量，用于少样本泛化

    Unleashing the Power of Meta-tuning for Few-shot Generalization Through Sparse Interpolated Experts

    [https://arxiv.org/abs/2403.08477](https://arxiv.org/abs/2403.08477)

    本文提出了一种名为Sparse MetA-Tuning（SMAT）的方法，通过灵感来自稀疏专家混合方法，成功克服了域外任务敏感性，实现了增强视觉基础模型转移能力的目标。

    

    传统智慧建议参数高效的微调基础模型，是视觉迁移学习的最先进方法，取代了诸如元学习之类的丰富文献。为了兼顾两者的利益，元调整引入了基础模型的随后优化阶段，但迄今只展现了有限的成功，关键地在域外（OOD）任务上表现不佳。本文介绍了一种灵感来自稀疏专家混合方法的 Sparse MetA-Tuning（SMAT）方法，它经过训练以自动地为每个任务隔离预训练参数子集以进行元调整。SMAT成功克服了OOD敏感性，并实现了增强视觉基础模型转移能力的承诺。我们在Meta-Dataset与额外的OO挑战组合上建立了新的最先进结果。

    arXiv:2403.08477v1 Announce Type: cross  Abstract: Conventional wisdom suggests parameter-efficient fine-tuning of foundation models as the state-of-the-art method for transfer learning in vision, replacing the rich literature of alternatives such as meta-learning. In trying to harness the best of both worlds, meta-tuning introduces a subsequent optimization stage of foundation models but has so far only shown limited success and crucially tends to underperform on out-of-domain (OOD) tasks. In this paper, we introduce Sparse MetA-Tuning (SMAT), a method inspired by sparse mixture-of-experts approaches and trained to isolate subsets of pre-trained parameters automatically for meta-tuning on each task. SMAT successfully overcomes OOD sensitivity and delivers on the promise of enhancing the transfer abilities of vision foundation models beyond parameter-efficient finetuning. We establish new state-of-the-art results on a challenging combination of Meta-Dataset augmented with additional OO
    
[^3]: 挑战遗忘：揭示机器遗忘中最坏情况遗忘集

    Challenging Forgets: Unveiling the Worst-Case Forget Sets in Machine Unlearning

    [https://arxiv.org/abs/2403.07362](https://arxiv.org/abs/2403.07362)

    该论文从对抗的角度提出了一种新的机器遗忘评估方法，通过确定最具挑战性的数据子集，即最坏情况遗忘集，来增强对影响擦除的挑战。

    

    靠谱的机器学习(Machine Learning, ML)社区越来越认识到模型在训练后有选择性地“遗忘”数据点的重要性。这引出了机器遗忘(Machine Unlearning, MU)问题，旨在消除选定数据点对模型性能的影响，同时仍保持模型在遗忘后的实用性。尽管有各种MU方法来擦除数据影响，评估主要集中在随机数据遗忘上，忽视了对于真实衡量遗忘性能的数据子集选择的重要探究。为解决这一问题，我们从对抗的角度引入了一种新的MU评估视角。我们提出确定那些对影响擦除构成最大挑战的数据子集，即找出最坏情况遗忘集。利用双层优化原则，我们增强了在上层优化中的遗忘挑战。

    arXiv:2403.07362v1 Announce Type: cross  Abstract: The trustworthy machine learning (ML) community is increasingly recognizing the crucial need for models capable of selectively 'unlearning' data points after training. This leads to the problem of machine unlearning (MU), aiming to eliminate the influence of chosen data points on model performance, while still maintaining the model's utility post-unlearning. Despite various MU methods for data influence erasure, evaluations have largely focused on random data forgetting, ignoring the vital inquiry into which subset should be chosen to truly gauge the authenticity of unlearning performance. To tackle this issue, we introduce a new evaluative angle for MU from an adversarial viewpoint. We propose identifying the data subset that presents the most significant challenge for influence erasure, i.e., pinpointing the worst-case forget set. Utilizing a bi-level optimization principle, we amplify unlearning challenges at the upper optimization 
    
[^4]: 恢复生成模型的预微调权重

    Recovering the Pre-Fine-Tuning Weights of Generative Models

    [https://arxiv.org/abs/2402.10208](https://arxiv.org/abs/2402.10208)

    该论文提出了一种恢复生成模型预微调权重的方法，通过少量低秩微调模型可以恢复准确的预微调权重，利用这个新漏洞攻击大规模模型。

    

    在生成建模中，主流模式包括两个步骤：i) 在大规模但不安全的数据集上进行预训练，ii) 通过微调将预训练模型与人类价值观对齐。这种做法被认为是安全的，因为目前没有一种方法可以恢复不安全的预微调模型权重。本文证明了这种假设通常是错误的。具体而言，我们提出了一种称为谱反调的方法，可以使用少量低秩（LoRA）微调模型恢复预微调模型的权重。与先前试图恢复预微调能力的攻击不同，我们的方法旨在恢复精确的预微调权重。我们的方法利用了这个新的对大规模模型的漏洞，例如个性化的稳定扩散和对齐的Mistral模型。

    arXiv:2402.10208v1 Announce Type: cross  Abstract: The dominant paradigm in generative modeling consists of two steps: i) pre-training on a large-scale but unsafe dataset, ii) aligning the pre-trained model with human values via fine-tuning. This practice is considered safe, as no current method can recover the unsafe, pre-fine-tuning model weights. In this paper, we demonstrate that this assumption is often false. Concretely, we present Spectral DeTuning, a method that can recover the weights of the pre-fine-tuning model using a few low-rank (LoRA) fine-tuned models. In contrast to previous attacks that attempt to recover pre-fine-tuning capabilities, our method aims to recover the exact pre-fine-tuning weights. Our approach exploits this new vulnerability against large-scale models such as a personalized Stable Diffusion and an aligned Mistral.
    
[^5]: AdaTreeFormer: 从一张高分辨率图像中进行树木计数的少样本领域自适应

    AdaTreeFormer: Few Shot Domain Adaptation for Tree Counting from a Single High-Resolution Image

    [https://arxiv.org/abs/2402.02956](https://arxiv.org/abs/2402.02956)

    AdaTreeFormer是一种从源领域学习并适应只有有限数量标注树木的目标领域的框架，利用一个共享的编码器和分层特征提取方案，实现了树木计数的少样本领域自适应。

    

    仅使用一张航空或卫星图像来估计和计数树木密度是摄影测量和遥感领域中一项困难的任务。然而，它在森林管理中起着至关重要的作用。不同地形上各种各样的树木种类严重阻碍了树木计数模型的良好表现。本文旨在提出一个从具有足够标注树木的源领域学习并适应只有有限数量标注树木的目标领域的框架。我们的方法称为AdaTreeFormer，包含一个共享的编码器和一个分层特征提取方案，用于从源领域和目标领域中提取稳健的特征。它还包括三个子网络：两个用于分别从源领域和目标领域提取自注意力图，并一个用于提取跨领域注意力图。对于后者，引入了一种注意力适应机制，用于从不同领域中提取相关信息。

    The process of estimating and counting tree density using only a single aerial or satellite image is a difficult task in the fields of photogrammetry and remote sensing. However, it plays a crucial role in the management of forests. The huge variety of trees in varied topography severely hinders tree counting models to perform well. The purpose of this paper is to propose a framework that is learnt from the source domain with sufficient labeled trees and is adapted to the target domain with only a limited number of labeled trees. Our method, termed as AdaTreeFormer, contains one shared encoder with a hierarchical feature extraction scheme to extract robust features from the source and target domains. It also consists of three subnets: two for extracting self-domain attention maps from source and target domains respectively and one for extracting cross-domain attention maps. For the latter, an attention-to-adapt mechanism is introduced to distill relevant information from different doma
    
[^6]: 探索面向人脸变形的扩散自编码器的设计空间

    Exploring the Design Space of Diffusion Autoencoders for Face Morphing. (arXiv:2310.09484v1 [cs.CV])

    [http://arxiv.org/abs/2310.09484](http://arxiv.org/abs/2310.09484)

    这项研究探索了面向人脸变形的扩散自编码器的设计空间，研究了采样算法、逆向DDIM求解器和部分采样的方法。

    

    通过扩散自编码器创建的人脸变形是一种最近的创新，而这种方法的设计空间尚未得到充分探索。我们探索了设计空间的三个方面，即1）采样算法，2）逆向DDIM求解器，以及3）通过添加少量噪声进行部分采样。

    Face morphs created by Diffusion Autoencoders are a recent innovation and the design space of such an approach has not been well explored. We explore three axes of the design space, i.e., 1) sampling algorithms, 2) the reverse DDIM solver, and 3) partial sampling through small amounts of added noise.
    
[^7]: 微调可能削弱基础模型；保留特征可能是解决方案

    Fine-tuning can cripple your foundation model; preserving features may be the solution. (arXiv:2308.13320v1 [cs.LG])

    [http://arxiv.org/abs/2308.13320](http://arxiv.org/abs/2308.13320)

    在微调过程中，基础模型可能会遗忘概念，我们提出了一种名为LDIFS的方法，用于解决这个问题，该方法在实验证明效果显著。

    

    预训练的基础模型主要由于其巨大的容量和对从互联网上爬取的大量训练数据的暴露，享有存储关于许多现实世界概念的知识的优势。这些模型通常在下游数据集上进行微调，以产生出色的最新性能。然而，我们观察到，与预训练模型相比，微调模型在与下游任务不同的任务上识别概念的能力显著降低。这显然是不可取的，因为在首次学习这些概念时，投入了大量的时间和金钱。我们将这种不可取的现象称为“概念遗忘”，通过实验证明大多数端到端微调方法都严重受到这种副作用的影响。为此，我们还提出了一个相当简单的解决方法，即设计了一种名为LDIFS的方法。

    Pre-trained foundation models, owing primarily to their enormous capacity and exposure to vast amount of training data scraped from the internet, enjoy the advantage of storing knowledge about plenty of real-world concepts. Such models are typically fine-tuned on downstream datasets to produce remarkable state-of-the-art performances. While various fine-tuning methods have been devised and are shown to be highly effective, we observe that a fine-tuned model's ability to recognize concepts on tasks $\textit{different}$ from the downstream one is reduced significantly compared to its pre-trained counterpart. This is clearly undesirable as a huge amount of time and money went into learning those very concepts in the first place. We call this undesirable phenomenon "concept forgetting" and via experiments show that most end-to-end fine-tuning approaches suffer heavily from this side effect. To this end, we also propose a rather simple fix to this problem by designing a method called LDIFS 
    
[^8]: 适应性正则化在类增量学习中的应用

    Adaptive Regularization for Class-Incremental Learning. (arXiv:2303.13113v1 [cs.LG])

    [http://arxiv.org/abs/2303.13113](http://arxiv.org/abs/2303.13113)

    本文研究了适应性正则化在类增量学习中的应用，通过根据任务复杂度动态调整正则化强度，在学习新类别同时防止遗忘之前学习的类别。实验表明适应性正则化可以实现更加准确和不易遗忘的视觉增量学习。

    

    类增量学习是指在维持先前学习的分类准确度的同时，更新具有新类别的深度分类器。在学习新类别的同时，通过正则化神经网络权重来防止遗忘之前学习的类别是常见的方法。然而，现有的正则化方法在整个增量学习过程中使用恒定的强度，可能无法反映所遇到的任务难度的变化。因此，本研究探讨了适应性正则化在类增量学习中的必要性，该方法根据手头任务的复杂度动态调整正则化强度。我们提出了一种基于贝叶斯优化的方法，自动确定每个学习任务的最佳正则化强度。通过两个数据集上的两种正则化方法的实验，结果表明适应性正则化对于实现更加准确和不易遗忘的视觉增量学习非常重要。

    Class-Incremental Learning updates a deep classifier with new categories while maintaining the previously observed class accuracy. Regularizing the neural network weights is a common method to prevent forgetting previously learned classes while learning novel ones. However, existing regularizers use a constant magnitude throughout the learning sessions, which may not reflect the varying levels of difficulty of the tasks encountered during incremental learning. This study investigates the necessity of adaptive regularization in Class-Incremental Learning, which dynamically adjusts the regularization strength according to the complexity of the task at hand. We propose a Bayesian Optimization-based approach to automatically determine the optimal regularization magnitude for each learning task. Our experiments on two datasets via two regularizers demonstrate the importance of adaptive regularization for achieving accurate and less forgetful visual incremental learning.
    
[^9]: 视觉语言模型补丁-令牌对齐的贝叶斯提示学习

    Patch-Token Aligned Bayesian Prompt Learning for Vision-Language Models. (arXiv:2303.09100v1 [cs.CV])

    [http://arxiv.org/abs/2303.09100](http://arxiv.org/abs/2303.09100)

    本文提出了一种基于贝叶斯概率的视觉语言模型提示学习方法，通过将提示标记推向忠实捕捉标签特定的视觉概念，而不是过度拟合训练类别，解决了现有提示工程的问题。在各种视觉语言任务上的广泛实验表明，该方法优于现有的最先进模型。

    

    在视觉语言预训练模型的下游应用中，构建有效提示引起了极大关注。现有的提示工程方法要么需要费时费力的手动设计，要么将提示调优作为点估计问题进行优化，这可能无法描述类别的多样特征并限制了它们的应用。本文提出了一种基于贝叶斯概率的提示学习方法，其中通过从潜在分布中首先采样隐向量，然后采用轻量级生成模型来生成标签特定的随机提示。重要的是，我们将视觉知识与图像的语义规则化，并将图像和相应的提示视为补丁和令牌集，通过最优传输将提示标记推向忠实捕捉标签特定的视觉概念，而不是过度拟合训练类别。此外，所提出的模型还可以通过使用额外的基于文本的信息来生成更具信息量和准确性的提示。在各种视觉语言任务上的广泛实验表明，我们的补丁-令牌对齐的贝叶斯提示学习（PTBPL）优于现有的最先进模型。

    For downstream applications of vision-language pre-trained models, there has been significant interest in constructing effective prompts. Existing works on prompt engineering, which either require laborious manual designs or optimize the prompt tuning as a point estimation problem, may fail to describe diverse characteristics of categories and limit their applications. We introduce a Bayesian probabilistic resolution to prompt learning, where the label-specific stochastic prompts are generated hierarchically by first sampling a latent vector from an underlying distribution and then employing a lightweight generative model. Importantly, we semantically regularize prompt learning with the visual knowledge and view images and the corresponding prompts as patch and token sets under optimal transport, which pushes the prompt tokens to faithfully capture the label-specific visual concepts, instead of overfitting the training categories. Moreover, the proposed model can also be straightforwar
    

