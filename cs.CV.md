# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tutorial on Diffusion Models for Imaging and Vision](https://arxiv.org/abs/2403.18103) | 该教程讨论了图像和视觉领域中扩散模型的基本理念，适合对扩散模型研究或应用感兴趣的本科生和研究生。 |
| [^2] | [A Bag of Tricks for Few-Shot Class-Incremental Learning](https://arxiv.org/abs/2403.14392) | 提出了针对少样本类增量学习的一揽子技巧框架，将八种关键技术结合在一起，改进了稳定性、适应性和整体性能 |
| [^3] | [Outlier detection by ensembling uncertainty with negative objectness](https://arxiv.org/abs/2402.15374) | 提出一种利用不确定性和负对象性集成的异常检测方法，通过直接预测K+1个logits并在密集预测结构中嵌入，可独立检测异常值。 |
| [^4] | [MIM-Refiner: A Contrastive Learning Boost from Intermediate Pre-Trained Representations](https://arxiv.org/abs/2402.10093) | MIM-Refiner是一种对比学习提升方法，通过利用MIM模型中的中间层表示和多个对比头，能够将MIM模型的特征从次优的状态提升到最先进的状态，并在ImageNet-1K数据集上取得了新的最先进结果。 |
| [^5] | [MixedNUTS: Training-Free Accuracy-Robustness Balance via Nonlinearly Mixed Classifiers](https://arxiv.org/abs/2402.02263) | MixedNUTS是一种无需训练的方法，通过非线性混合分类器的转换和概率混合来实现准确性和鲁棒性的平衡。 |
| [^6] | [CMMMU: A Chinese Massive Multi-discipline Multimodal Understanding Benchmark](https://arxiv.org/abs/2401.11944) | CMMMU是一个旨在评估大型多模型模型在大学级学科知识和深思熟虑推理任务中表现的中文大规模多学科多模态理解基准，为填补在非英语环境中评估先进知识和推理能力的空白而设计。 |
| [^7] | [DEFormer: DCT-driven Enhancement Transformer for Low-light Image and Dark Vision.](http://arxiv.org/abs/2309.06941) | 该论文提出了一种新的DCT驱动增强Transformer（DEFormer），可以在低光图像中恢复丢失的细节，通过引入频率作为新的线索，通过可学习的频率分支（LFB）和基于曲率的频率增强（CFE）来实现。此外，还提出了交叉域融合（CDF）来减少领域之间的差异，DEFormer还可以作为暗部检测的预处理，有效提高了性能。 |
| [^8] | [PointLLM: Empowering Large Language Models to Understand Point Clouds.](http://arxiv.org/abs/2308.16911) | PointLLM是一种使大型语言模型理解点云的方法，它利用点云编码器和强大的LLM将几何、外观和语言信息融合，并通过人类指导生成环境上恰当的响应。该方法通过收集大规模的点-文本指令对数据集进行两阶段的训练，以提高模型的感知能力和泛化能力。 |
| [^9] | [The Principle of Uncertain Maximum Entropy.](http://arxiv.org/abs/2305.09868) | 介绍了不确定最大熵原理，该原理可以处理模型元素不可观测的情况，并优于特定条件下的最大熵方法。同时将黑匣子机器学习模型的输出用作不确定机器熵框架的输入，性能得到了提高。 |
| [^10] | [Launching a Robust Backdoor Attack under Capability Constrained Scenarios.](http://arxiv.org/abs/2304.10985) | 深度神经网络的后门攻击一直是一个安全性问题，现有的改进方法需要强大的攻击者能力，在能力受限场景下还没有找到令人满意的解决办法，此外，模型鲁棒性仍然值得关注。 |

# 详细

[^1]: 关于图像和视觉扩散模型的教程

    Tutorial on Diffusion Models for Imaging and Vision

    [https://arxiv.org/abs/2403.18103](https://arxiv.org/abs/2403.18103)

    该教程讨论了图像和视觉领域中扩散模型的基本理念，适合对扩散模型研究或应用感兴趣的本科生和研究生。

    

    近年来生成工具的惊人增长使得文本到图像生成和文本到视频生成等许多令人兴奋的应用成为可能。这些生成工具背后的基本原理是扩散概念，一种特殊的采样机制，克服了以前方法中被认为困难的一些缺点。本教程的目标是讨论扩散模型的基本理念。本教程的目标受众包括对研究扩散模型或将这些模型应用于解决其他问题感兴趣的本科生和研究生。

    arXiv:2403.18103v1 Announce Type: new  Abstract: The astonishing growth of generative tools in recent years has empowered many exciting applications in text-to-image generation and text-to-video generation. The underlying principle behind these generative tools is the concept of diffusion, a particular sampling mechanism that has overcome some shortcomings that were deemed difficult in the previous approaches. The goal of this tutorial is to discuss the essential ideas underlying the diffusion models. The target audience of this tutorial includes undergraduate and graduate students who are interested in doing research on diffusion models or applying these models to solve other problems.
    
[^2]: 用于少样本类增量学习的一揽子技巧

    A Bag of Tricks for Few-Shot Class-Incremental Learning

    [https://arxiv.org/abs/2403.14392](https://arxiv.org/abs/2403.14392)

    提出了针对少样本类增量学习的一揽子技巧框架，将八种关键技术结合在一起，改进了稳定性、适应性和整体性能

    

    我们提出了一个一揽子技巧框架，用于少样本类增量学习（FSCIL），这是一种具有挑战性的连续学习形式，涉及对新任务进行连续适应，并且样本有限。 FSCIL 需要保持稳定性和适应性，即在学习新任务时保持先前学习任务的熟练程度。我们提出的一揽子技巧将八种关键且具有高影响力的技术汇集在一起，针对 FSCIL 在一个统一框架下改进稳定性、适应性和整体性能。我们将这些技巧组织成三类：稳定性技巧、适应性技巧和训练技巧。稳定性技巧旨在通过增强已学习类别的嵌入之间的分离和在学习新类别时最小化干扰来减轻先前学习类别的遗忘。另一方面，适应性技巧侧重于有效学习新类别。

    arXiv:2403.14392v1 Announce Type: cross  Abstract: We present a bag of tricks framework for few-shot class-incremental learning (FSCIL), which is a challenging form of continual learning that involves continuous adaptation to new tasks with limited samples. FSCIL requires both stability and adaptability, i.e., preserving proficiency in previously learned tasks while learning new ones. Our proposed bag of tricks brings together eight key and highly influential techniques that improve stability, adaptability, and overall performance under a unified framework for FSCIL. We organize these tricks into three categories: stability tricks, adaptability tricks, and training tricks. Stability tricks aim to mitigate the forgetting of previously learned classes by enhancing the separation between the embeddings of learned classes and minimizing interference when learning new ones. On the other hand, adaptability tricks focus on the effective learning of new classes. Finally, training tricks improv
    
[^3]: 利用不确定性和负对象性集成的异常检测

    Outlier detection by ensembling uncertainty with negative objectness

    [https://arxiv.org/abs/2402.15374](https://arxiv.org/abs/2402.15374)

    提出一种利用不确定性和负对象性集成的异常检测方法，通过直接预测K+1个logits并在密集预测结构中嵌入，可独立检测异常值。

    

    异常检测是监督式视觉识别中关键的功能。现有的大多数方法通过鼓励标准封闭集模型在负训练数据中产生低置信度预测来获得最佳结果。然而，这种方法混淆了预测不确定性和对负类别的识别。因此，我们重新考虑了直接预测K+1个logits，这些logits对应于K个基本真实类别和一个异常类别。这种设置允许我们制定一种新奇的异常得分，作为分布内不确定性和异常类别的后验的集合，我们称之为负对象性。现在，异常值可以通过高预测不确定性或与负数据相似之处独立检测。我们将我们的方法嵌入到一个密集预测结构中，该结构具有K+2个类别的掩码级别识别。训练过程鼓励新颖的K+2-th类别去学习

    arXiv:2402.15374v1 Announce Type: cross  Abstract: Outlier detection is an essential capability in safety-critical applications of supervised visual recognition. Most of the existing methods deliver best results by encouraging standard closed-set models to produce low-confidence predictions in negative training data. However, that approach conflates prediction uncertainty with recognition of the negative class. We therefore reconsider direct prediction of K+1 logits that correspond to K groundtruth classes and one outlier class. This setup allows us to formulate a novel anomaly score as an ensemble of in-distribution uncertainty and the posterior of the outlier class which we term negative objectness. Now outliers can be independently detected due to i) high prediction uncertainty or ii) similarity with negative data. We embed our method into a dense prediction architecture with mask-level recognition over K+2 classes. The training procedure encourages the novel K+2-th class to learn n
    
[^4]: MIM-Refiner：一种从中间预训练表示中获得对比学习提升的方法

    MIM-Refiner: A Contrastive Learning Boost from Intermediate Pre-Trained Representations

    [https://arxiv.org/abs/2402.10093](https://arxiv.org/abs/2402.10093)

    MIM-Refiner是一种对比学习提升方法，通过利用MIM模型中的中间层表示和多个对比头，能够将MIM模型的特征从次优的状态提升到最先进的状态，并在ImageNet-1K数据集上取得了新的最先进结果。

    

    我们引入了MIM-Refiner，这是一种用于预训练MIM模型的对比学习提升方法。MIM-Refiner的动机在于MIM模型中的最佳表示通常位于中间层。因此，MIM-Refiner利用连接到不同中间层的多个对比头。在每个头中，修改后的最近邻目标帮助构建相应的语义聚类。此过程短而有效，在几个epochs内，我们将MIM模型的特征从次优的状态提升到最先进的状态。使用data2vec 2.0在ImageNet-1K上预训练的ViT-H经过改进后，在线性探测和低样本分类方面取得了新的最先进结果（分别为84.7%和64.2%），超过了在ImageNet-1K上预训练的其他模型的表现。

    arXiv:2402.10093v1 Announce Type: cross  Abstract: We introduce MIM (Masked Image Modeling)-Refiner, a contrastive learning boost for pre-trained MIM models. The motivation behind MIM-Refiner is rooted in the insight that optimal representations within MIM models generally reside in intermediate layers. Accordingly, MIM-Refiner leverages multiple contrastive heads that are connected to diverse intermediate layers. In each head, a modified nearest neighbor objective helps to construct respective semantic clusters.   The refinement process is short but effective. Within a few epochs, we refine the features of MIM models from subpar to state-of-the-art, off-the-shelf features. Refining a ViT-H, pre-trained with data2vec 2.0 on ImageNet-1K, achieves new state-of-the-art results in linear probing (84.7%) and low-shot classification among models that are pre-trained on ImageNet-1K. In ImageNet-1K 1-shot classification, MIM-Refiner sets a new state-of-the-art of 64.2%, outperforming larger mo
    
[^5]: MixedNUTS: 通过非线性混合分类器实现无需训练的准确性和鲁棒性平衡

    MixedNUTS: Training-Free Accuracy-Robustness Balance via Nonlinearly Mixed Classifiers

    [https://arxiv.org/abs/2402.02263](https://arxiv.org/abs/2402.02263)

    MixedNUTS是一种无需训练的方法，通过非线性混合分类器的转换和概率混合来实现准确性和鲁棒性的平衡。

    

    鲁棒性往往牺牲了准确性，阻碍了鲁棒分类模型在实际应用中的使用。基于训练的解决方案在与已训练的大型高性能模型兼容性方面存在限制，因此需要探索无需训练的集成方法。我们观察到鲁棒模型在干净数据和对抗数据上的正确预测比错误预测更自信，我们推测通过增强这种“良性置信度特性”可以在集成环境中实现准确性和鲁棒性的平衡。为了实现这一点，我们提出了“MixedNUTS”，一种无需训练的方法，利用仅有三个参数的非线性转换来处理鲁棒分类器和标准非鲁棒分类器的输出Logits，并通过高效算法进行优化。然后，MixedNUTS将转换后的Logits转换为概率，并将它们混合作为最终的输出。在CIFAR-10、CIFAR-100和ImageNet数据集上进行了实验。

    Adversarial robustness often comes at the cost of degraded accuracy, impeding the real-life application of robust classification models. Training-based solutions for better trade-offs are limited by incompatibilities with already-trained high-performance large models, necessitating the exploration of training-free ensemble approaches. Observing that robust models are more confident in correct predictions than in incorrect ones on clean and adversarial data alike, we speculate amplifying this "benign confidence property" can reconcile accuracy and robustness in an ensemble setting. To achieve so, we propose "MixedNUTS", a training-free method where the output logits of a robust classifier and a standard non-robust classifier are processed by nonlinear transformations with only three parameters, which are optimized through an efficient algorithm. MixedNUTS then converts the transformed logits into probabilities and mixes them as the overall output. On CIFAR-10, CIFAR-100, and ImageNet da
    
[^6]: CMMMU：一个中国大规模多学科多模态理解基准

    CMMMU: A Chinese Massive Multi-discipline Multimodal Understanding Benchmark

    [https://arxiv.org/abs/2401.11944](https://arxiv.org/abs/2401.11944)

    CMMMU是一个旨在评估大型多模型模型在大学级学科知识和深思熟虑推理任务中表现的中文大规模多学科多模态理解基准，为填补在非英语环境中评估先进知识和推理能力的空白而设计。

    

    随着大型多模型模型(LMMs)的能力不断提升，评估LMMs的表现日益成为一个迫切的需求。此外，在评估LMMs在中文等非英语环境中先进知识和推理能力方面存在更大差距。我们引入了CMMMU，一个新的中文大规模多学科多模态理解基准，旨在评估LMMs在需要大学水平学科知识和深思熟虑推理的任务中的表现。CMMMU受到了MMMUs的标注和分析模式的启发并严格遵循。CMMMU包括来自大学考试、测验和教科书的1.2万个手动收集的多模态问题，涵盖六个核心学科：艺术与设计、商业、科学、健康与医学、人文社科以及技术与工程，就像其伙伴MMMMU一样。这些问题涵盖30个学科，包括39个高度异质的图像。

    arXiv:2401.11944v2 Announce Type: replace-cross  Abstract: As the capabilities of large multimodal models (LMMs) continue to advance, evaluating the performance of LMMs emerges as an increasing need. Additionally, there is an even larger gap in evaluating the advanced knowledge and reasoning abilities of LMMs in non-English contexts such as Chinese. We introduce CMMMU, a new Chinese Massive Multi-discipline Multimodal Understanding benchmark designed to evaluate LMMs on tasks demanding college-level subject knowledge and deliberate reasoning in a Chinese context. CMMMU is inspired by and strictly follows the annotation and analysis pattern of MMMU.   CMMMU includes 12k manually collected multimodal questions from college exams, quizzes, and textbooks, covering six core disciplines: Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, and Tech & Engineering, like its companion, MMMU. These questions span 30 subjects and comprise 39 highly heterogeneous image 
    
[^7]: DEFormer: 用于低光图像和暗视觉的DCT驱动增强Transformer

    DEFormer: DCT-driven Enhancement Transformer for Low-light Image and Dark Vision. (arXiv:2309.06941v1 [cs.CV])

    [http://arxiv.org/abs/2309.06941](http://arxiv.org/abs/2309.06941)

    该论文提出了一种新的DCT驱动增强Transformer（DEFormer），可以在低光图像中恢复丢失的细节，通过引入频率作为新的线索，通过可学习的频率分支（LFB）和基于曲率的频率增强（CFE）来实现。此外，还提出了交叉域融合（CDF）来减少领域之间的差异，DEFormer还可以作为暗部检测的预处理，有效提高了性能。

    

    低光图像增强的目标是恢复图像的颜色和细节，对于自动驾驶中的高级视觉任务非常重要。然而，仅依靠RGB领域很难恢复暗区域的丢失细节。本文将频率作为网络的新线索，并提出了一种新颖的DCT驱动增强Transformer（DEFormer）。首先，我们提出了一个可学习的频率分支（LFB）用于频率增强，包括DCT处理和基于曲率的频率增强（CFE）。CFE计算每个通道的曲率以表示不同频率带的细节丰富度，然后我们将频率特征划分为更丰富纹理的频率带。此外，我们提出了一个交叉域融合（CDF）来减少RGB领域和频率领域之间的差异。我们还将DEFormer作为暗部检测的预处理，DEFormer有效提高了性能。

    The goal of low-light image enhancement is to restore the color and details of the image and is of great significance for high-level visual tasks in autonomous driving. However, it is difficult to restore the lost details in the dark area by relying only on the RGB domain. In this paper we introduce frequency as a new clue into the network and propose a novel DCT-driven enhancement transformer (DEFormer). First, we propose a learnable frequency branch (LFB) for frequency enhancement contains DCT processing and curvature-based frequency enhancement (CFE). CFE calculates the curvature of each channel to represent the detail richness of different frequency bands, then we divides the frequency features, which focuses on frequency bands with richer textures. In addition, we propose a cross domain fusion (CDF) for reducing the differences between the RGB domain and the frequency domain. We also adopt DEFormer as a preprocessing in dark detection, DEFormer effectively improves the performance
    
[^8]: PointLLM：赋予大型语言模型理解点云的能力

    PointLLM: Empowering Large Language Models to Understand Point Clouds. (arXiv:2308.16911v1 [cs.CV])

    [http://arxiv.org/abs/2308.16911](http://arxiv.org/abs/2308.16911)

    PointLLM是一种使大型语言模型理解点云的方法，它利用点云编码器和强大的LLM将几何、外观和语言信息融合，并通过人类指导生成环境上恰当的响应。该方法通过收集大规模的点-文本指令对数据集进行两阶段的训练，以提高模型的感知能力和泛化能力。

    

    大型语言模型（LLM）的前所未有的进展对自然语言处理产生了深远影响，但在3D理解领域仍有待完全发展。本文介绍了PointLLM，这是一项填补这一空白的初步工作，使LLM能够理解点云，并提供了超越2D视觉数据的新途径。PointLLM通过人类指导处理带有颜色的物体点云，并生成环境上恰当的响应，展示了其对点云和常识的掌握。具体来说，它利用了一个点云编码器和一个强大的LLM，有效地融合了几何、外观和语言信息。我们收集了一个新颖的数据集，包括66万个简单和7万个复杂的点-文本指令对，以实现两阶段的训练策略：首先对齐潜在空间，然后对统一模型进行指令调整。为了严格评估我们模型的感知能力和其泛化能力，我们建立了评估基准数据集进行实验。

    The unprecedented advancements in Large Language Models (LLMs) have created a profound impact on natural language processing but are yet to fully embrace the realm of 3D understanding. This paper introduces PointLLM, a preliminary effort to fill this gap, thereby enabling LLMs to understand point clouds and offering a new avenue beyond 2D visual data. PointLLM processes colored object point clouds with human instructions and generates contextually appropriate responses, illustrating its grasp of point clouds and common sense. Specifically, it leverages a point cloud encoder with a powerful LLM to effectively fuse geometric, appearance, and linguistic information. We collect a novel dataset comprising 660K simple and 70K complex point-text instruction pairs to enable a two-stage training strategy: initially aligning latent spaces and subsequently instruction-tuning the unified model. To rigorously evaluate our model's perceptual abilities and its generalization capabilities, we establis
    
[^9]: 不确定最大熵原理

    The Principle of Uncertain Maximum Entropy. (arXiv:2305.09868v1 [cs.IT])

    [http://arxiv.org/abs/2305.09868](http://arxiv.org/abs/2305.09868)

    介绍了不确定最大熵原理，该原理可以处理模型元素不可观测的情况，并优于特定条件下的最大熵方法。同时将黑匣子机器学习模型的输出用作不确定机器熵框架的输入，性能得到了提高。

    

    最大熵原理在信息理论中的引入，为统计力学，机器学习和生态学等各个领域的发展做出了贡献。其得到的解决方案作为催化剂，促进研究人员将他们的经验观察映射到获取无偏模型，同时加深了对复杂系统和现象的理解。然而，在模型元素不直接可观测的情况下，例如存在噪声或眼部遮挡的情况下，标准最大熵方法可能会失败，因为它们无法匹配特征约束。在这里，我们展示了不确定最大熵原理作为一种方法，尽管存在任意噪声观察，它同时将所有可用信息编码，而且优于一些特定条件下的最大熵方法的准确度。此外，我们将黑匣子机器学习模型的输出用作不确定机器熵框架的输入，从而在与最大似然算法相比时建立了改进的性能。

    The principle of maximum entropy, as introduced by Jaynes in information theory, has contributed to advancements in various domains such as Statistical Mechanics, Machine Learning, and Ecology. Its resultant solutions have served as a catalyst, facilitating researchers in mapping their empirical observations to the acquisition of unbiased models, whilst deepening the understanding of complex systems and phenomena. However, when we consider situations in which the model elements are not directly observable, such as when noise or ocular occlusion is present, possibilities arise for which standard maximum entropy approaches may fail, as they are unable to match feature constraints. Here we show the Principle of Uncertain Maximum Entropy as a method that both encodes all available information in spite of arbitrarily noisy observations while surpassing the accuracy of some ad-hoc methods. Additionally, we utilize the output of a black-box machine learning model as input into an uncertain ma
    
[^10]: 在能力受限场景下启动强韧后门攻击

    Launching a Robust Backdoor Attack under Capability Constrained Scenarios. (arXiv:2304.10985v1 [cs.CR])

    [http://arxiv.org/abs/2304.10985](http://arxiv.org/abs/2304.10985)

    深度神经网络的后门攻击一直是一个安全性问题，现有的改进方法需要强大的攻击者能力，在能力受限场景下还没有找到令人满意的解决办法，此外，模型鲁棒性仍然值得关注。

    

    随着深度神经网络在关键领域的应用不断增加，人们开始担心它们的安全性。由于缺乏透明度，深度学习模型容易受到后门攻击的威胁。污染的后门模型在普通环境下可能表现正常，但当输入包含触发器时，会显示出恶意行为。目前对后门攻击的研究集中于改善触发器的秘密性，大多数方法需要强大的攻击者能力，例如对模型结构的了解或对训练过程的控制。由于在大多数情况下攻击者的能力受到限制，这些攻击是不切实际的。此外，模型鲁棒性的问题还未得到充分关注。例如，模型蒸馏常用于简化模型大小，但随着参数数量指数级增长，以前的许多后门攻击在模型蒸馏后均失败;图像增强操作可以破坏触发器，从而使后门攻击失效。

    As deep neural networks continue to be used in critical domains, concerns over their security have emerged. Deep learning models are vulnerable to backdoor attacks due to the lack of transparency. A poisoned backdoor model may perform normally in routine environments, but exhibit malicious behavior when the input contains a trigger. Current research on backdoor attacks focuses on improving the stealthiness of triggers, and most approaches require strong attacker capabilities, such as knowledge of the model structure or control over the training process. These attacks are impractical since in most cases the attacker's capabilities are limited. Additionally, the issue of model robustness has not received adequate attention. For instance, model distillation is commonly used to streamline model size as the number of parameters grows exponentially, and most of previous backdoor attacks failed after model distillation; the image augmentation operations can destroy the trigger and thus disabl
    

