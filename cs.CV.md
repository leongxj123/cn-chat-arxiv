# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning from Models and Data for Visual Grounding](https://arxiv.org/abs/2403.13804) | 结合数据驱动学习和模型知识传递的新框架，通过优化一致性目标来增强预训练视觉和语言模型的视觉定位能力。 |
| [^2] | [Content-aware Masked Image Modeling Transformer for Stereo Image Compression](https://arxiv.org/abs/2403.08505) | 提出了一种名为CAMSIC的立体图像压缩框架，通过引入面向内容感知的掩码图像建模（MIM）技术，使得无需额外Transformer解码器就能捕捉空间和视差依赖关系，实验结果表明实现了最先进的率失真结果。 |
| [^3] | [DyRoNet: A Low-Rank Adapter Enhanced Dynamic Routing Network for Streaming Perception](https://arxiv.org/abs/2403.05050) | DyRoNet采用低秩动态路由并结合分支网络优化流媒体感知性能，为多种分支选择策略设定了新的性能标杆 |
| [^4] | [A Literature Review of Literature Reviews in Pattern Analysis and Machine Intelligence](https://arxiv.org/abs/2402.12928) | 本文旨在提供对模式分析与机器智能领域文献综述的全面评估，引入大语言模型驱动的文献计量指标，并构建了RiPAMI元数据数据库和主题数据集以获取PAMI综述的统计特征。 |
| [^5] | [Evaluating Image Review Ability of Vision Language Models](https://arxiv.org/abs/2402.12121) | 本论文通过引入基于排名相关分析的评估方法，探讨了大规模视觉语言模型（LVLM）在生成图像评价文本方面的能力，并创建了一个评估数据集来验证这种方法。 |
| [^6] | [Privacy-Preserving Low-Rank Adaptation for Latent Diffusion Models](https://arxiv.org/abs/2402.11989) | 提出了隐私保护的低秩适应解决方案PrivateLoRA，通过最小化适应损失和代理攻击模型的MI增益来抵御成员推断攻击。 |
| [^7] | [Not all Minorities are Equal: Empty-Class-Aware Distillation for Heterogeneous Federated Learning.](http://arxiv.org/abs/2401.02329) | 本研究提出了一种异质联邦学习方法FedED，通过同时进行空类别蒸馏和逻辑抑制，解决了在联邦学习中尚未充分识别空类别的问题。 |
| [^8] | [JEN-1 Composer: A Unified Framework for High-Fidelity Multi-Track Music Generation.](http://arxiv.org/abs/2310.19180) | JEN-1 Composer是一个统一的框架，能够以高保真、灵活的方式生成多音轨音乐。 |
| [^9] | [Application of Quantum Pre-Processing Filter for Binary Image Classification with Small Samples.](http://arxiv.org/abs/2308.14930) | 本研究探讨了量子预处理滤波器（QPF）在二值图像分类中的应用，并通过在MNIST、EMNIST和CIFAR-10上提高了分类准确率，并在GTSRB上降低了分类准确率。 |

# 详细

[^1]: 从模型和数据中学习进行视觉定位

    Learning from Models and Data for Visual Grounding

    [https://arxiv.org/abs/2403.13804](https://arxiv.org/abs/2403.13804)

    结合数据驱动学习和模型知识传递的新框架，通过优化一致性目标来增强预训练视觉和语言模型的视觉定位能力。

    

    我们介绍了SynGround，这是一个结合了数据驱动学习和从各种大规模预训练模型中进行知识传递的新型框架，以增强预训练视觉和语言模型的视觉定位能力。从模型中进行的知识传递引发了通过图像描述生成器生成图像描述。这些描述具有双重作用：它们作为文本到图像生成器合成图像的提示，以及作为查询来合成文本，从其中使用大型语言模型提取短语。最后，我们利用一个开放词汇的对象检测器为合成图像和文本生成合成边界框。通过优化一个遮罩-注意力一致性目标，在这个数据集上微调预训练的视觉和语言模型，该目标将区域注释与基于梯度的模型解释进行对齐。最终的模型提升了定位能力。

    arXiv:2403.13804v1 Announce Type: cross  Abstract: We introduce SynGround, a novel framework that combines data-driven learning and knowledge transfer from various large-scale pretrained models to enhance the visual grounding capabilities of a pretrained vision-and-language model. The knowledge transfer from the models initiates the generation of image descriptions through an image description generator. These descriptions serve dual purposes: they act as prompts for synthesizing images through a text-to-image generator, and as queries for synthesizing text, from which phrases are extracted using a large language model. Finally, we leverage an open-vocabulary object detector to generate synthetic bounding boxes for the synthetic images and texts. We finetune a pretrained vision-and-language model on this dataset by optimizing a mask-attention consistency objective that aligns region annotations with gradient-based model explanations. The resulting model improves the grounding capabilit
    
[^2]: 面向内容感知的掩码图像建模变压器用于立体图像压缩

    Content-aware Masked Image Modeling Transformer for Stereo Image Compression

    [https://arxiv.org/abs/2403.08505](https://arxiv.org/abs/2403.08505)

    提出了一种名为CAMSIC的立体图像压缩框架，通过引入面向内容感知的掩码图像建模（MIM）技术，使得无需额外Transformer解码器就能捕捉空间和视差依赖关系，实验结果表明实现了最先进的率失真结果。

    

    现有基于学习的立体图像编解码器采用了复杂的转换方法，但在编码潜在表示时却采用了从单个图像编解码器导出的简单熵模型。然而，这些熵模型难以有效捕捉立体图像固有的空间-视差特征，导致亚最优的率失真结果。本文提出了一种名为CAMSIC的立体图像压缩框架。 CAMSIC 独立地将每个图像转换为潜在表示，并采用强大的无解码器变压器熵模型来捕捉空间和视差依赖关系，引入了一种新颖的面向内容感知的掩码图像建模（MIM）技术。我们的面向内容感知的MIM促进了先验信息与估计令牌之间的高效双向交互，自然地消除了额外的Transformer解码器的需求。实验证明，我们的立体图像编解码器实现了最先进的率失真结果。

    arXiv:2403.08505v1 Announce Type: cross  Abstract: Existing learning-based stereo image codec adopt sophisticated transformation with simple entropy models derived from single image codecs to encode latent representations. However, those entropy models struggle to effectively capture the spatial-disparity characteristics inherent in stereo images, which leads to suboptimal rate-distortion results. In this paper, we propose a stereo image compression framework, named CAMSIC. CAMSIC independently transforms each image to latent representation and employs a powerful decoder-free Transformer entropy model to capture both spatial and disparity dependencies, by introducing a novel content-aware masked image modeling (MIM) technique. Our content-aware MIM facilitates efficient bidirectional interaction between prior information and estimated tokens, which naturally obviates the need for an extra Transformer decoder. Experiments show that our stereo image codec achieves state-of-the-art rate-d
    
[^3]: DyRoNet：一种低秩适配器增强的动态路由网络，用于流媒体感知

    DyRoNet: A Low-Rank Adapter Enhanced Dynamic Routing Network for Streaming Perception

    [https://arxiv.org/abs/2403.05050](https://arxiv.org/abs/2403.05050)

    DyRoNet采用低秩动态路由并结合分支网络优化流媒体感知性能，为多种分支选择策略设定了新的性能标杆

    

    自主驾驶系统需要实时、准确的感知来应对复杂环境。为解决这一问题，我们引入了动态路由网络（DyRoNet），这是一个创新性的框架，采用低秩动态路由以增强流媒体感知。通过集成专门预训练的分支网络，针对各种环境条件进行微调，DyRoNet在延迟和精度之间取得了平衡。其核心特征是速度路由模块，智能地将输入数据引导到最适合的分支网络，优化性能。广泛的评估结果显示，DyRoNet有效地适应多种分支选择策略，为各种场景性能设定了新的标杆。DyRoNet不仅为流媒体感知建立了新的标杆，还为未来的工作提供了宝贵的工程洞见。有关更多项目信息，请访问 https://tastevision.github.io/DyRoNet/

    arXiv:2403.05050v1 Announce Type: cross  Abstract: Autonomous driving systems demand real-time, accurate perception to navigate complex environments. Addressing this, we introduce the Dynamic Router Network (DyRoNet), a framework that innovates with low-rank dynamic routing for enhanced streaming perception. By integrating specialized pre-trained branch networks, fine-tuned for various environmental conditions, DyRoNet achieves a balance between latency and precision. Its core feature, the speed router module, intelligently directs input data to the best-suited branch network, optimizing performance. The extensive evaluations reveal that DyRoNet adapts effectively to multiple branch selection strategies, setting a new benchmark in performance across a range of scenarios. DyRoNet not only establishes a new benchmark for streaming perception but also provides valuable engineering insights for future work. More project information is available at https://tastevision.github.io/DyRoNet/
    
[^4]: 模式分析与机器智能领域文献综述的文献综述

    A Literature Review of Literature Reviews in Pattern Analysis and Machine Intelligence

    [https://arxiv.org/abs/2402.12928](https://arxiv.org/abs/2402.12928)

    本文旨在提供对模式分析与机器智能领域文献综述的全面评估，引入大语言模型驱动的文献计量指标，并构建了RiPAMI元数据数据库和主题数据集以获取PAMI综述的统计特征。

    

    通过整合分散的知识，文献综述提供了对所研究主题的全面了解。然而，在模式分析与机器智能（PAMI）这一蓬勃发展的领域中，过多的综述引起了研究人员和评论者的关注。作为对这些关注的回应，本文旨在从多个角度全面审视PAMI领域的综述文献。

    arXiv:2402.12928v1 Announce Type: cross  Abstract: By consolidating scattered knowledge, the literature review provides a comprehensive understanding of the investigated topic. However, excessive reviews, especially in the booming field of pattern analysis and machine intelligence (PAMI), raise concerns for both researchers and reviewers. In response to these concerns, this Analysis aims to provide a thorough review of reviews in the PAMI field from diverse perspectives. First, large language model-empowered bibliometric indicators are proposed to evaluate literature reviews automatically. To facilitate this, a meta-data database dubbed RiPAMI, and a topic dataset are constructed, which are utilized to obtain statistical characteristics of PAMI reviews. Unlike traditional bibliometric measurements, the proposed article-level indicators provide real-time and field-normalized quantified assessments of reviews without relying on user-defined keywords. Second, based on these indicators, th
    
[^5]: 评估视觉语言模型的图像评价能力

    Evaluating Image Review Ability of Vision Language Models

    [https://arxiv.org/abs/2402.12121](https://arxiv.org/abs/2402.12121)

    本论文通过引入基于排名相关分析的评估方法，探讨了大规模视觉语言模型（LVLM）在生成图像评价文本方面的能力，并创建了一个评估数据集来验证这种方法。

    

    大规模视觉语言模型（LVLM）是能够通过单个模型处理图像和文本输入的语言模型。本文探讨了使用LVLM生成图像评价文本的方法。LVLM对图像的评价能力尚未完全被理解，突显了对其评价能力进行系统评估的必要性。与图像标题不同，评价文本可以从图像构图和曝光等不同视角撰写。这种评价角度的多样性使得难以唯一确定图像的正确评价。为了解决这一挑战，我们提出了一种基于排名相关分析的评估方法，通过人类和LVLM对评价文本进行排名，然后测量这些排名之间的相关性。我们进一步通过创建一个旨在评估最新LVLM图像评价能力的基准数据集来验证这种方法。

    arXiv:2402.12121v1 Announce Type: cross  Abstract: Large-scale vision language models (LVLMs) are language models that are capable of processing images and text inputs by a single model. This paper explores the use of LVLMs to generate review texts for images. The ability of LVLMs to review images is not fully understood, highlighting the need for a methodical evaluation of their review abilities. Unlike image captions, review texts can be written from various perspectives such as image composition and exposure. This diversity of review perspectives makes it difficult to uniquely determine a single correct review for an image. To address this challenge, we introduce an evaluation method based on rank correlation analysis, in which review texts are ranked by humans and LVLMs, then, measures the correlation between these rankings. We further validate this approach by creating a benchmark dataset aimed at assessing the image review ability of recent LVLMs. Our experiments with the dataset
    
[^6]: 隐私保护的低秩适应Latent扩散模型

    Privacy-Preserving Low-Rank Adaptation for Latent Diffusion Models

    [https://arxiv.org/abs/2402.11989](https://arxiv.org/abs/2402.11989)

    提出了隐私保护的低秩适应解决方案PrivateLoRA，通过最小化适应损失和代理攻击模型的MI增益来抵御成员推断攻击。

    

    低秩适应（LoRA）是一种有效的策略，用于通过最小化适应损失，自训练数据集中适应Latent扩散模型（LDM）以生成特定对象。然而，通过LoRA适应的LDM容易受到成员推断（MI）攻击的影响，这种攻击可以判断特定数据点是否属于私人训练数据集，因此面临严重的隐私泄露风险。为了抵御MI攻击，我们首次提出了一个直接的解决方案：隐私保护的LoRA（PrivateLoRA）。PrivateLoRA被构建为一个最小最大优化问题，其中通过最大化MI增益来训练代理攻击模型，而LDM则通过最小化适应损失和代理攻击模型的MI增益之和来进行调整。然而，我们在实践中发现PrivateLoRA存在稳定性优化问题，即由于梯度规模的大幅波动而妨碍适应。

    arXiv:2402.11989v1 Announce Type: new  Abstract: Low-rank adaptation (LoRA) is an efficient strategy for adapting latent diffusion models (LDMs) on a training dataset to generate specific objects by minimizing the adaptation loss. However, adapted LDMs via LoRA are vulnerable to membership inference (MI) attacks that can judge whether a particular data point belongs to private training datasets, thus facing severe risks of privacy leakage. To defend against MI attacks, we make the first effort to propose a straightforward solution: privacy-preserving LoRA (PrivateLoRA). PrivateLoRA is formulated as a min-max optimization problem where a proxy attack model is trained by maximizing its MI gain while the LDM is adapted by minimizing the sum of the adaptation loss and the proxy attack model's MI gain. However, we empirically disclose that PrivateLoRA has the issue of unstable optimization due to the large fluctuation of the gradient scale which impedes adaptation. To mitigate this issue, w
    
[^7]: 不是所有的少数群体都是平等的: 空类别感知的异质联邦学习方法

    Not all Minorities are Equal: Empty-Class-Aware Distillation for Heterogeneous Federated Learning. (arXiv:2401.02329v1 [cs.LG])

    [http://arxiv.org/abs/2401.02329](http://arxiv.org/abs/2401.02329)

    本研究提出了一种异质联邦学习方法FedED，通过同时进行空类别蒸馏和逻辑抑制，解决了在联邦学习中尚未充分识别空类别的问题。

    

    数据异质性是联邦学习中的一个重大挑战，表现为客户端之间本地数据分布的差异。现有方法常常在本地训练过程中采用类别平衡的技术来解决本地类别分布的异质性问题。然而，在少数类别中由于过拟合本地不平衡数据而导致准确性较差的问题仍然存在。本文提出了FedED，这是一种新颖的异质联邦学习方法，同时整合了空类别蒸馏和逻辑抑制。具体而言，空类别蒸馏利用知识蒸馏的方法在每个客户端的本地训练中保留了与空类别相关的重要信息。此外，逻辑抑制直接阻断了预测结果中对空类别的输出。

    Data heterogeneity, characterized by disparities in local data distribution across clients, poses a significant challenge in federated learning. Substantial efforts have been devoted to addressing the heterogeneity in local label distribution. As minority classes suffer from worse accuracy due to overfitting on local imbalanced data, prior methods often incorporate class-balanced learning techniques during local training. Despite the improved mean accuracy across all classes, we observe that empty classes-referring to categories absent from a client's data distribution-are still not well recognized. This paper introduces FedED, a novel approach in heterogeneous federated learning that integrates both empty-class distillation and logit suppression simultaneously. Specifically, empty-class distillation leverages knowledge distillation during local training on each client to retain essential information related to empty classes from the global model. Moreover, logit suppression directly p
    
[^8]: JEN-1 Composer: 一个用于高保真多音轨音乐生成的统一框架

    JEN-1 Composer: A Unified Framework for High-Fidelity Multi-Track Music Generation. (arXiv:2310.19180v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2310.19180](http://arxiv.org/abs/2310.19180)

    JEN-1 Composer是一个统一的框架，能够以高保真、灵活的方式生成多音轨音乐。

    

    随着生成式人工智能的快速发展，从零开始生成音乐的文本到音乐合成任务已成为一个有前景的方向。然而，对于多音轨生成的更细粒度控制仍然是一个挑战。现有模型具有较强的原始生成能力，但缺乏以可控的方式单独组成和组合多音轨的灵活性，这与人类作曲家的典型工作流程不同。为了解决这个问题，我们提出了JEN-1 Composer，一个统一的框架，通过一个模型高效地建模多音轨音乐的边缘、条件和联合分布。JEN-1 Composer框架能够无缝地整合任何基于扩散的音乐生成系统，例如Jen-1，增强其多功能多音轨音乐生成能力。我们引入了一种课程训练策略，以逐步指导模型从单音轨生成到灵活的生成过程。

    With rapid advances in generative artificial intelligence, the text-to-music synthesis task has emerged as a promising direction for music generation from scratch. However, finer-grained control over multi-track generation remains an open challenge. Existing models exhibit strong raw generation capability but lack the flexibility to compose separate tracks and combine them in a controllable manner, differing from typical workflows of human composers. To address this issue, we propose JEN-1 Composer, a unified framework to efficiently model marginal, conditional, and joint distributions over multi-track music via a single model. JEN-1 Composer framework exhibits the capacity to seamlessly incorporate any diffusion-based music generation system, \textit{e.g.} Jen-1, enhancing its capacity for versatile multi-track music generation. We introduce a curriculum training strategy aimed at incrementally instructing the model in the transition from single-track generation to the flexible genera
    
[^9]: 量子预处理滤波器在小样本二值图像分类中的应用

    Application of Quantum Pre-Processing Filter for Binary Image Classification with Small Samples. (arXiv:2308.14930v1 [cs.CV])

    [http://arxiv.org/abs/2308.14930](http://arxiv.org/abs/2308.14930)

    本研究探讨了量子预处理滤波器（QPF）在二值图像分类中的应用，并通过在MNIST、EMNIST和CIFAR-10上提高了分类准确率，并在GTSRB上降低了分类准确率。

    

    过去几年来，量子机器学习（QML）在研究人员中引起了极大的兴趣，因为它有潜力改变机器学习领域。已开发出利用量子力学特性的几种模型用于实际应用。本研究探讨了我们之前提出的量子预处理滤波器（QPF）在二值图像分类中的应用。我们对四个数据集进行了QPF的评估：MNIST（手写数字）、EMNIST（手写数字和字母）、CIFAR-10（照片图像）和GTSRB（真实交通标志图像）。与我们之前的多类别分类结果类似，应用QPF使得使用神经网络对MNIST、EMNIST和CIFAR-10的二值图像分类准确率分别从98.9%提高到99.2%、从97.8%提高到98.3%和从71.2%提高到76.1%，但在GTSRB上的准确率下降了从93.5%到92.0%。然后我们将QPF应用于训练样本数量较少的情况下。

    Over the past few years, there has been significant interest in Quantum Machine Learning (QML) among researchers, as it has the potential to transform the field of machine learning. Several models that exploit the properties of quantum mechanics have been developed for practical applications. In this study, we investigated the application of our previously proposed quantum pre-processing filter (QPF) to binary image classification. We evaluated the QPF on four datasets: MNIST (handwritten digits), EMNIST (handwritten digits and alphabets), CIFAR-10 (photographic images) and GTSRB (real-life traffic sign images). Similar to our previous multi-class classification results, the application of QPF improved the binary image classification accuracy using neural network against MNIST, EMNIST, and CIFAR-10 from 98.9% to 99.2%, 97.8% to 98.3%, and 71.2% to 76.1%, respectively, but degraded it against GTSRB from 93.5% to 92.0%. We then applied QPF in cases using a smaller number of training and 
    

