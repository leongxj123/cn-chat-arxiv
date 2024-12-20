# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Spectral Motion Alignment for Video Motion Transfer using Diffusion Models](https://arxiv.org/abs/2403.15249) | 提出了一种名为Spectral Motion Alignment（SMA）的新框架，通过傅立叶和小波变换来优化和对齐运动向量，学习整帧全局运动动态，减轻空间伪影，有效改善运动转移。 |
| [^2] | [Image Classification with Rotation-Invariant Variational Quantum Circuits](https://arxiv.org/abs/2403.15031) | 提出了一种使用旋转不变变分量子电路进行图像分类的等变架构，通过引入几何归纳偏差，成功提升了模型性能。 |
| [^3] | [PALM: Pushing Adaptive Learning Rate Mechanisms for Continual Test-Time Adaptation](https://arxiv.org/abs/2403.10650) | 本研究通过对模型预测不确定性的量化来选择需要进一步适应的层，从而克服了持续测试时间自适应方法中由于伪标签引起的不准确性困扰。 |
| [^4] | [Enhancing Multimodal Large Language Models with Vision Detection Models: An Empirical Study](https://arxiv.org/abs/2401.17981) | 本文通过实证研究，将最先进的目标检测和光学字符识别模型与多模态大型语言模型相结合，在提高图像理解和减少回答错误插入方面取得了显著改进。 |

# 详细

[^1]: 使用扩散模型进行视频运动转移的光谱运动对齐

    Spectral Motion Alignment for Video Motion Transfer using Diffusion Models

    [https://arxiv.org/abs/2403.15249](https://arxiv.org/abs/2403.15249)

    提出了一种名为Spectral Motion Alignment（SMA）的新框架，通过傅立叶和小波变换来优化和对齐运动向量，学习整帧全局运动动态，减轻空间伪影，有效改善运动转移。

    

    扩散模型的发展在视频生成和理解方面产生了巨大影响。特别是，文本到视频扩散模型（VDMs）显著促进了将输入视频定制为目标外观、运动等。尽管取得了这些进展，但准确提取视频帧的运动信息仍然存在挑战。现有作品利用连续帧残差作为目标运动向量，但它们固有地缺乏全局运动背景，并容易受到逐帧失真的影响。为了解决这个问题，我们提出了光谱运动对齐（SMA），这是一个通过傅立叶和小波变换来优化和对齐运动向量的新框架。SMA通过整合频域正则化来学习运动模式，促进整帧全局运动动态的学习，并减轻空间伪影。大量实验证明了SMA在改善运动转移方面的有效性。

    arXiv:2403.15249v1 Announce Type: cross  Abstract: The evolution of diffusion models has greatly impacted video generation and understanding. Particularly, text-to-video diffusion models (VDMs) have significantly facilitated the customization of input video with target appearance, motion, etc. Despite these advances, challenges persist in accurately distilling motion information from video frames. While existing works leverage the consecutive frame residual as the target motion vector, they inherently lack global motion context and are vulnerable to frame-wise distortions. To address this, we present Spectral Motion Alignment (SMA), a novel framework that refines and aligns motion vectors using Fourier and wavelet transforms. SMA learns motion patterns by incorporating frequency-domain regularization, facilitating the learning of whole-frame global motion dynamics, and mitigating spatial artifacts. Extensive experiments demonstrate SMA's efficacy in improving motion transfer while main
    
[^2]: 使用旋转不变变分量子电路进行图像分类

    Image Classification with Rotation-Invariant Variational Quantum Circuits

    [https://arxiv.org/abs/2403.15031](https://arxiv.org/abs/2403.15031)

    提出了一种使用旋转不变变分量子电路进行图像分类的等变架构，通过引入几何归纳偏差，成功提升了模型性能。

    

    变分量子算法作为嘈杂中等规模量子（NISQ）设备的早期应用正受到关注。变分方法的主要问题之一在于Barren Plateaus现象，在变分参数优化过程中存在。提出将几何归纳偏差添加到量子模型作为缓解这一问题的潜在解决方案，从而导致了一个称为几何量子机器学习的新领域。本文引入了一种等变结构的变分量子分类器，以创建具有$C_4$旋转标签对称性的图像分类的标签不变模型。等变电路与两种不同的结构进行了基准测试，实验证明几何方法提升了模型的性能。最后，提出了经典等变卷积操作，以扩展量子模型处理更大图像的能力。

    arXiv:2403.15031v1 Announce Type: cross  Abstract: Variational quantum algorithms are gaining attention as an early application of Noisy Intermediate-Scale Quantum (NISQ) devices. One of the main problems of variational methods lies in the phenomenon of Barren Plateaus, present in the optimization of variational parameters. Adding geometric inductive bias to the quantum models has been proposed as a potential solution to mitigate this problem, leading to a new field called Geometric Quantum Machine Learning. In this work, an equivariant architecture for variational quantum classifiers is introduced to create a label-invariant model for image classification with $C_4$ rotational label symmetry. The equivariant circuit is benchmarked against two different architectures, and it is experimentally observed that the geometric approach boosts the model's performance. Finally, a classical equivariant convolution operation is proposed to extend the quantum model for the processing of larger ima
    
[^3]: PALM：推进用于持续测试时间自适应的自适应学习率机制

    PALM: Pushing Adaptive Learning Rate Mechanisms for Continual Test-Time Adaptation

    [https://arxiv.org/abs/2403.10650](https://arxiv.org/abs/2403.10650)

    本研究通过对模型预测不确定性的量化来选择需要进一步适应的层，从而克服了持续测试时间自适应方法中由于伪标签引起的不准确性困扰。

    

    实际环境中的视觉模型面临领域分布的快速转变，导致识别性能下降。持续测试时间自适应（CTTA）直接根据测试数据调整预训练的源判别模型以适应这些不断变化的领域。一种高度有效的CTTA方法涉及应用逐层自适应学习率，并选择性地调整预训练层。然而，它受到领域转移估计不准确和由伪标签引起的不准确性所困扰。在这项工作中，我们旨在通过识别层来克服这些限制，通过对模型预测不确定性的量化来选择层，而无须依赖伪标签。我们利用梯度的大小作为一个度量标准，通过反向传播softmax输出与均匀分布之间的KL散度来计算，以选择需要进一步适应的层。随后，仅属于这些层的参数将被进一步适应。

    arXiv:2403.10650v1 Announce Type: cross  Abstract: Real-world vision models in dynamic environments face rapid shifts in domain distributions, leading to decreased recognition performance. Continual test-time adaptation (CTTA) directly adjusts a pre-trained source discriminative model to these changing domains using test data. A highly effective CTTA method involves applying layer-wise adaptive learning rates, and selectively adapting pre-trained layers. However, it suffers from the poor estimation of domain shift and the inaccuracies arising from the pseudo-labels. In this work, we aim to overcome these limitations by identifying layers through the quantification of model prediction uncertainty without relying on pseudo-labels. We utilize the magnitude of gradients as a metric, calculated by backpropagating the KL divergence between the softmax output and a uniform distribution, to select layers for further adaptation. Subsequently, for the parameters exclusively belonging to these se
    
[^4]: 通过视觉检测模型增强多模态大型语言模型：一项实证研究

    Enhancing Multimodal Large Language Models with Vision Detection Models: An Empirical Study

    [https://arxiv.org/abs/2401.17981](https://arxiv.org/abs/2401.17981)

    本文通过实证研究，将最先进的目标检测和光学字符识别模型与多模态大型语言模型相结合，在提高图像理解和减少回答错误插入方面取得了显著改进。

    

    尽管多模态大型语言模型（MLLMs）在集成文本和图像模态方面具有令人印象深刻的能力，但在准确解释细节视觉元素方面仍存在挑战。本文通过将最先进的目标检测和光学字符识别模型与MLLMs结合，进行实证研究，旨在提高对细粒度图像理解，并减少回答中的错误插入。我们的研究探讨了基于嵌入的检测信息的融合，这种融合对MLLMs的原始能力的影响，以及检测模型的可互换性。我们对LLaVA-1.5、DINO和PaddleOCRv2等模型进行了系统实验，发现我们的方法不仅改善了MLLMs在特定视觉任务中的性能，而且保持了它们的原始优势。通过在10个基准测试中，增强的MLLMs在9个测试中超越了最先进模型，标准化平均得分提升高达12.99%，取得了显著的改进。

    Despite the impressive capabilities of Multimodal Large Language Models (MLLMs) in integrating text and image modalities, challenges remain in accurately interpreting detailed visual elements. This paper presents an empirical study on enhancing MLLMs with state-of-the-art (SOTA) object detection and Optical Character Recognition models to improve fine-grained image understanding and reduce hallucination in responses. Our research investigates the embedding-based infusion of detection information, the impact of such infusion on the MLLMs' original abilities, and the interchangeability of detection models. We conduct systematic experiments with models such as LLaVA-1.5, DINO, and PaddleOCRv2, revealing that our approach not only refines MLLMs' performance in specific visual tasks but also maintains their original strengths. The resulting enhanced MLLMs outperform SOTA models on 9 out of 10 benchmarks, achieving an improvement of up to 12.99% on the normalized average score, marking a not
    

