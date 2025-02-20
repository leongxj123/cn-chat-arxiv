# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Describe-and-Dissect: Interpreting Neurons in Vision Networks with Language Models](https://arxiv.org/abs/2403.13771) | 提出了一种新颖的方法 Describe-and-Dissect（DnD），利用多模态深度学习生成复杂的自然语言描述，无需标记的训练数据或预定义的概念选择，并且通过广泛的定性和定量分析显示优于先前的工作。 |
| [^2] | [Image-Text Matching with Multi-View Attention](https://arxiv.org/abs/2402.17237) | 本研究提出了一种使用多视图注意力的双流图像文本匹配方法，以解决单一表示难以全面覆盖复杂内容和缺乏交互的挑战。 |
| [^3] | [Customize-It-3D: High-Quality 3D Creation from A Single Image Using Subject-Specific Knowledge Prior.](http://arxiv.org/abs/2312.11535) | 该论文提出了一种使用主体特定知识先验的两阶段方法，通过考虑阴影模式和纹理增强来生成高质量、有丰富纹理的3D模型，与以前的方法相比具有显著的优势。 |
| [^4] | [PILOT: A Pre-Trained Model-Based Continual Learning Toolbox.](http://arxiv.org/abs/2309.07117) | 本论文介绍了一个名为PILOT的基于预训练模型的持续学习工具箱，为在处理流式数据并适应新数据到来的现实场景中，利用预训练模型进行增量学习提供了一种有前景的方法。 |
| [^5] | [MRI Field-transfer Reconstruction with Limited Data: Regularization by Neural Style Transfer.](http://arxiv.org/abs/2308.10968) | 本论文通过使用神经风格转换进行正规化，实现了在有限数据条件下从低质量图像重建高质量图像的目标。实验结果验证了该方法在临床MRI扫描中的有效性和潜力。 |
| [^6] | [CARSO: Counter-Adversarial Recall of Synthetic Observations.](http://arxiv.org/abs/2306.06081) | 本文提出了一种新的图像分类的对抗性防御机制CARSO，该方法可以比最先进的对抗性训练更好地保护分类器，通过利用生成模型进行对抗净化来进行最终分类，并成功地保护自己免受未预见的威胁和最终攻击。 |
| [^7] | [Layer-level activation mechanism.](http://arxiv.org/abs/2306.04940) | 去噪声更好，表现更好的分层级别激活机制 |

# 详细

[^1]: 使用语言模型解释视觉网络中的神经元：描述与解剖

    Describe-and-Dissect: Interpreting Neurons in Vision Networks with Language Models

    [https://arxiv.org/abs/2403.13771](https://arxiv.org/abs/2403.13771)

    提出了一种新颖的方法 Describe-and-Dissect（DnD），利用多模态深度学习生成复杂的自然语言描述，无需标记的训练数据或预定义的概念选择，并且通过广泛的定性和定量分析显示优于先前的工作。

    

    在本文中，我们提出了Describe-and-Dissect（DnD），一种新颖的方法，用于描述视觉网络中隐藏神经元的作用。DnD利用多模态深度学习的最新进展，生成复杂的自然语言描述，无需标记的训练数据或预定义的概念选择。此外，DnD是无需训练的，这意味着我们不训练任何新模型，未来可以轻松利用更强大的通用模型。我们进行了广泛的定性和定量分析，表明DnD通过提供更高质量的神经元描述优于先前的工作。具体而言，我们的方法平均提供最高质量的标签，并且被选为神经元的最佳解释的概率是最佳基线的两倍多。

    arXiv:2403.13771v1 Announce Type: cross  Abstract: In this paper, we propose Describe-and-Dissect (DnD), a novel method to describe the roles of hidden neurons in vision networks. DnD utilizes recent advancements in multimodal deep learning to produce complex natural language descriptions, without the need for labeled training data or a predefined set of concepts to choose from. Additionally, DnD is training-free, meaning we don't train any new models and can easily leverage more capable general purpose models in the future. We have conducted extensive qualitative and quantitative analysis to show that DnD outperforms prior work by providing higher quality neuron descriptions. Specifically, our method on average provides the highest quality labels and is more than 2 times as likely to be selected as the best explanation for a neuron than the best baseline.
    
[^2]: 使用多视图注意力的图像文本匹配

    Image-Text Matching with Multi-View Attention

    [https://arxiv.org/abs/2402.17237](https://arxiv.org/abs/2402.17237)

    本研究提出了一种使用多视图注意力的双流图像文本匹配方法，以解决单一表示难以全面覆盖复杂内容和缺乏交互的挑战。

    

    现有的用于图像文本匹配的双流模型在确保检索速度的同时表现出良好的性能，并受到工业界和学术界的广泛关注。这些方法使用单一表示来分别编码图像和文本，并使用余弦相似度或向量内积得到匹配分数。然而，双流模型的性能往往不太理想。一方面，单一表示难以全面覆盖复杂内容。另一方面，在这种缺乏交互的框架中，匹配多重含义是具有挑战性的，这导致信息被忽略。为了解决上述问题并促进双流模型的性能，我们提出了一种双流图像文本匹配的多视图注意力方法MVAM（多视图注意力模型）。

    arXiv:2402.17237v1 Announce Type: cross  Abstract: Existing two-stream models for image-text matching show good performance while ensuring retrieval speed and have received extensive attention from industry and academia. These methods use a single representation to encode image and text separately and get a matching score with cosine similarity or the inner product of vectors. However, the performance of the two-stream model is often sub-optimal. On the one hand, a single representation is challenging to cover complex content comprehensively. On the other hand, in this framework of lack of interaction, it is challenging to match multiple meanings which leads to information being ignored. To address the problems mentioned above and facilitate the performance of the two-stream model, we propose a multi-view attention approach for two-stream image-text matching MVAM (\textbf{M}ulti-\textbf{V}iew \textbf{A}ttention \textbf{M}odel). It first learns multiple image and text representations by
    
[^3]: Customize-It-3D：使用主体特定知识先验从单个图像创建高质量的3D模型

    Customize-It-3D: High-Quality 3D Creation from A Single Image Using Subject-Specific Knowledge Prior. (arXiv:2312.11535v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2312.11535](http://arxiv.org/abs/2312.11535)

    该论文提出了一种使用主体特定知识先验的两阶段方法，通过考虑阴影模式和纹理增强来生成高质量、有丰富纹理的3D模型，与以前的方法相比具有显著的优势。

    

    在本文中，我们提出了一种新的两阶段方法，充分利用参考图像提供的信息，建立图像到3D生成的自定义知识先验。之前的方法主要依赖于通用的扩散先验，这些方法在与参考图像得到一致结果方面存在困难，我们提出了一种主体特定且多模态的扩散模型。该模型不仅通过考虑阴影模式来改善几何优化和纹理增强的粗略结果，还有助于使3D内容与主题保持一致。大量实验证明了我们的方法的优越性，Customize-It-3D在视觉质量上远远超过了以前的方法，能够产生出色的360度重建结果，非常适合各种应用，包括文本到3D的创建。

    In this paper, we present a novel two-stage approach that fully utilizes the information provided by the reference image to establish a customized knowledge prior for image-to-3D generation. While previous approaches primarily rely on a general diffusion prior, which struggles to yield consistent results with the reference image, we propose a subject-specific and multi-modal diffusion model. This model not only aids NeRF optimization by considering the shading mode for improved geometry but also enhances texture from the coarse results to achieve superior refinement. Both aspects contribute to faithfully aligning the 3D content with the subject. Extensive experiments showcase the superiority of our method, Customize-It-3D, outperforming previous works by a substantial margin. It produces faithful 360-degree reconstructions with impressive visual quality, making it well-suited for various applications, including text-to-3D creation.
    
[^4]: PILOT：一个基于预训练模型的持续学习工具箱

    PILOT: A Pre-Trained Model-Based Continual Learning Toolbox. (arXiv:2309.07117v1 [cs.LG])

    [http://arxiv.org/abs/2309.07117](http://arxiv.org/abs/2309.07117)

    本论文介绍了一个名为PILOT的基于预训练模型的持续学习工具箱，为在处理流式数据并适应新数据到来的现实场景中，利用预训练模型进行增量学习提供了一种有前景的方法。

    

    传统机器学习可以有效地解决各种问题，但主要在封闭环境中运作，处理流式数据时存在局限性。作为解决方案，增量学习应运而生，用于处理涉及新数据到来的现实场景。最近，预训练在不断取得重要进展，并引起了众多研究人员的关注。这些预训练模型（PTMs）的强大性能为开发能够有效适应现实场景的持续学习算法提供了有希望的途径。因此，探索在增量学习中利用PTMs已经成为必需。本文介绍了一个名为PILOT的基于预训练模型的持续学习工具箱。一方面，PILOT实施了一些基于预训练模型的最新班级增量学习算法，如L2P、DualPrompt和CODA-Prompt。另一方面，PILOT也适应了典型的班级增量学习场景。

    While traditional machine learning can effectively tackle a wide range of problems, it primarily operates within a closed-world setting, which presents limitations when dealing with streaming data. As a solution, incremental learning emerges to address real-world scenarios involving new data's arrival. Recently, pre-training has made significant advancements and garnered the attention of numerous researchers. The strong performance of these pre-trained models (PTMs) presents a promising avenue for developing continual learning algorithms that can effectively adapt to real-world scenarios. Consequently, exploring the utilization of PTMs in incremental learning has become essential. This paper introduces a pre-trained model-based continual learning toolbox known as PILOT. On the one hand, PILOT implements some state-of-the-art class-incremental learning algorithms based on pre-trained models, such as L2P, DualPrompt, and CODA-Prompt. On the other hand, PILOT also fits typical class-incre
    
[^5]: 使用有限数据的MRI场转移重建：通过神经风格转换进行正规化

    MRI Field-transfer Reconstruction with Limited Data: Regularization by Neural Style Transfer. (arXiv:2308.10968v1 [cs.CV])

    [http://arxiv.org/abs/2308.10968](http://arxiv.org/abs/2308.10968)

    本论文通过使用神经风格转换进行正规化，实现了在有限数据条件下从低质量图像重建高质量图像的目标。实验结果验证了该方法在临床MRI扫描中的有效性和潜力。

    

    最近的研究表明，使用基于深度学习模型的MRI重建取得了成功。然而，大多数报告的方法都需要在特定任务的大规模数据集上进行训练。通过降噪（RED）正规化是一种将降噪器作为图像重建先验的通用流程。RED的潜力已经在多个与图像相关的任务（如降噪、去模糊和超分辨率）中得到了证明。本文提出了一种通过神经风格转换（RNST）方法进行正规化的方法，进一步利用神经转移和降噪引擎的先验信息。这使得RNST能够从有噪声的低质量图像中重建出高质量图像，图像风格和有限数据不同。我们使用1.5T和3T的临床MRI扫描验证了RNST，并且显示RNST可以显著提高图像质量。我们的结果突显了RNST框架在MRI重建和有限数据重建任务中的能力。

    Recent works have demonstrated success in MRI reconstruction using deep learning-based models. However, most reported approaches require training on a task-specific, large-scale dataset. Regularization by denoising (RED) is a general pipeline which embeds a denoiser as a prior for image reconstruction. The potential of RED has been demonstrated for multiple image-related tasks such as denoising, deblurring and super-resolution. In this work, we propose a regularization by neural style transfer (RNST) method to further leverage the priors from the neural transfer and denoising engine. This enables RNST to reconstruct a high-quality image from a noisy low-quality image with different image styles and limited data. We validate RNST with clinical MRI scans from 1.5T and 3T and show that RNST can significantly boost image quality. Our results highlight the capability of the RNST framework for MRI reconstruction and the potential for reconstruction tasks with limited data.
    
[^6]: CARSO: 对抗性合成观测的反对抗性召回

    CARSO: Counter-Adversarial Recall of Synthetic Observations. (arXiv:2306.06081v1 [cs.CV])

    [http://arxiv.org/abs/2306.06081](http://arxiv.org/abs/2306.06081)

    本文提出了一种新的图像分类的对抗性防御机制CARSO，该方法可以比最先进的对抗性训练更好地保护分类器，通过利用生成模型进行对抗净化来进行最终分类，并成功地保护自己免受未预见的威胁和最终攻击。

    

    本文提出了一种新的对抗性防御机制CARSO，用于图像分类，灵感来自认知神经科学的线索。该方法与对抗训练具有协同互补性，并依赖于被攻击分类器的内部表示的知识。通过利用生成模型进行对抗净化，该方法采样输入的重构来进行最终分类。在各种图像数据集和分类器体系结构上进行的实验评估表明，CARSO能够比最先进的对抗性训练更好地保护分类器——同时具有可接受的清洁准确度损失。此外，防御体系结构成功地保护自己免受未预见的威胁和最终攻击。代码和预训练模型可在https://github.com/获得。

    In this paper, we propose a novel adversarial defence mechanism for image classification -- CARSO -- inspired by cues from cognitive neuroscience. The method is synergistically complementary to adversarial training and relies on knowledge of the internal representation of the attacked classifier. Exploiting a generative model for adversarial purification, conditioned on such representation, it samples reconstructions of inputs to be finally classified. Experimental evaluation by a well-established benchmark of varied, strong adaptive attacks, across diverse image datasets and classifier architectures, shows that CARSO is able to defend the classifier significantly better than state-of-the-art adversarial training alone -- with a tolerable clean accuracy toll. Furthermore, the defensive architecture succeeds in effectively shielding itself from unforeseen threats, and end-to-end attacks adapted to fool stochastic defences. Code and pre-trained models are available at https://github.com/
    
[^7]: 分层级别激活机制

    Layer-level activation mechanism. (arXiv:2306.04940v1 [cs.LG])

    [http://arxiv.org/abs/2306.04940](http://arxiv.org/abs/2306.04940)

    去噪声更好，表现更好的分层级别激活机制

    

    本文提出了一种新颖的激活机制，旨在建立分层级别激活功能（LayerAct）。这些功能旨在通过减少输入偏移所导致的激活输出的分层级波动来降低传统元素级激活功能的噪音鲁棒性。此外，LayerAct功能实现了类似于零的平均激活输出，而不限制激活输出空间。我们进行了分析和实验，证明LayerAct功能在噪声鲁棒性方面优于元素级激活功能，并且经验证明这些功能的平均激活结果类似于零。在三个基准图像分类任务的实验结果表明，在处理嘈杂的图像数据集时，LayerAct功能比元素级激活功能表现更好，而在大多数情况下，清洁数据集的表现也是优越的。

    In this work, we propose a novel activation mechanism aimed at establishing layer-level activation (LayerAct) functions. These functions are designed to be more noise-robust compared to traditional element-level activation functions by reducing the layer-level fluctuation of the activation outputs due to shift in inputs. Moreover, the LayerAct functions achieve a zero-like mean activation output without restricting the activation output space. We present an analysis and experiments demonstrating that LayerAct functions exhibit superior noise-robustness compared to element-level activation functions, and empirically show that these functions have a zero-like mean activation. Experimental results on three benchmark image classification tasks show that LayerAct functions excel in handling noisy image datasets, outperforming element-level activation functions, while the performance on clean datasets is also superior in most cases.
    

