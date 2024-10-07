# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AID: Attention Interpolation of Text-to-Image Diffusion](https://arxiv.org/abs/2403.17924) | 提出了一种新颖的无训练技术，即Attention Interpolation via Diffusion (AID)，通过内/外插值注意力层、插值注意力与自注意力融合以提高保真度，以及应用贝塔分布进行选择以增加平滑度来改进文本到图像插值的问题。 |
| [^2] | [Exploiting Semantic Reconstruction to Mitigate Hallucinations in Vision-Language Models](https://arxiv.org/abs/2403.16167) | 通过准确定位和惩罚幻觉标记，ESREAL引入了一种新颖的无监督学习框架，通过语义重建来抑制生成幻觉，解决了视觉-语言模型中幻觉问题。 |
| [^3] | [Data-centric Prediction Explanation via Kernelized Stein Discrepancy](https://arxiv.org/abs/2403.15576) | 该论文提出了一种基于内核化斯坦不相容性的数据中心预测解释方法，通过利用内核函数识别提供最佳预测支持给测试点的训练样本，取得了优异性能。 |
| [^4] | [Finer: Investigating and Enhancing Fine-Grained Visual Concept Recognition in Large Vision Language Models](https://arxiv.org/abs/2402.16315) | Finer工作揭示了大型视觉语言模型在细粒度视觉分类上的短板，尤其是难以生成准确的细致属性解释，尽管具有生成高水平图像解释的能力。 |
| [^5] | [CommVQA: Situating Visual Question Answering in Communicative Contexts](https://arxiv.org/abs/2402.15002) | CommVQA数据集将图像置于自然环境中，挑战了当前的VQA模型，结果表明为模型提供上下文信息能够提高性能。 |
| [^6] | [Statistical Test for Generated Hypotheses by Diffusion Models](https://arxiv.org/abs/2402.11789) | 本研究提出了一种统计检验方法，通过选择性推断框架，在考虑生成图像是由训练的扩散模型产生的条件下，量化医学图像诊断结果的可靠性。 |
| [^7] | [MITS-GAN: Safeguarding Medical Imaging from Tampering with Generative Adversarial Networks.](http://arxiv.org/abs/2401.09624) | MITS-GAN是一种用于保护医学影像免受篡改的新方法，通过引入适当的高斯噪声作为防护措施，打乱攻击者的生成对抗网络架构的输出。实验结果表明MITS-GAN能够生成耐篡改图像，具有优越的性能。 |
| [^8] | [MosaicFusion: Diffusion Models as Data Augmenters for Large Vocabulary Instance Segmentation.](http://arxiv.org/abs/2309.13042) | MosaicFusion是一种用于大词汇实例分割的数据增强方法，通过将扩散模型作为数据集生成器，能够生成大量合成标记数据。在实验中，我们的方法在准确率和泛化能力方面取得了显著的提升。 |
| [^9] | [Scattering Spectra Models for Physics.](http://arxiv.org/abs/2306.17210) | 本文介绍了物理学中的散射谱模型，用于描述各种场的统计特性。这些模型基于散射系数的协方差，结合了场的小波分解和点位模，能够准确且稳健地重现标准统计量，捕捉了关键特性。 |

# 详细

[^1]: AID: 文本到图像扩散的注重插值

    AID: Attention Interpolation of Text-to-Image Diffusion

    [https://arxiv.org/abs/2403.17924](https://arxiv.org/abs/2403.17924)

    提出了一种新颖的无训练技术，即Attention Interpolation via Diffusion (AID)，通过内/外插值注意力层、插值注意力与自注意力融合以提高保真度，以及应用贝塔分布进行选择以增加平滑度来改进文本到图像插值的问题。

    

    arXiv:2403.17924v1 公告类型: 跨领域 摘要: 有条件的扩散模型可以在各种设置中创建看不见的图像，有助于图像插值。潜在空间中的插值已经得到了深入研究，但是具有特定条件（如文本或姿势）的插值却了解不多。简单的方法，如在条件空间中的线性插值，通常会导致图像缺乏一致性、平滑度和保真度。为此，我们引入一个名为Attention Interpolation via Diffusion (AID)的新颖无训练技术。我们的主要贡献包括1）提出了一个内/外插值注意力层；2）将插值注意力与自注意力融合以提高保真度；3）应用贝塔分布进行选择以增加平滑度。我们还提出了一种变体，Prompt-guided Attention Interpolation via Diffusion (PAID)，它将插值视为依赖于条件的生成过程。这种方法可以创建出更具创造性的新图像。

    arXiv:2403.17924v1 Announce Type: cross  Abstract: Conditional diffusion models can create unseen images in various settings, aiding image interpolation. Interpolation in latent spaces is well-studied, but interpolation with specific conditions like text or poses is less understood. Simple approaches, such as linear interpolation in the space of conditions, often result in images that lack consistency, smoothness, and fidelity. To that end, we introduce a novel training-free technique named Attention Interpolation via Diffusion (AID). Our key contributions include 1) proposing an inner/outer interpolated attention layer; 2) fusing the interpolated attention with self-attention to boost fidelity; and 3) applying beta distribution to selection to increase smoothness. We also present a variant, Prompt-guided Attention Interpolation via Diffusion (PAID), that considers interpolation as a condition-dependent generative process. This method enables the creation of new images with greater con
    
[^2]: 利用语义重建减少视觉-语言模型中的幻觉

    Exploiting Semantic Reconstruction to Mitigate Hallucinations in Vision-Language Models

    [https://arxiv.org/abs/2403.16167](https://arxiv.org/abs/2403.16167)

    通过准确定位和惩罚幻觉标记，ESREAL引入了一种新颖的无监督学习框架，通过语义重建来抑制生成幻觉，解决了视觉-语言模型中幻觉问题。

    

    视觉-语言模型中的幻觉对其可靠性构成重大挑战，特别是在生成长标题时。当前方法无法准确识别和减轻这些幻觉。为了解决这个问题，我们引入了ESREAL，这是一个新颖的无监督学习框架，旨在通过准确定位和惩罚幻觉标记来抑制幻觉生成。最初，ESREAL根据生成的标题创建一个重建图像，并将其对应区域与原始图像的区域对齐。这种语义重建有助于识别生成标题中的标记级幻觉的存在和类型。随后，ESREAL通过评估对齐区域的语义相似性来计算标记级幻觉分数，基于幻觉的类型。最后，ESREAL采用一种近端策略优化算法，进行...

    arXiv:2403.16167v1 Announce Type: cross  Abstract: Hallucinations in vision-language models pose a significant challenge to their reliability, particularly in the generation of long captions. Current methods fall short of accurately identifying and mitigating these hallucinations. To address this issue, we introduce ESREAL, a novel unsupervised learning framework designed to suppress the generation of hallucinations through accurate localization and penalization of hallucinated tokens. Initially, ESREAL creates a reconstructed image based on the generated caption and aligns its corresponding regions with those of the original image. This semantic reconstruction aids in identifying both the presence and type of token-level hallucinations within the generated caption. Subsequently, ESREAL computes token-level hallucination scores by assessing the semantic similarity of aligned regions based on the type of hallucination. Finally, ESREAL employs a proximal policy optimization algorithm, wh
    
[^3]: 基于内核化斯坦不相容性的数据中心预测解释

    Data-centric Prediction Explanation via Kernelized Stein Discrepancy

    [https://arxiv.org/abs/2403.15576](https://arxiv.org/abs/2403.15576)

    该论文提出了一种基于内核化斯坦不相容性的数据中心预测解释方法，通过利用内核函数识别提供最佳预测支持给测试点的训练样本，取得了优异性能。

    

    现有的基于示例的预测解释方法通常通过模型的参数或潜在表示来连接测试和训练数据点。尽管这些方法提供了有关模型预测原因的线索，但它们经常表现出固有的缺陷，比如产生显着的计算开销或生成粗粒度的解释。本文提出了一种高精度和数据中心的解释（HD-Explain），这是一种利用内核化斯坦不相容性（KSD）属性的简单预测解释方法。具体来说，KSD唯一地为经过训练的模型定义了一个参数化的内核函数，用于编码与模型相关的数据相关性。通过利用内核函数，可以有效地识别提供最佳预测支持给测试点的训练样本。我们在多个分类领域进行了彻底的分析和实验，结果表明HD-Explain取得了优异的性能。

    arXiv:2403.15576v1 Announce Type: new  Abstract: Existing example-based prediction explanation methods often bridge test and training data points through the model's parameters or latent representations. While these methods offer clues to the causes of model predictions, they often exhibit innate shortcomings, such as incurring significant computational overhead or producing coarse-grained explanations. This paper presents a Highly-precise and Data-centric Explanation (HD-Explain), a straightforward prediction explanation method exploiting properties of Kernelized Stein Discrepancy (KSD). Specifically, the KSD uniquely defines a parameterized kernel function for a trained model that encodes model-dependent data correlation. By leveraging the kernel function, one can identify training samples that provide the best predictive support to a test point efficiently. We conducted thorough analyses and experiments across multiple classification domains, where we show that HD-Explain outperform
    
[^4]: Finer: 在大型视觉语言模型中研究和增强细粒度视觉概念识别

    Finer: Investigating and Enhancing Fine-Grained Visual Concept Recognition in Large Vision Language Models

    [https://arxiv.org/abs/2402.16315](https://arxiv.org/abs/2402.16315)

    Finer工作揭示了大型视觉语言模型在细粒度视觉分类上的短板，尤其是难以生成准确的细致属性解释，尽管具有生成高水平图像解释的能力。

    

    最近指导调整的大型视觉语言模型（LVLMs）的进展使模型能够轻松生成高水平的基于图像的解释。尽管这种能力主要归因于大型语言模型（LLMs）中包含的丰富世界知识，但我们的工作揭示了它们在六个不同基准设置下的细粒度视觉分类（FGVC）上的缺陷。最近的LVLMs最先进的模型，如LLaVa-1.5，InstructBLIP和GPT-4V，在分类性能方面严重下降，例如，LLaVA-1.5在斯坦福狗的EM平均下降了65.58，而且还难以根据出现在输入图像中的概念生成具有详细属性的准确解释，尽管它们有生成整体图像级描述的能力。深入分析表明，经过指导调整的LVLMs在给定文本时呈现出模态差距，显示出存在不一致性

    arXiv:2402.16315v1 Announce Type: cross  Abstract: Recent advances in instruction-tuned Large Vision-Language Models (LVLMs) have imbued the models with the ability to generate high-level, image-grounded explanations with ease. While such capability is largely attributed to the rich world knowledge contained within the Large Language Models (LLMs), our work reveals their shortcomings in fine-grained visual categorization (FGVC) across six different benchmark settings. Most recent state-of-the-art LVLMs like LLaVa-1.5, InstructBLIP and GPT-4V not only severely deteriorate in terms of classification performance, e.g., average drop of 65.58 in EM for Stanford Dogs for LLaVA-1.5, but also struggle to generate an accurate explanation with detailed attributes based on the concept that appears within an input image despite their capability to generate holistic image-level descriptions. In-depth analyses show that instruction-tuned LVLMs exhibit modality gap, showing discrepancy when given tex
    
[^5]: 将视觉问答置于交际背景中的CommVQA

    CommVQA: Situating Visual Question Answering in Communicative Contexts

    [https://arxiv.org/abs/2402.15002](https://arxiv.org/abs/2402.15002)

    CommVQA数据集将图像置于自然环境中，挑战了当前的VQA模型，结果表明为模型提供上下文信息能够提高性能。

    

    当前的视觉问答（VQA）模型往往在孤立的图像-问题对上进行训练和评估。然而，人们提出的问题取决于他们的信息需求和对图像内容的先前了解。为了评估将图像置于自然环境中如何塑造视觉问题，我们引入了CommVQA，这是一个包含图像、图像描述、图像可能出现的真实交际场景（例如旅行网站）以及依赖于场景的后续问题和答案的VQA数据集。我们展示了CommVQA对当前模型提出了挑战。为VQA模型提供上下文信息可广泛提高性能，突显将系统置于交际场景中的相关性。

    arXiv:2402.15002v1 Announce Type: new  Abstract: Current visual question answering (VQA) models tend to be trained and evaluated on image-question pairs in isolation. However, the questions people ask are dependent on their informational needs and prior knowledge about the image content. To evaluate how situating images within naturalistic contexts shapes visual questions, we introduce CommVQA, a VQA dataset consisting of images, image descriptions, real-world communicative scenarios where the image might appear (e.g., a travel website), and follow-up questions and answers conditioned on the scenario. We show that CommVQA poses a challenge for current models. Providing contextual information to VQA models improves performance broadly, highlighting the relevance of situating systems within a communicative scenario.
    
[^6]: 通过扩散模型生成的假设的统计检验

    Statistical Test for Generated Hypotheses by Diffusion Models

    [https://arxiv.org/abs/2402.11789](https://arxiv.org/abs/2402.11789)

    本研究提出了一种统计检验方法，通过选择性推断框架，在考虑生成图像是由训练的扩散模型产生的条件下，量化医学图像诊断结果的可靠性。

    

    AI的增强性能加速了其融入科学研究。特别是，利用生成式AI创建科学假设是很有前途的，并且正在越来越多地应用于各个领域。然而，当使用AI生成的假设进行关键决策（如医学诊断）时，验证它们的可靠性至关重要。在本研究中，我们考虑使用扩散模型生成的图像进行医学诊断任务，并提出了一种统计检验来量化其可靠性。所提出的统计检验的基本思想是使用选择性推断框架，我们考虑在生成的图像是由经过训练的扩散模型产生的这一事实条件下的统计检验。利用所提出的方法，医学图像诊断结果的统计可靠性可以以p值的形式量化，从而实现在控制错误率的情况下进行决策。

    arXiv:2402.11789v1 Announce Type: cross  Abstract: The enhanced performance of AI has accelerated its integration into scientific research. In particular, the use of generative AI to create scientific hypotheses is promising and is increasingly being applied across various fields. However, when employing AI-generated hypotheses for critical decisions, such as medical diagnoses, verifying their reliability is crucial. In this study, we consider a medical diagnostic task using generated images by diffusion models, and propose a statistical test to quantify its reliability. The basic idea behind the proposed statistical test is to employ a selective inference framework, where we consider a statistical test conditional on the fact that the generated images are produced by a trained diffusion model. Using the proposed method, the statistical reliability of medical image diagnostic results can be quantified in the form of a p-value, allowing for decision-making with a controlled error rate. 
    
[^7]: MITS-GAN: 用生成对抗网络保护医学影像免受篡改

    MITS-GAN: Safeguarding Medical Imaging from Tampering with Generative Adversarial Networks. (arXiv:2401.09624v1 [eess.IV])

    [http://arxiv.org/abs/2401.09624](http://arxiv.org/abs/2401.09624)

    MITS-GAN是一种用于保护医学影像免受篡改的新方法，通过引入适当的高斯噪声作为防护措施，打乱攻击者的生成对抗网络架构的输出。实验结果表明MITS-GAN能够生成耐篡改图像，具有优越的性能。

    

    生成模型，特别是生成对抗网络（GANs），在图像生成方面取得了进展，但也引发了潜在的恶意使用的担忧，尤其是在医学影像等敏感领域。本研究提出了一种新颖的方法MITS-GAN，用于防止医学影像中的篡改，特别关注CT扫描。该方法通过引入不可察觉但精确的扰动来打乱攻击者的CT-GAN架构的输出。具体而言，所提出的方法涉及将适当的高斯噪声引入到输入中作为对各种攻击的保护措施。我们的方法旨在提高防篡改能力，与现有技术相比具有优势。对CT扫描数据集的实验结果表明MITS-GAN具有卓越的性能，强调了其能够生成具有可忽略伪影的耐篡改图像的能力。由于医学领域中的图像篡改带来了危及生命的风险，我们的主动防护方法具有重要意义。

    The progress in generative models, particularly Generative Adversarial Networks (GANs), opened new possibilities for image generation but raised concerns about potential malicious uses, especially in sensitive areas like medical imaging. This study introduces MITS-GAN, a novel approach to prevent tampering in medical images, with a specific focus on CT scans. The approach disrupts the output of the attacker's CT-GAN architecture by introducing imperceptible but yet precise perturbations. Specifically, the proposed approach involves the introduction of appropriate Gaussian noise to the input as a protective measure against various attacks. Our method aims to enhance tamper resistance, comparing favorably to existing techniques. Experimental results on a CT scan dataset demonstrate MITS-GAN's superior performance, emphasizing its ability to generate tamper-resistant images with negligible artifacts. As image tampering in medical domains poses life-threatening risks, our proactive approac
    
[^8]: MosaicFusion: 将扩散模型作为大词汇实例分割的数据增强器

    MosaicFusion: Diffusion Models as Data Augmenters for Large Vocabulary Instance Segmentation. (arXiv:2309.13042v1 [cs.CV])

    [http://arxiv.org/abs/2309.13042](http://arxiv.org/abs/2309.13042)

    MosaicFusion是一种用于大词汇实例分割的数据增强方法，通过将扩散模型作为数据集生成器，能够生成大量合成标记数据。在实验中，我们的方法在准确率和泛化能力方面取得了显著的提升。

    

    我们提出了MosaicFusion，一种简单而有效的基于扩散的数据增强方法，用于大词汇实例分割。我们的方法无需训练，也不依赖于任何标签监督。两个关键设计使我们能够将现成的文本到图像扩散模型作为有用的数据集生成器，用于对象实例和蒙版注释。首先，我们将图像画布分为几个区域，并执行一轮扩散过程，同时基于不同的文本提示生成多个实例。其次，我们通过聚合与对象提示相关联的跨注意力图在层和扩散时间步上，然后进行简单的阈值处理和边缘感知的细化处理，得到相应的实例蒙版。我们的MosaicFusion可以为稀缺和新颖类别产生大量的合成标记数据，而无需复杂的处理。在具有挑战性的LVIS长尾和开放词汇基准上进行的实验结果表明，我们的方法在准确率和泛化能力方面均取得了显著的提升。

    We present MosaicFusion, a simple yet effective diffusion-based data augmentation approach for large vocabulary instance segmentation. Our method is training-free and does not rely on any label supervision. Two key designs enable us to employ an off-the-shelf text-to-image diffusion model as a useful dataset generator for object instances and mask annotations. First, we divide an image canvas into several regions and perform a single round of diffusion process to generate multiple instances simultaneously, conditioning on different text prompts. Second, we obtain corresponding instance masks by aggregating cross-attention maps associated with object prompts across layers and diffusion time steps, followed by simple thresholding and edge-aware refinement processing. Without bells and whistles, our MosaicFusion can produce a significant amount of synthetic labeled data for both rare and novel categories. Experimental results on the challenging LVIS long-tailed and open-vocabulary benchma
    
[^9]: 物理学的散射谱模型

    Scattering Spectra Models for Physics. (arXiv:2306.17210v1 [physics.data-an])

    [http://arxiv.org/abs/2306.17210](http://arxiv.org/abs/2306.17210)

    本文介绍了物理学中的散射谱模型，用于描述各种场的统计特性。这些模型基于散射系数的协方差，结合了场的小波分解和点位模，能够准确且稳健地重现标准统计量，捕捉了关键特性。

    

    物理学家常常需要概率模型来进行参数推断或生成一个场的新实现。针对高度非高斯场的建立这样的模型是一项挑战，特别是当样本数量有限时。在本文中，我们介绍了散射谱模型用于平稳场，并展示了它们在物理学中遇到的各种场的准确且稳健的统计描述。这些模型基于散射系数的协方差，即场的小波分解和点位模。在介绍利用旋转和缩放下场的规律性进行有用的维度约简后，我们验证了这些模型在不同多尺度物理场上的效果，并证明它们能够重现标准统计量，包括四阶空间矩。这些散射谱为我们提供了一种低维结构化表示，捕捉了关键特性。

    Physicists routinely need probabilistic models for a number of tasks such as parameter inference or the generation of new realizations of a field. Establishing such models for highly non-Gaussian fields is a challenge, especially when the number of samples is limited. In this paper, we introduce scattering spectra models for stationary fields and we show that they provide accurate and robust statistical descriptions of a wide range of fields encountered in physics. These models are based on covariances of scattering coefficients, i.e. wavelet decomposition of a field coupled with a point-wise modulus. After introducing useful dimension reductions taking advantage of the regularity of a field under rotation and scaling, we validate these models on various multi-scale physical fields and demonstrate that they reproduce standard statistics, including spatial moments up to 4th order. These scattering spectra provide us with a low-dimensional structured representation that captures key prop
    

