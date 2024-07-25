# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MambaMixer: Efficient Selective State Space Models with Dual Token and Channel Selection](https://arxiv.org/abs/2403.19888) | MambaMixer是一种新的架构，提出了具有数据依赖权重的双重选择机制，称为选择性标记和通道混合器，对长序列建模具有潜在优势。 |
| [^2] | [PathM3: A Multimodal Multi-Task Multiple Instance Learning Framework for Whole Slide Image Classification and Captioning](https://arxiv.org/abs/2403.08967) | 提出了一种 PathM3 框架，用于WSI分类和字幕生成，采用了多模态、多任务、多实例学习方法，通过查询式transformer有效对齐WSIs与诊断性字幕。 |
| [^3] | [MM-Soc: Benchmarking Multimodal Large Language Models in Social Media Platforms](https://arxiv.org/abs/2402.14154) | 该研究介绍了MM-Soc，一个旨在评估多模态大型语言模型（MLLMs）对社交媒体内容理解的综合基准，通过对十种大小变体的四个开源MLLMs进行详尽评估，发现了显著的性能差异。 |
| [^4] | [Copyright Protection in Generative AI: A Technical Perspective](https://arxiv.org/abs/2402.02333) | 本文从技术角度全面概述了在生成型人工智能中的版权保护问题，包括数据版权和模型版权两个方面，并提出了一些创新的方法和技术。 |
| [^5] | [Video Understanding with Large Language Models: A Survey.](http://arxiv.org/abs/2312.17432) | 这项调查研究提供了对大型语言模型（Vid-LLMs）在视频理解中的最新进展的详细概述。Vid-LLMs的新兴能力包括开放式时空推理和常识知识，为未来的视频理解提供了有前途的方向。 |
| [^6] | [EXACT: How to Train Your Accuracy.](http://arxiv.org/abs/2205.09615) | 本文提出了一种新的分类任务优化框架，通过引入随机性，优化期望准确率，取得了强有力的替代分类损失的结果。 |

# 详细

[^1]: MambaMixer：具有双重标记和通道选择的高效选择性状态空间模型

    MambaMixer: Efficient Selective State Space Models with Dual Token and Channel Selection

    [https://arxiv.org/abs/2403.19888](https://arxiv.org/abs/2403.19888)

    MambaMixer是一种新的架构，提出了具有数据依赖权重的双重选择机制，称为选择性标记和通道混合器，对长序列建模具有潜在优势。

    

    深度学习的最新进展主要依赖于Transformers，因为它们具有数据依赖性并且能够实现大规模学习。然而，这些架构中的注意力模块展现出输入大小的二次时间和空间，限制了它们用于长序列建模的可扩展性。尽管最近有尝试为多维数据设计高效有效的架构主干，例如图像和多变量时间序列，但现有模型要么是数据独立的，要么无法允许跨维度和内部维度之间的通信。最近，状态空间模型（SSMs），尤其是具有高效硬件感知实现的选择性状态空间模型，展现出了用于长序列建模的潜在优势。受到SSMs成功的启发，我们提出了MambaMixer，一种新的具有数据依赖权重的架构，使用跨标记和通道的双重选择机制，称为选择性标记和通道混合器。

    arXiv:2403.19888v1 Announce Type: cross  Abstract: Recent advances in deep learning have mainly relied on Transformers due to their data dependency and ability to learn at scale. The attention module in these architectures, however, exhibits quadratic time and space in input size, limiting their scalability for long-sequence modeling. Despite recent attempts to design efficient and effective architecture backbone for multi-dimensional data, such as images and multivariate time series, existing models are either data independent, or fail to allow inter- and intra-dimension communication. Recently, State Space Models (SSMs), and more specifically Selective State Space Models, with efficient hardware-aware implementation, have shown promising potential for long sequence modeling. Motivated by the success of SSMs, we present MambaMixer, a new architecture with data-dependent weights that uses a dual selection mechanism across tokens and channels, called Selective Token and Channel Mixer. M
    
[^2]: PathM3: 一种用于全切片图像分类和字幕生成的多模态多任务多实例学习框架

    PathM3: A Multimodal Multi-Task Multiple Instance Learning Framework for Whole Slide Image Classification and Captioning

    [https://arxiv.org/abs/2403.08967](https://arxiv.org/abs/2403.08967)

    提出了一种 PathM3 框架，用于WSI分类和字幕生成，采用了多模态、多任务、多实例学习方法，通过查询式transformer有效对齐WSIs与诊断性字幕。

    

    在计算组织病理学领域，整个切片图像（WSIs）和诊断性字幕都为做出诊断决策提供了宝贵的见解。然而，将WSIs与诊断性字幕对齐是一个重大挑战，主要源自两个因素：1）巨型像素WSIs不适合直接输入深度学习模型，而图块间的冗余性和相关性要求更多注意；2）真实的WSI诊断性字幕极其有限，使得难以训练出有效的模型。

    arXiv:2403.08967v1 Announce Type: cross  Abstract: In the field of computational histopathology, both whole slide images (WSIs) and diagnostic captions provide valuable insights for making diagnostic decisions. However, aligning WSIs with diagnostic captions presents a significant challenge. This difficulty arises from two main factors: 1) Gigapixel WSIs are unsuitable for direct input into deep learning models, and the redundancy and correlation among the patches demand more attention; and 2) Authentic WSI diagnostic captions are extremely limited, making it difficult to train an effective model. To overcome these obstacles, we present PathM3, a multimodal, multi-task, multiple instance learning (MIL) framework for WSI classification and captioning. PathM3 adapts a query-based transformer to effectively align WSIs with diagnostic captions. Given that histopathology visual patterns are redundantly distributed across WSIs, we aggregate each patch feature with MIL method that considers t
    
[^3]: 在社交媒体平台上对多模态大型语言模型进行基准测试

    MM-Soc: Benchmarking Multimodal Large Language Models in Social Media Platforms

    [https://arxiv.org/abs/2402.14154](https://arxiv.org/abs/2402.14154)

    该研究介绍了MM-Soc，一个旨在评估多模态大型语言模型（MLLMs）对社交媒体内容理解的综合基准，通过对十种大小变体的四个开源MLLMs进行详尽评估，发现了显著的性能差异。

    

    社交媒体平台是多模态信息交流的中心，包括文本、图片和视频，这使得机器难以理解在线空间中交互所关联的信息或情绪。多模态大型语言模型（MLLMs）已经成为解决这些挑战的一个有前途的解决方案，但是它们在准确解释人类情绪和诸如虚假信息等复杂内容方面存在困难。本文介绍了MM-Soc，一个旨在评估MLLMs对多模态社交媒体内容理解的综合基准。MM-Soc整合了著名的多模态数据集，并融入了一个新颖的大规模YouTube标记数据集，旨在针对从虚假信息检测、仇恨言论检测到社交上下文生成等一系列任务。通过对四个开源MLLMs的十种不同规模变体进行详尽评估，我们发现了显著的性能差异，凸显出了对性能平衡的需求。

    arXiv:2402.14154v1 Announce Type: new  Abstract: Social media platforms are hubs for multimodal information exchange, encompassing text, images, and videos, making it challenging for machines to comprehend the information or emotions associated with interactions in online spaces. Multimodal Large Language Models (MLLMs) have emerged as a promising solution to address these challenges, yet struggle with accurately interpreting human emotions and complex contents like misinformation. This paper introduces MM-Soc, a comprehensive benchmark designed to evaluate MLLMs' understanding of multimodal social media content. MM-Soc compiles prominent multimodal datasets and incorporates a novel large-scale YouTube tagging dataset, targeting a range of tasks from misinformation detection, hate speech detection, and social context generation. Through our exhaustive evaluation on ten size-variants of four open-source MLLMs, we have identified significant performance disparities, highlighting the need
    
[^4]: 生成型人工智能中的版权保护：技术视角

    Copyright Protection in Generative AI: A Technical Perspective

    [https://arxiv.org/abs/2402.02333](https://arxiv.org/abs/2402.02333)

    本文从技术角度全面概述了在生成型人工智能中的版权保护问题，包括数据版权和模型版权两个方面，并提出了一些创新的方法和技术。

    

    近年来，生成型人工智能（Generative AI）取得了快速发展，扩展了其创建文本、图像、音频和代码等合成内容的能力。这些深度生成模型（Deep Generative Models，DGMs）生成的内容高保真度和真实性引发了重大的版权问题。关于如何有效保护DGMs中的版权问题，已经进行了各种法律辩论。本文从技术角度提供了版权保护的全面概述。我们从两个不同的视角来进行研究：一是与数据所有者所持有的源数据相关的版权，二是与模型构建者所维护的生成模型相关的版权。对于数据版权，我们深入探讨了数据所有者如何保护其内容，并在不侵犯这些权利的情况下使用DGMs。对于模型版权，我们的讨论延伸到防止模型盗窃和识别特定模型生成的输出的策略。最后，我们强调了一些创新的方法和技术来处理这些版权问题。

    Generative AI has witnessed rapid advancement in recent years, expanding their capabilities to create synthesized content such as text, images, audio, and code. The high fidelity and authenticity of contents generated by these Deep Generative Models (DGMs) have sparked significant copyright concerns. There have been various legal debates on how to effectively safeguard copyrights in DGMs. This work delves into this issue by providing a comprehensive overview of copyright protection from a technical perspective. We examine from two distinct viewpoints: the copyrights pertaining to the source data held by the data owners and those of the generative models maintained by the model builders. For data copyright, we delve into methods data owners can protect their content and DGMs can be utilized without infringing upon these rights. For model copyright, our discussion extends to strategies for preventing model theft and identifying outputs generated by specific models. Finally, we highlight 
    
[^5]: 大型语言模型在视频理解中的应用：一项调查研究

    Video Understanding with Large Language Models: A Survey. (arXiv:2312.17432v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2312.17432](http://arxiv.org/abs/2312.17432)

    这项调查研究提供了对大型语言模型（Vid-LLMs）在视频理解中的最新进展的详细概述。Vid-LLMs的新兴能力包括开放式时空推理和常识知识，为未来的视频理解提供了有前途的方向。

    

    随着在线视频平台的不断增长和视频内容的不断增多，对熟练的视频理解工具的需求显著增加。鉴于大型语言模型在语言和多模态任务中的卓越能力，本调查提供了对利用大型语言模型（Vid-LLMs）技术进行视频理解的最新进展的详细概述。Vid-LLMs的新兴能力令人惊讶，尤其是它们在开放式时空推理和常识知识方面的能力，为未来的视频理解提供了一个有前途的方向。本调查对Vid-LLMs的独特特点和能力进行了分类，分为四种主要类型：基于LLM的视频代理、Vid-LLMs的预训练、Vid-LLMs的指令调整和混合方法。此外，本调查对Vid-LLMs的任务、数据集和评估方法进行了全面的研究。另外，它还探讨了Vid-LLMs技术的局限性和未来的挑战。

    With the burgeoning growth of online video platforms and the escalating volume of video content, the demand for proficient video understanding tools has intensified markedly. Given the remarkable capabilities of Large Language Models (LLMs) in language and multimodal tasks, this survey provides a detailed overview of the recent advancements in video understanding harnessing the power of LLMs (Vid-LLMs). The emergent capabilities of Vid-LLMs are surprisingly advanced, particularly their ability for open-ended spatial-temporal reasoning combined with commonsense knowledge, suggesting a promising path for future video understanding. We examine the unique characteristics and capabilities of Vid-LLMs, categorizing the approaches into four main types: LLM-based Video Agents, Vid-LLMs Pretraining, Vid-LLMs Instruction Tuning, and Hybrid Methods. Furthermore, this survey presents a comprehensive study of the tasks, datasets, and evaluation methodologies for Vid-LLMs. Additionally, it explores 
    
[^6]: EXACT: 如何提高准确率的训练方法。

    EXACT: How to Train Your Accuracy. (arXiv:2205.09615v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2205.09615](http://arxiv.org/abs/2205.09615)

    本文提出了一种新的分类任务优化框架，通过引入随机性，优化期望准确率，取得了强有力的替代分类损失的结果。

    

    分类任务通常会以准确率作为评估标准。然而，准确率是不连续的，无法直接使用梯度上升进行优化。流行的方法是通过最小化交叉熵、铰链损失或其他替代损失来优化，但这可能导致次优结果。本文提出了一种新的优化框架，通过向模型的输出引入随机性并优化期望准确率，即随机模型的准确率。对线性模型和深度图像分类进行了广泛的实验，结果表明所提出的优化方法是广泛使用的分类损失的强有力替代方案。

    Classification tasks are usually evaluated in terms of accuracy. However, accuracy is discontinuous and cannot be directly optimized using gradient ascent. Popular methods minimize cross-entropy, hinge loss, or other surrogate losses, which can lead to suboptimal results. In this paper, we propose a new optimization framework by introducing stochasticity to a model's output and optimizing expected accuracy, i.e. accuracy of the stochastic model. Extensive experiments on linear models and deep image classification show that the proposed optimization method is a powerful alternative to widely used classification losses.
    

