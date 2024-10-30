# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Benchmarking Counterfactual Image Generation](https://arxiv.org/abs/2403.20287) | 提出了一个针对对照图像生成方法的基准测试框架，包含评估对照多个方面的度量标准以及评估三种不同类型的条件图像生成模型性能。 |
| [^2] | [Refining Text-to-Image Generation: Towards Accurate Training-Free Glyph-Enhanced Image Generation](https://arxiv.org/abs/2403.16422) | 通过引入LenCom-Eval基准测试，研究者发现基于扩散模型的文本到图像生成方法仍面临三个主要挑战，并为未来研究提供了一个测试平台。 |
| [^3] | [Simplified Diffusion Schr\"odinger Bridge](https://arxiv.org/abs/2403.14623) | 介绍了简化后的扩散薛定谔桥（DSB），通过与基于得分的生成模型（SGM）的统一解决了复杂数据生成中的限制，提高了性能并加快了收敛速度。 |
| [^4] | [KEBench: A Benchmark on Knowledge Editing for Large Vision-Language Models](https://arxiv.org/abs/2403.07350) | KEBench提出了一个新的基准测试，采用不同的数据收集方法和新增加的度量标准（可移植性），以全面评估大型视觉-语言模型知识编辑的质量。 |
| [^5] | [A Probabilistic Hadamard U-Net for MRI Bias Field Correction](https://arxiv.org/abs/2403.05024) | 提出了一种用于前列腺MRI偏置场校正的概率哈达玛U-Net，引入了哈达玛U-Net（HU-Net）通过哈达玛变换将输入图像从时域转换为频域，并使用可训练的滤波器和硬阈值层消除高频部分。 |
| [^6] | [D\'ej\`a Vu Memorization in Vision-Language Models](https://arxiv.org/abs/2402.02103) | 这项研究提出了一种新方法来衡量视觉语言模型中的记忆现象，并发现对于使用图像-标题对进行训练的VLMs，模型确实会保留关于训练图像中的个别对象的信息，文本随机化可以在很大程度上减轻记忆现象而对模型的下游任务性能影响较小。 |
| [^7] | [A Critical Look at Classic Test-Time Adaptation Methods in Semantic Segmentation.](http://arxiv.org/abs/2310.05341) | 这项研究对语义分割中的经典测试时适应方法进行了批判性探究，揭示了分割TTA所面临的独特挑战，并发现经典TTA策略在这一任务中并不有效。 |
| [^8] | [What Makes ImageNet Look Unlike LAION.](http://arxiv.org/abs/2306.15769) | 本研究通过重新搜索大规模的LAION数据集，尝试重新创建图像网，并发现与原始图像网相比，新建的LAIONet具有明显不同之处。这种差异的原因是，在基于图像描述进行搜索时，存在信息瓶颈，从而减轻了选择偏差。 |
| [^9] | [RHINO: Rotated DETR with Dynamic Denoising via Hungarian Matching for Oriented Object Detection.](http://arxiv.org/abs/2305.07598) | 本文提出了一种面向定向目标检测的DINO基线模型RHINO。并通过匈牙利匹配和查询对齐的方式实现动态降噪，解决了重复预测的问题，从而在公共基准测试中达到最先进的性能水平。 |
| [^10] | [GPT-NAS: Neural Architecture Search with the Generative Pre-Trained Model.](http://arxiv.org/abs/2305.05351) | GPT-NAS使用生成式预训练模型优化神经架构搜索，通过提出近似的架构组件减小搜索空间，并明显优于其他NAS方法。 |

# 详细

[^1]: 基准对照图像生成

    Benchmarking Counterfactual Image Generation

    [https://arxiv.org/abs/2403.20287](https://arxiv.org/abs/2403.20287)

    提出了一个针对对照图像生成方法的基准测试框架，包含评估对照多个方面的度量标准以及评估三种不同类型的条件图像生成模型性能。

    

    对照图像生成在理解变量因果关系方面具有关键作用，在解释性和生成无偏合成数据方面有应用。然而，评估图像生成本身就是一个长期存在的挑战。对于评估对照生成的需求进一步加剧了这一挑战，因为根据定义，对照情景是没有可观测基准事实的假设情况。本文提出了一个旨在对照图像生成方法进行基准测试的新颖综合框架。我们结合了侧重于评估对照的不同方面的度量标准，例如组成、有效性、干预的最小性和图像逼真度。我们评估了基于结构因果模型范式的三种不同条件图像生成模型类型的性能。我们的工作还配备了一个用户友好的Python软件包，可以进一步评估。

    arXiv:2403.20287v1 Announce Type: cross  Abstract: Counterfactual image generation is pivotal for understanding the causal relations of variables, with applications in interpretability and generation of unbiased synthetic data. However, evaluating image generation is a long-standing challenge in itself. The need to evaluate counterfactual generation compounds on this challenge, precisely because counterfactuals, by definition, are hypothetical scenarios without observable ground truths. In this paper, we present a novel comprehensive framework aimed at benchmarking counterfactual image generation methods. We incorporate metrics that focus on evaluating diverse aspects of counterfactuals, such as composition, effectiveness, minimality of interventions, and image realism. We assess the performance of three distinct conditional image generation model types, based on the Structural Causal Model paradigm. Our work is accompanied by a user-friendly Python package which allows to further eval
    
[^2]: 优化文本到图像生成：向准确的无需训练的字形增强图像生成迈进

    Refining Text-to-Image Generation: Towards Accurate Training-Free Glyph-Enhanced Image Generation

    [https://arxiv.org/abs/2403.16422](https://arxiv.org/abs/2403.16422)

    通过引入LenCom-Eval基准测试，研究者发现基于扩散模型的文本到图像生成方法仍面临三个主要挑战，并为未来研究提供了一个测试平台。

    

    过去几年，基于扩散模型的文本到图像（T2I）生成方法引起了广泛关注。然而，普通扩散模型通常在生成图像中显示的文本中存在拼写不准确的问题。生成视觉文本的能力至关重要，不仅具有学术价值，还有广泛的实际应用。为了生成准确的视觉文本图像，最先进的技术采用了一种字形控制的图像生成方法，包括文本布局生成器，然后是一个在生成的文本布局的条件下生成图像的图像生成器。然而，我们的研究发现这些模型仍然面临三个主要挑战，促使我们开发了一个测试平台来促进未来的研究。我们引入了一个名为LenCom-Eval的基准测试，专门用于测试模型在生成具有复杂视觉文本的图像方面的能力。

    arXiv:2403.16422v1 Announce Type: cross  Abstract: Over the past few years, Text-to-Image (T2I) generation approaches based on diffusion models have gained significant attention. However, vanilla diffusion models often suffer from spelling inaccuracies in the text displayed within the generated images. The capability to generate visual text is crucial, offering both academic interest and a wide range of practical applications. To produce accurate visual text images, state-of-the-art techniques adopt a glyph-controlled image generation approach, consisting of a text layout generator followed by an image generator that is conditioned on the generated text layout. Nevertheless, our study reveals that these models still face three primary challenges, prompting us to develop a testbed to facilitate future research. We introduce a benchmark, LenCom-Eval, specifically designed for testing models' capability in generating images with Lengthy and Complex visual text. Subsequently, we introduce 
    
[^3]: 简化扩散薛定谔桥

    Simplified Diffusion Schr\"odinger Bridge

    [https://arxiv.org/abs/2403.14623](https://arxiv.org/abs/2403.14623)

    介绍了简化后的扩散薛定谔桥（DSB），通过与基于得分的生成模型（SGM）的统一解决了复杂数据生成中的限制，提高了性能并加快了收敛速度。

    

    这篇论文介绍了一种新的理论简化扩散薛定谔桥（DSB），便于将其与基于得分的生成模型（SGM）统一起来，解决了DSB在复杂数据生成方面的局限性，实现更快的收敛速度和增强的性能。通过将SGM作为DSB的初始解决方案，我们的方法利用了这两个框架的优势，确保了更高效的训练过程，并改进了SGM的性能。我们还提出了一种重新参数化技术，尽管存在理论近似，但实际上提高了网络的拟合能力。我们进行了大量的实验证实，证实了简化的DSB的有效性，展示了其显著的改进。我们相信这项工作的贡献为先进的生成建模铺平了道路。

    arXiv:2403.14623v1 Announce Type: new  Abstract: This paper introduces a novel theoretical simplification of the Diffusion Schr\"odinger Bridge (DSB) that facilitates its unification with Score-based Generative Models (SGMs), addressing the limitations of DSB in complex data generation and enabling faster convergence and enhanced performance. By employing SGMs as an initial solution for DSB, our approach capitalizes on the strengths of both frameworks, ensuring a more efficient training process and improving the performance of SGM. We also propose a reparameterization technique that, despite theoretical approximations, practically improves the network's fitting capabilities. Our extensive experimental evaluations confirm the effectiveness of the simplified DSB, demonstrating its significant improvements. We believe the contributions of this work pave the way for advanced generative modeling. The code is available at https://github.com/tzco/Simplified-Diffusion-Schrodinger-Bridge.
    
[^4]: KEBench: 用于大型视觉-语言模型知识编辑的基准测试

    KEBench: A Benchmark on Knowledge Editing for Large Vision-Language Models

    [https://arxiv.org/abs/2403.07350](https://arxiv.org/abs/2403.07350)

    KEBench提出了一个新的基准测试，采用不同的数据收集方法和新增加的度量标准（可移植性），以全面评估大型视觉-语言模型知识编辑的质量。

    

    arXiv:2403.07350v1 公告类型: 跨领域 摘要: 目前，针对大型视觉-语言模型(LVLMs)的知识编辑研究很少。编辑LVLMs面临着有效整合多种模态（图像和文本）的挑战，同时确保修改连贯且与上下文相关。现有基准测试具有三个度量标准（可靠性、局部性和一般性）用于衡量LVLMs的知识编辑。然而，该基准测试在评估中使用的生成图像质量不足，并且无法评估模型是否有效地利用与相关内容相关的编辑知识。我们采用不同的数据收集方法构建了一个新的基准测试$\textbf{KEBench}$，并扩展了新度量标准(可移植性)以进行全面评估。借助多模态知识图，我们的图像数据呈现出明确的给实体方向性。这种方向性可以进一步用于提取与实体相关的知识和进行编辑。

    arXiv:2403.07350v1 Announce Type: cross  Abstract: Currently, little research has been done on knowledge editing for Large Vision-Language Models (LVLMs). Editing LVLMs faces the challenge of effectively integrating diverse modalities (image and text) while ensuring coherent and contextually relevant modifications. An existing benchmark has three metrics (Reliability, Locality and Generality) to measure knowledge editing for LVLMs. However, the benchmark falls short in the quality of generated images used in evaluation and cannot assess whether models effectively utilize edited knowledge in relation to the associated content. We adopt different data collection methods to construct a new benchmark, $\textbf{KEBench}$, and extend new metric (Portability) for a comprehensive evaluation. Leveraging a multimodal knowledge graph, our image data exhibits clear directionality towards entities. This directional aspect can be further utilized to extract entity-related knowledge and form editing 
    
[^5]: 一种用于MRI偏置场校正的概率哈达玛U-Net

    A Probabilistic Hadamard U-Net for MRI Bias Field Correction

    [https://arxiv.org/abs/2403.05024](https://arxiv.org/abs/2403.05024)

    提出了一种用于前列腺MRI偏置场校正的概率哈达玛U-Net，引入了哈达玛U-Net（HU-Net）通过哈达玛变换将输入图像从时域转换为频域，并使用可训练的滤波器和硬阈值层消除高频部分。

    

    磁场不均匀性校正在MRI分析中仍然是一个具有挑战性的任务。大多数已建立的技术是为脑MRI设计的，假设相同组织中的图像强度遵循均匀分布。这种假设不易适用于其他器官，特别是那些体积小，质地不均匀（强度变化大）的器官，比如前列腺。为了解决这个问题，本文提出了一种用于前列腺MRI偏置场校正的概率哈达玛U-Net（PHU-Net）。首先，引入了一种新颖的哈达玛U-Net（HU-Net）以提取低频标量场，将其乘以原始输入以获得原型校正图像。HU-Net通过哈达玛变换将输入图像从时域转换为频域。在频域中，使用可训练的滤波器（缩放层）、硬阈值层消除高频部分。

    arXiv:2403.05024v1 Announce Type: cross  Abstract: Magnetic field inhomogeneity correction remains a challenging task in MRI analysis. Most established techniques are designed for brain MRI by supposing that image intensities in the identical tissue follow a uniform distribution. Such an assumption cannot be easily applied to other organs, especially those that are small in size and heterogeneous in texture (large variations in intensity), such as the prostate. To address this problem, this paper proposes a probabilistic Hadamard U-Net (PHU-Net) for prostate MRI bias field correction. First, a novel Hadamard U-Net (HU-Net) is introduced to extract the low-frequency scalar field, multiplied by the original input to obtain the prototypical corrected image. HU-Net converts the input image from the time domain into the frequency domain via Hadamard transform. In the frequency domain, high-frequency components are eliminated using the trainable filter (scaling layer), hard-thresholding laye
    
[^6]: 视觉语言模型中的心理现象记忆

    D\'ej\`a Vu Memorization in Vision-Language Models

    [https://arxiv.org/abs/2402.02103](https://arxiv.org/abs/2402.02103)

    这项研究提出了一种新方法来衡量视觉语言模型中的记忆现象，并发现对于使用图像-标题对进行训练的VLMs，模型确实会保留关于训练图像中的个别对象的信息，文本随机化可以在很大程度上减轻记忆现象而对模型的下游任务性能影响较小。

    

    视觉语言模型（VLM）作为最先进的表示学习解决方案出现，具有诸多下游应用，如图像分类、检索和生成。一个自然的问题是这些模型是否会记忆训练数据，这也对泛化有着影响。我们提出了一种衡量VLMs中记忆的新方法，称之为心理现象记忆。对于在图像-标题对上训练的VLMs，我们展示了该模型确实保留了关于训练图像中个别对象的信息，超出了从相关性或图像标题中可以推断出的范畴。我们在样本和总体水平上评估了心理现象记忆，并展示了OpenCLIP在多达5000万个图像-标题对上训练时的显著性。最后，我们展示了文本随机化在很大程度上减轻了记忆，同时对模型的下游任务性能产生了适度影响。

    Vision-Language Models (VLMs) have emerged as the state-of-the-art representation learning solution, with myriads of downstream applications such as image classification, retrieval and generation. A natural question is whether these models memorize their training data, which also has implications for generalization. We propose a new method for measuring memorization in VLMs, which we call d\'ej\`a vu memorization. For VLMs trained on image-caption pairs, we show that the model indeed retains information about individual objects in the training images beyond what can be inferred from correlations or the image caption. We evaluate d\'ej\`a vu memorization at both sample and population level, and show that it is significant for OpenCLIP trained on as many as 50M image-caption pairs. Finally, we show that text randomization considerably mitigates memorization while only moderately impacting the model's downstream task performance.
    
[^7]: 对语义分割中经典的测试时适应方法的批判性探究

    A Critical Look at Classic Test-Time Adaptation Methods in Semantic Segmentation. (arXiv:2310.05341v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2310.05341](http://arxiv.org/abs/2310.05341)

    这项研究对语义分割中的经典测试时适应方法进行了批判性探究，揭示了分割TTA所面临的独特挑战，并发现经典TTA策略在这一任务中并不有效。

    

    测试时适应（TTA）旨在将最初在训练数据上训练的模型适应于测试数据中的可能分布变化。然而，大多数现有的TTA研究都集中在分类任务上，对于语义分割的TTA探索非常有限。这种对分类的突出重视可能导致许多新手和工程师错误地认为为分类设计的经典TTA方法可以直接应用于分割任务。然而，这一假设仍未经验证，是一个待解决的问题。为了解决这个问题，我们进行了一项系统的实证研究，揭示了分割TTA的独特挑战，并确定经典TTA策略是否可以有效应对这一任务。我们全面的结果得出了三个关键观察结果。首先，常用于分类TTA的经典批归一化更新策略只能带来轻微的性能改善，在某些情况下甚至会对结果产生逆向影响。

    Test-time adaptation (TTA) aims to adapt a model, initially trained on training data, to potential distribution shifts in the test data. Most existing TTA studies, however, focus on classification tasks, leaving a notable gap in the exploration of TTA for semantic segmentation. This pronounced emphasis on classification might lead numerous newcomers and engineers to mistakenly assume that classic TTA methods designed for classification can be directly applied to segmentation. Nonetheless, this assumption remains unverified, posing an open question. To address this, we conduct a systematic, empirical study to disclose the unique challenges of segmentation TTA, and to determine whether classic TTA strategies can effectively address this task. Our comprehensive results have led to three key observations. First, the classic batch norm updating strategy, commonly used in classification TTA, only brings slight performance improvement, and in some cases it might even adversely affect the resu
    
[^8]: 图像网为何与LAION网络截然不同

    What Makes ImageNet Look Unlike LAION. (arXiv:2306.15769v1 [cs.LG])

    [http://arxiv.org/abs/2306.15769](http://arxiv.org/abs/2306.15769)

    本研究通过重新搜索大规模的LAION数据集，尝试重新创建图像网，并发现与原始图像网相比，新建的LAIONet具有明显不同之处。这种差异的原因是，在基于图像描述进行搜索时，存在信息瓶颈，从而减轻了选择偏差。

    

    图像网是通过Flickr图像搜索结果创建的。如果我们仅根据图像描述重新创建图像网，搜索大规模的LAION数据集会发生什么呢？本研究进行了这个反事实的调查。我们发现重新创建的图像网，我们称之为LAIONet，与原始图像网有明显不同之处。具体而言，原始图像网中图像的类内相似性远高于LAIONet。因此，在图像网上训练的模型在LAIONet上表现明显较差。我们提出了一个严格解释这种差异的观点，并通过系统性的实验予以支持。简而言之，仅基于图像描述进行搜索会产生信息瓶颈，从而减轻了基于图像过滤时存在的选择偏差。我们的解释形式化了一个长期的直觉。

    ImageNet was famously created from Flickr image search results. What if we recreated ImageNet instead by searching the massive LAION dataset based on image captions alone? In this work, we carry out this counterfactual investigation. We find that the resulting ImageNet recreation, which we call LAIONet, looks distinctly unlike the original. Specifically, the intra-class similarity of images in the original ImageNet is dramatically higher than it is for LAIONet. Consequently, models trained on ImageNet perform significantly worse on LAIONet. We propose a rigorous explanation for the discrepancy in terms of a subtle, yet important, difference in two plausible causal data-generating processes for the respective datasets, that we support with systematic experimentation. In a nutshell, searching based on an image caption alone creates an information bottleneck that mitigates the selection bias otherwise present in image-based filtering. Our explanation formalizes a long-held intuition in th
    
[^9]: RHINO：通过匈牙利匹配实现动态降噪的旋转目标检测的旋转DETR

    RHINO: Rotated DETR with Dynamic Denoising via Hungarian Matching for Oriented Object Detection. (arXiv:2305.07598v1 [cs.CV])

    [http://arxiv.org/abs/2305.07598](http://arxiv.org/abs/2305.07598)

    本文提出了一种面向定向目标检测的DINO基线模型RHINO。并通过匈牙利匹配和查询对齐的方式实现动态降噪，解决了重复预测的问题，从而在公共基准测试中达到最先进的性能水平。

    

    随着DINO的发布，一种DETR的变体，检测变压器正在通过其端到端设计和可扩展性在目标检测基准中刷新记录。然而，虽然预计从其端到端架构中获得更多的好处，如消除NMS和与锚相关的成本，但尚未彻底研究DETR在定向目标检测方面的扩展。本文提出了首个面向定向目标检测的DINO基线。我们发现，直接使用DETR进行定向目标检测并不能保证不重复预测，并提出了一种简单的成本来减轻这种情况。此外，我们介绍了一种新的去噪策略，该策略使用匈牙利匹配来过滤冗余的带噪声的查询，并使用查询对齐来保持Transformer解码器层之间的匹配一致性。我们提出的模型在公共基准测试中优于以前的旋转DETR和其他对手，实现了最先进的性能。

    With the publication of DINO, a variant of the Detection Transformer (DETR), Detection Transformers are breaking the record in the object detection benchmark with the merits of their end-to-end design and scalability. However, the extension of DETR to oriented object detection has not been thoroughly studied although more benefits from its end-to-end architecture are expected such as removing NMS and anchor-related costs. In this paper, we propose a first strong DINO-based baseline for oriented object detection. We found that straightforward employment of DETRs for oriented object detection does not guarantee non-duplicate prediction, and propose a simple cost to mitigate this. Furthermore, we introduce a novel denoising strategy that uses Hungarian matching to filter redundant noised queries and query alignment to preserve matching consistency between Transformer decoder layers. Our proposed model outperforms previous rotated DETRs and other counterparts, achieving state-of-the-art pe
    
[^10]: GPT-NAS: 以生成式预训练模型为基础的神经架构搜索

    GPT-NAS: Neural Architecture Search with the Generative Pre-Trained Model. (arXiv:2305.05351v1 [cs.CV] CROSS LISTED)

    [http://arxiv.org/abs/2305.05351](http://arxiv.org/abs/2305.05351)

    GPT-NAS使用生成式预训练模型优化神经架构搜索，通过提出近似的架构组件减小搜索空间，并明显优于其他NAS方法。

    

    神经架构搜索(NAS)已经成为了一种自动设计最优神经网络架构的有效方法之一。虽然一些人工设计的神经网络已经在多项任务中取得了人类水平的表现，但在NAS方法中很少出现这类成果，主要原因在于神经架构的搜索空间太大了，导致NAS算法效率低下。这项工作提出了一种新的架构搜索算法，称为GPT-NAS，通过生成式预训练模型来优化神经架构。在GPT-NAS中，我们假设一个在大规模语料库上预训练的生成模型能够学习构建神经架构的基本规律。因此，GPT-NAS利用生成式预训练模型来提出合理的架构组件，从而大大减少了搜索空间，引入了搜索过程中的先验知识。广泛的实验结果表明，我们的GPT-NAS方法明显优于其他NAS方法。

    Neural Architecture Search (NAS) has emerged as one of the effective methods to design the optimal neural network architecture automatically. Although neural architectures have achieved human-level performances in several tasks, few of them are obtained from the NAS method. The main reason is the huge search space of neural architectures, making NAS algorithms inefficient. This work presents a novel architecture search algorithm, called GPT-NAS, that optimizes neural architectures by Generative Pre-Trained (GPT) model. In GPT-NAS, we assume that a generative model pre-trained on a large-scale corpus could learn the fundamental law of building neural architectures. Therefore, GPT-NAS leverages the generative pre-trained (GPT) model to propose reasonable architecture components given the basic one. Such an approach can largely reduce the search space by introducing prior knowledge in the search process. Extensive experimental results show that our GPT-NAS method significantly outperforms
    

