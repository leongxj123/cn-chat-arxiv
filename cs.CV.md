# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DeNetDM: Debiasing by Network Depth Modulation](https://arxiv.org/abs/2403.19863) | DeNetDM 是一种基于网络深度调制的新型去偏见方法，通过使用来自专家乘积的训练范式，在创建深浅架构的偏见和去偏见分支后，将知识提炼产生目标去偏见模型，相比当前去偏见技术取得更优异的效果。 |
| [^2] | [CMP: Cooperative Motion Prediction with Multi-Agent Communication](https://arxiv.org/abs/2403.17916) | 该论文提出了一种名为CMP的方法，利用LiDAR信号作为输入，通过合作感知和运动预测模块共享信息，解决了合作运动预测的问题。 |
| [^3] | [LSTP: Language-guided Spatial-Temporal Prompt Learning for Long-form Video-Text Understanding](https://arxiv.org/abs/2402.16050) | LSTP提出了语言引导的时空提示学习方法，通过整合时间提示采样器（TPS）和空间提示求解器（SPS）以及一致的训练策略，显著提升了计算效率、时间理解和空间-时间对齐。 |
| [^4] | [Large-Scale Actionless Video Pre-Training via Discrete Diffusion for Efficient Policy Learning](https://arxiv.org/abs/2402.14407) | 利用离散扩散结合生成式预训练和少量机器人视频微调，实现从人类视频到机器人策略学习的知识迁移。 |
| [^5] | [The Instinctive Bias: Spurious Images lead to Hallucination in MLLMs](https://arxiv.org/abs/2402.03757) | 本论文研究发现，虚假图像会导致多模态大型语言模型产生幻觉，作者提出了评估幻觉程度的基准CorrelationQA，并发现主流多模态大型语言模型普遍受到这种本能偏见的影响。 |
| [^6] | [Generalizing Medical Image Representations via Quaternion Wavelet Networks.](http://arxiv.org/abs/2310.10224) | 本文提出了一种名为QUAVE的四元数小波网络，可以从医学图像中提取显著特征。该网络可以与现有的医学图像分析或综合任务结合使用，并推广了对单通道数据的采用。通过四元数小波变换和加权处理，QUAVE能够处理具有较大变化的医学数据。 |
| [^7] | [JL-lemma derived Optimal Projections for Discriminative Dictionary Learning.](http://arxiv.org/abs/2308.13991) | 本文提出了一种名为JLSPCADL的新方法，通过利用Johnson-Lindenstrauss引理选择一个维数的转换空间，从而实现判别式字典学习并提供更好的分类结果。 |
| [^8] | [Towards Data-and Knowledge-Driven Artificial Intelligence: A Survey on Neuro-Symbolic Computing.](http://arxiv.org/abs/2210.15889) | 神经符号计算是将符号和统计认知范式整合的研究领域，在推理和解释性以及神经网络学习等方面具有潜力。本文综述了该领域的最新进展和重要贡献，并探讨了其成功应用案例和未来发展方向。 |

# 详细

[^1]: DeNetDM: 通过网络深度调制来消除偏见

    DeNetDM: Debiasing by Network Depth Modulation

    [https://arxiv.org/abs/2403.19863](https://arxiv.org/abs/2403.19863)

    DeNetDM 是一种基于网络深度调制的新型去偏见方法，通过使用来自专家乘积的训练范式，在创建深浅架构的偏见和去偏见分支后，将知识提炼产生目标去偏见模型，相比当前去偏见技术取得更优异的效果。

    

    当神经网络在偏见数据集上训练时，它们往往会无意间学习到虚假的相关性，从而导致在实现强大的泛化性和鲁棒性方面面临挑战。目前解决这种偏见的方法通常包括利用偏见注释、根据伪偏见标签进行加权重、或通过增强技术增加偏见冲突数据点的多样性。我们引入了DeNetDM，这是一种基于观察结果的新型去偏见方法，浅层神经网络优先学习核心属性，而更深层次的神经网络在获取不同信息时强调偏见。我们利用从专家乘积中推导出的训练范式，创建了深浅架构的偏见和去偏见分支，然后用知识提炼产生目标的去偏见模型。大量实验证明，我们的方法优于当前的去偏见技术，实现了一个...

    arXiv:2403.19863v1 Announce Type: new  Abstract: When neural networks are trained on biased datasets, they tend to inadvertently learn spurious correlations, leading to challenges in achieving strong generalization and robustness. Current approaches to address such biases typically involve utilizing bias annotations, reweighting based on pseudo-bias labels, or enhancing diversity within bias-conflicting data points through augmentation techniques. We introduce DeNetDM, a novel debiasing method based on the observation that shallow neural networks prioritize learning core attributes, while deeper ones emphasize biases when tasked with acquiring distinct information. Using a training paradigm derived from Product of Experts, we create both biased and debiased branches with deep and shallow architectures and then distill knowledge to produce the target debiased model. Extensive experiments and analyses demonstrate that our approach outperforms current debiasing techniques, achieving a not
    
[^2]: CMP：具有多智能体通信的合作运动预测

    CMP: Cooperative Motion Prediction with Multi-Agent Communication

    [https://arxiv.org/abs/2403.17916](https://arxiv.org/abs/2403.17916)

    该论文提出了一种名为CMP的方法，利用LiDAR信号作为输入，通过合作感知和运动预测模块共享信息，解决了合作运动预测的问题。

    

    随着自动驾驶车辆（AVs）的发展和车联网（V2X）通信的成熟，合作连接的自动化车辆（CAVs）的功能变得可能。本文基于合作感知，探讨了合作运动预测的可行性和有效性。我们的方法CMP以LiDAR信号作为输入，以增强跟踪和预测能力。与过去专注于合作感知或运动预测的工作不同，我们的框架是我们所知的第一个解决CAVs在感知和预测模块中共享信息的统一问题。我们的设计中还融入了能够容忍现实V2X带宽限制和传输延迟的独特能力，同时处理庞大的感知表示。我们还提出了预测聚合模块，统一了预测

    arXiv:2403.17916v1 Announce Type: cross  Abstract: The confluence of the advancement of Autonomous Vehicles (AVs) and the maturity of Vehicle-to-Everything (V2X) communication has enabled the capability of cooperative connected and automated vehicles (CAVs). Building on top of cooperative perception, this paper explores the feasibility and effectiveness of cooperative motion prediction. Our method, CMP, takes LiDAR signals as input to enhance tracking and prediction capabilities. Unlike previous work that focuses separately on either cooperative perception or motion prediction, our framework, to the best of our knowledge, is the first to address the unified problem where CAVs share information in both perception and prediction modules. Incorporated into our design is the unique capability to tolerate realistic V2X bandwidth limitations and transmission delays, while dealing with bulky perception representations. We also propose a prediction aggregation module, which unifies the predict
    
[^3]: LSTP: 语言引导的时空提示学习用于长篇视频文本理解

    LSTP: Language-guided Spatial-Temporal Prompt Learning for Long-form Video-Text Understanding

    [https://arxiv.org/abs/2402.16050](https://arxiv.org/abs/2402.16050)

    LSTP提出了语言引导的时空提示学习方法，通过整合时间提示采样器（TPS）和空间提示求解器（SPS）以及一致的训练策略，显著提升了计算效率、时间理解和空间-时间对齐。

    

    尽管视频语言建模取得了进展，但在回应特定任务的语言查询时解释长篇视频的计算挑战仍然存在，这主要是由于高维视频数据的复杂性和语言与空间和时间上视觉线索之间的不一致性。为解决这一问题，我们引入了一种名为语言引导的时空提示学习（LSTP）的新方法。该方法具有两个关键组件：利用光流先验的时间提示采样器（TPS），可利用时间信息有效提取相关视频内容；以及灵巧地捕捉视觉和文本元素之间复杂空间关系的空间提示求解器（SPS）。通过将TPS和SPS与一致的训练策略相协调，我们的框架显著提升了计算效率、时间理解和空间-时间对齐。在两个挑战中的实证评估显示，我们的方法优于现有技术。

    arXiv:2402.16050v1 Announce Type: cross  Abstract: Despite progress in video-language modeling, the computational challenge of interpreting long-form videos in response to task-specific linguistic queries persists, largely due to the complexity of high-dimensional video data and the misalignment between language and visual cues over space and time. To tackle this issue, we introduce a novel approach called Language-guided Spatial-Temporal Prompt Learning (LSTP). This approach features two key components: a Temporal Prompt Sampler (TPS) with optical flow prior that leverages temporal information to efficiently extract relevant video content, and a Spatial Prompt Solver (SPS) that adeptly captures the intricate spatial relationships between visual and textual elements. By harmonizing TPS and SPS with a cohesive training strategy, our framework significantly enhances computational efficiency, temporal understanding, and spatial-temporal alignment. Empirical evaluations across two challeng
    
[^4]: 通过离散扩散进行大规模无动作视频预训练，以实现高效策略学习

    Large-Scale Actionless Video Pre-Training via Discrete Diffusion for Efficient Policy Learning

    [https://arxiv.org/abs/2402.14407](https://arxiv.org/abs/2402.14407)

    利用离散扩散结合生成式预训练和少量机器人视频微调，实现从人类视频到机器人策略学习的知识迁移。

    

    学习一个能够完成多个任务的通用实体代理面临挑战，主要源自缺乏有标记动作的机器人数据集。相比之下，存在大量捕捉复杂任务和与物理世界互动的人类视频。本文介绍了一种新颖框架，利用统一的离散扩散将人类视频上的生成式预训练与少量有标记机器人视频上的策略微调结合起来。我们首先将人类和机器人视频压缩成统一的视频标记。在预训练阶段，我们使用一个带有蒙版替换扩散策略的离散扩散模型来预测潜空间中的未来视频标记。在微调阶段，我们 h

    arXiv:2402.14407v1 Announce Type: new  Abstract: Learning a generalist embodied agent capable of completing multiple tasks poses challenges, primarily stemming from the scarcity of action-labeled robotic datasets. In contrast, a vast amount of human videos exist, capturing intricate tasks and interactions with the physical world. Promising prospects arise for utilizing actionless human videos for pre-training and transferring the knowledge to facilitate robot policy learning through limited robot demonstrations. In this paper, we introduce a novel framework that leverages a unified discrete diffusion to combine generative pre-training on human videos and policy fine-tuning on a small number of action-labeled robot videos. We start by compressing both human and robot videos into unified video tokens. In the pre-training stage, we employ a discrete diffusion model with a mask-and-replace diffusion strategy to predict future video tokens in the latent space. In the fine-tuning stage, we h
    
[^5]: 本能偏见：虚假图像导致MLLMs产生幻觉

    The Instinctive Bias: Spurious Images lead to Hallucination in MLLMs

    [https://arxiv.org/abs/2402.03757](https://arxiv.org/abs/2402.03757)

    本论文研究发现，虚假图像会导致多模态大型语言模型产生幻觉，作者提出了评估幻觉程度的基准CorrelationQA，并发现主流多模态大型语言模型普遍受到这种本能偏见的影响。

    

    大型语言模型（LLMs）近年来取得了显著进展，多模态大型语言模型（MLLMs）的出现使LLMs具备了视觉能力，在各种多模态任务中表现出色。然而，像GPT-4V这样强大的MLLMs在面对某些图像和文本输入时仍然以惊人的方式失败了。本文中，我们确定了一类典型输入，这些输入令MLLMs困惑，它们由高度相关但与答案不一致的图像组成，导致MLLMs产生幻觉。为了量化这种影响，我们提出了CorrelationQA，这是首个评估给定虚假图像的幻觉程度的基准。该基准包含13个类别的7,308个文本-图像对。基于提出的CorrelationQA，我们对9个主流MLLMs进行了深入分析，表明它们普遍受到这种本能偏见的不同程度的影响。我们希望我们精选的基准和评估结果能有所帮助。

    Large language models (LLMs) have recently experienced remarkable progress, where the advent of multi-modal large language models (MLLMs) has endowed LLMs with visual capabilities, leading to impressive performances in various multi-modal tasks. However, those powerful MLLMs such as GPT-4V still fail spectacularly when presented with certain image and text inputs. In this paper, we identify a typical class of inputs that baffles MLLMs, which consist of images that are highly relevant but inconsistent with answers, causing MLLMs to suffer from hallucination. To quantify the effect, we propose CorrelationQA, the first benchmark that assesses the hallucination level given spurious images. This benchmark contains 7,308 text-image pairs across 13 categories. Based on the proposed CorrelationQA, we conduct a thorough analysis on 9 mainstream MLLMs, illustrating that they universally suffer from this instinctive bias to varying degrees. We hope that our curated benchmark and evaluation result
    
[^6]: 通过四元数小波网络推广医学图像表示

    Generalizing Medical Image Representations via Quaternion Wavelet Networks. (arXiv:2310.10224v2 [eess.IV] UPDATED)

    [http://arxiv.org/abs/2310.10224](http://arxiv.org/abs/2310.10224)

    本文提出了一种名为QUAVE的四元数小波网络，可以从医学图像中提取显著特征。该网络可以与现有的医学图像分析或综合任务结合使用，并推广了对单通道数据的采用。通过四元数小波变换和加权处理，QUAVE能够处理具有较大变化的医学数据。

    

    鉴于来自不同来源和各种任务的数据集日益增加，神经网络的普适性成为一个广泛研究的领域。当处理医学数据时，这个问题尤为广泛，因为缺乏方法论标准导致不同的成像中心或使用不同设备和辅助因素获取的数据存在较大变化。为了克服这些限制，我们引入了一种新颖的、普适的、数据-和任务不可知的框架，能够从医学图像中提取显著特征。所提出的四元数小波网络（QUAVE）可以很容易地与任何现有的医学图像分析或综合任务相结合，并且可以结合实际、四元数或超复值模型，推广它们对单通道数据的采用。QUAVE首先通过四元数小波变换提取不同的子带，得到低频/近似频带和高频/细粒度特征。然后，它对最有代表性的特征进行加权处理，从而减少了特征重要性不均匀性。

    Neural network generalizability is becoming a broad research field due to the increasing availability of datasets from different sources and for various tasks. This issue is even wider when processing medical data, where a lack of methodological standards causes large variations being provided by different imaging centers or acquired with various devices and cofactors. To overcome these limitations, we introduce a novel, generalizable, data- and task-agnostic framework able to extract salient features from medical images. The proposed quaternion wavelet network (QUAVE) can be easily integrated with any pre-existing medical image analysis or synthesis task, and it can be involved with real, quaternion, or hypercomplex-valued models, generalizing their adoption to single-channel data. QUAVE first extracts different sub-bands through the quaternion wavelet transform, resulting in both low-frequency/approximation bands and high-frequency/fine-grained features. Then, it weighs the most repr
    
[^7]: JL-引理推导的用于判别式字典学习的最优投影

    JL-lemma derived Optimal Projections for Discriminative Dictionary Learning. (arXiv:2308.13991v1 [cs.CV])

    [http://arxiv.org/abs/2308.13991](http://arxiv.org/abs/2308.13991)

    本文提出了一种名为JLSPCADL的新方法，通过利用Johnson-Lindenstrauss引理选择一个维数的转换空间，从而实现判别式字典学习并提供更好的分类结果。

    

    为了克服在具有大量类别的大维度数据分类中的困难，我们提出了一种名为JLSPCADL的新方法。本文利用Johnson-Lindenstrauss(JL)引理，在一个转换空间中选择一个维数，可以在其中学习用于信号分类的判别式字典。与通常使用JL进行降维的随机投影不同，我们使用从Modified Supervised PC Analysis (M-SPCA)推导得出的投影转换矩阵，其维数遵循JL的规定。JLSPCADL提供了一种启发式方法来推断适当的失真水平和相应的字典原子的适当描述长度(SDL)，以推导出一个最优的特征空间，从而提供更好的分类的字典原子的SDL。与最先进的基于降维的字典学习方法不同，从M-SPCA中一步得出的投影转换矩阵提供了最大的特征-标签一致性。

    To overcome difficulties in classifying large dimensionality data with a large number of classes, we propose a novel approach called JLSPCADL. This paper uses the Johnson-Lindenstrauss (JL) Lemma to select the dimensionality of a transformed space in which a discriminative dictionary can be learned for signal classification. Rather than reducing dimensionality via random projections, as is often done with JL, we use a projection transformation matrix derived from Modified Supervised PC Analysis (M-SPCA) with the JL-prescribed dimension.  JLSPCADL provides a heuristic to deduce suitable distortion levels and the corresponding Suitable Description Length (SDL) of dictionary atoms to derive an optimal feature space and thus the SDL of dictionary atoms for better classification. Unlike state-of-the-art dimensionality reduction-based dictionary learning methods, a projection transformation matrix derived in a single step from M-SPCA provides maximum feature-label consistency of the transfor
    
[^8]: 走向数据和知识驱动的人工智能：神经符号计算综述

    Towards Data-and Knowledge-Driven Artificial Intelligence: A Survey on Neuro-Symbolic Computing. (arXiv:2210.15889v4 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2210.15889](http://arxiv.org/abs/2210.15889)

    神经符号计算是将符号和统计认知范式整合的研究领域，在推理和解释性以及神经网络学习等方面具有潜力。本文综述了该领域的最新进展和重要贡献，并探讨了其成功应用案例和未来发展方向。

    

    多年来，神经符号计算（NeSy）一直是人工智能（AI）领域的研究热点，追求符号和统计认知范式的整合。由于NeSy在符号表示的推理和可解释性以及神经网络的强大学习中具有潜力，它可能成为下一代AI的催化剂。本文系统综述了NeSy研究的最新进展和重要贡献。首先，我们介绍了该领域的研究历史，涵盖了早期工作和基础知识。我们进一步讨论了背景概念，并确定了推动NeSy发展的关键因素。随后，我们按照几个主要特征对近期的里程碑方法进行分类，包括神经符号整合、知识表示、知识嵌入和功能性。接下来，我们简要讨论了成功的应用案例，以及NeSy领域的挑战和未来发展方向。

    Neural-symbolic computing (NeSy), which pursues the integration of the symbolic and statistical paradigms of cognition, has been an active research area of Artificial Intelligence (AI) for many years. As NeSy shows promise of reconciling the advantages of reasoning and interpretability of symbolic representation and robust learning in neural networks, it may serve as a catalyst for the next generation of AI. In the present paper, we provide a systematic overview of the recent developments and important contributions of NeSy research. Firstly, we introduce study history of this area, covering early work and foundations. We further discuss background concepts and identify key driving factors behind the development of NeSy. Afterward, we categorize recent landmark approaches along several main characteristics that underline this research paradigm, including neural-symbolic integration, knowledge representation, knowledge embedding, and functionality. Next, we briefly discuss the successfu
    

