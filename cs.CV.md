# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CMP: Cooperative Motion Prediction with Multi-Agent Communication](https://arxiv.org/abs/2403.17916) | 该论文提出了一种名为CMP的方法，利用LiDAR信号作为输入，通过合作感知和运动预测模块共享信息，解决了合作运动预测的问题。 |
| [^2] | [Non-autoregressive Sequence-to-Sequence Vision-Language Models](https://arxiv.org/abs/2403.02249) | 提出了一种非自回归序列到序列视觉语言模型，通过在解码器中边际化多个推理路径的方式，实现了对标记的联合分布建模，从而在保持性能的同时加快了推理速度。 |
| [^3] | [Subobject-level Image Tokenization](https://arxiv.org/abs/2402.14327) | 提出一种在子对象级别进行图像标记的方法，通过序列自编码器将子对象段压缩为紧凑的嵌入向量，实现了有效地将图像转换为对象和属性描述的学习。 |

# 详细

[^1]: CMP：具有多智能体通信的合作运动预测

    CMP: Cooperative Motion Prediction with Multi-Agent Communication

    [https://arxiv.org/abs/2403.17916](https://arxiv.org/abs/2403.17916)

    该论文提出了一种名为CMP的方法，利用LiDAR信号作为输入，通过合作感知和运动预测模块共享信息，解决了合作运动预测的问题。

    

    随着自动驾驶车辆（AVs）的发展和车联网（V2X）通信的成熟，合作连接的自动化车辆（CAVs）的功能变得可能。本文基于合作感知，探讨了合作运动预测的可行性和有效性。我们的方法CMP以LiDAR信号作为输入，以增强跟踪和预测能力。与过去专注于合作感知或运动预测的工作不同，我们的框架是我们所知的第一个解决CAVs在感知和预测模块中共享信息的统一问题。我们的设计中还融入了能够容忍现实V2X带宽限制和传输延迟的独特能力，同时处理庞大的感知表示。我们还提出了预测聚合模块，统一了预测

    arXiv:2403.17916v1 Announce Type: cross  Abstract: The confluence of the advancement of Autonomous Vehicles (AVs) and the maturity of Vehicle-to-Everything (V2X) communication has enabled the capability of cooperative connected and automated vehicles (CAVs). Building on top of cooperative perception, this paper explores the feasibility and effectiveness of cooperative motion prediction. Our method, CMP, takes LiDAR signals as input to enhance tracking and prediction capabilities. Unlike previous work that focuses separately on either cooperative perception or motion prediction, our framework, to the best of our knowledge, is the first to address the unified problem where CAVs share information in both perception and prediction modules. Incorporated into our design is the unique capability to tolerate realistic V2X bandwidth limitations and transmission delays, while dealing with bulky perception representations. We also propose a prediction aggregation module, which unifies the predict
    
[^2]: 非自回归序列到序列视觉语言模型

    Non-autoregressive Sequence-to-Sequence Vision-Language Models

    [https://arxiv.org/abs/2403.02249](https://arxiv.org/abs/2403.02249)

    提出了一种非自回归序列到序列视觉语言模型，通过在解码器中边际化多个推理路径的方式，实现了对标记的联合分布建模，从而在保持性能的同时加快了推理速度。

    

    序列到序列的视觉语言模型表现出了潜力，但由于它们生成预测的自回归方式，它们的推理延迟限制了它们的适用性。我们提出了一个并行解码的序列到序列视觉语言模型，使用Query-CTC损失进行训练，在解码器中边际化多个推理路径。这使我们能够对标记的联合分布进行建模，而不像自回归模型那样限制在条件分布上。结果模型NARVL在推理时间上达到了与最新自回归对应物相当的性能，但更快，从与顺序生成标记相关的线性复杂度减少到常量时间联合推理的范式。

    arXiv:2403.02249v1 Announce Type: cross  Abstract: Sequence-to-sequence vision-language models are showing promise, but their applicability is limited by their inference latency due to their autoregressive way of generating predictions. We propose a parallel decoding sequence-to-sequence vision-language model, trained with a Query-CTC loss, that marginalizes over multiple inference paths in the decoder. This allows us to model the joint distribution of tokens, rather than restricting to conditional distribution as in an autoregressive model. The resulting model, NARVL, achieves performance on-par with its state-of-the-art autoregressive counterpart, but is faster at inference time, reducing from the linear complexity associated with the sequential generation of tokens to a paradigm of constant time joint inference.
    
[^3]: 子对象级图像标记化

    Subobject-level Image Tokenization

    [https://arxiv.org/abs/2402.14327](https://arxiv.org/abs/2402.14327)

    提出一种在子对象级别进行图像标记的方法，通过序列自编码器将子对象段压缩为紧凑的嵌入向量，实现了有效地将图像转换为对象和属性描述的学习。

    

    基于Transformer的视觉模型通常将图像标记为固定大小的方形补丁作为输入单元，这种方法缺乏对图像内容的适应性，并忽略了固有的像素分组结构。受语言模型广泛采用的子词标记化启发，我们提出了一种在子对象级别进行图像标记的方法，其中子对象由通过分割模型（例如，分割任何模型）获得的具有语义意义的图像段表示。为了实现基于子对象标记化的学习系统，我们首先引入了一个序列自编码器（SeqAE），将不同大小和形状的子对象段压缩为紧凑的嵌入向量，然后将子对象嵌入馈送到大型语言模型进行视觉语言学习。实证结果表明，我们的子对象级别标记化显著促进了有效地将图像转换为对象和属性描述的学习。

    arXiv:2402.14327v1 Announce Type: cross  Abstract: Transformer-based vision models typically tokenize images into fixed-size square patches as input units, which lacks the adaptability to image content and overlooks the inherent pixel grouping structure. Inspired by the subword tokenization widely adopted in language models, we propose an image tokenizer at a subobject level, where the subobjects are represented by semantically meaningful image segments obtained by segmentation models (e.g., segment anything models). To implement a learning system based on subobject tokenization, we first introduced a Sequence-to-sequence AutoEncoder (SeqAE) to compress subobject segments of varying sizes and shapes into compact embedding vectors, then fed the subobject embeddings into a large language model for vision language learning. Empirical results demonstrated that our subobject-level tokenization significantly facilitates efficient learning of translating images into object and attribute descr
    

