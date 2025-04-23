# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Cross--domain Fiber Cluster Shape Analysis for Language Performance Cognitive Score Prediction](https://arxiv.org/abs/2403.19001) | 本研究通过新颖的框架SFFormer，结合了多头交叉注意力特征融合模块，基于dMRI纤维束追踪，预测了主观语言表现，拓展了脑结构与人类认知功能的关联研究。 |
| [^2] | [Ambient Diffusion Posterior Sampling: Solving Inverse Problems with Diffusion Models trained on Corrupted Data](https://arxiv.org/abs/2403.08728) | 提出了一种使用环境扩散后验采样解决逆问题的框架，能在受损数据上训练的扩散模型上表现出色，并在图像恢复和MRI模型训练中取得优越性能。 |
| [^3] | [LASER: Neuro-Symbolic Learning of Semantic Video Representations.](http://arxiv.org/abs/2304.07647) | LASER提出了一种神经符号学习方法来学习语义视频表示，通过逻辑规范捕捉视频数据中的时空属性，能够对齐原始视频和规范，有效地训练低级感知模型以提取符合所需高级规范的视频表示。 |

# 详细

[^1]: 跨领域的纤维簇形状分析用于语言表现认知分数预测

    Cross--domain Fiber Cluster Shape Analysis for Language Performance Cognitive Score Prediction

    [https://arxiv.org/abs/2403.19001](https://arxiv.org/abs/2403.19001)

    本研究通过新颖的框架SFFormer，结合了多头交叉注意力特征融合模块，基于dMRI纤维束追踪，预测了主观语言表现，拓展了脑结构与人类认知功能的关联研究。

    

    形状在计算机图形学中扮演重要角色，提供了有关对象形态和功能的信息特征。脑成像中的形状分析可帮助解释人脑结构和功能的相关性。本研究调查了大脑的3D白质连接的形状及其与人类认知功能的潜在预测关系。我们使用扩散磁共振成像（dMRI）纤维束追踪将大脑连接重建为3D点序列。为了描述每个连接，我们提取了12个形状描述符以及传统的dMRI连接和组织微结构特征。我们引入了一种新颖的框架，形状融合纤维簇变换器（SFFormer），利用多头交叉注意力特征融合模块基于dMRI纤维束追踪来预测特定个体的语言表现。我们在一个大型数据集上评估了该方法的性能。

    arXiv:2403.19001v1 Announce Type: cross  Abstract: Shape plays an important role in computer graphics, offering informative features to convey an object's morphology and functionality. Shape analysis in brain imaging can help interpret structural and functionality correlations of the human brain. In this work, we investigate the shape of the brain's 3D white matter connections and its potential predictive relationship to human cognitive function. We reconstruct brain connections as sequences of 3D points using diffusion magnetic resonance imaging (dMRI) tractography. To describe each connection, we extract 12 shape descriptors in addition to traditional dMRI connectivity and tissue microstructure features. We introduce a novel framework, Shape--fused Fiber Cluster Transformer (SFFormer), that leverages a multi-head cross-attention feature fusion module to predict subject-specific language performance based on dMRI tractography. We assess the performance of the method on a large dataset
    
[^2]: 使用环境扩散后验采样：在受损数据上训练的扩散模型解决逆问题

    Ambient Diffusion Posterior Sampling: Solving Inverse Problems with Diffusion Models trained on Corrupted Data

    [https://arxiv.org/abs/2403.08728](https://arxiv.org/abs/2403.08728)

    提出了一种使用环境扩散后验采样解决逆问题的框架，能在受损数据上训练的扩散模型上表现出色，并在图像恢复和MRI模型训练中取得优越性能。

    

    我们提供了一个框架，用于使用从线性受损数据中学习的扩散模型解决逆问题。我们的方法，Ambient Diffusion Posterior Sampling (A-DPS)，利用一个预先在一种类型的损坏数据上进行过训练的生成模型，以在可能来自不同前向过程（例如图像模糊）的测量条件下执行后验采样。我们在标准自然图像数据集（CelebA、FFHQ 和 AFHQ）上测试了我们的方法的有效性，并展示了 A-DPS 有时在速度和性能上都能胜过在清洁数据上训练的模型，用于几个图像恢复任务。我们进一步扩展了环境扩散框架，以仅访问傅里叶子采样的多线圈 MRI 测量数据来训练 MRI 模型，其加速因子为不同的加速因子（R=2、4、6、8）。我们再次观察到，在高度子采样数据上训练的模型更适用于解决高加速 MRI 逆问题。

    arXiv:2403.08728v1 Announce Type: cross  Abstract: We provide a framework for solving inverse problems with diffusion models learned from linearly corrupted data. Our method, Ambient Diffusion Posterior Sampling (A-DPS), leverages a generative model pre-trained on one type of corruption (e.g. image inpainting) to perform posterior sampling conditioned on measurements from a potentially different forward process (e.g. image blurring). We test the efficacy of our approach on standard natural image datasets (CelebA, FFHQ, and AFHQ) and we show that A-DPS can sometimes outperform models trained on clean data for several image restoration tasks in both speed and performance. We further extend the Ambient Diffusion framework to train MRI models with access only to Fourier subsampled multi-coil MRI measurements at various acceleration factors (R=2, 4, 6, 8). We again observe that models trained on highly subsampled data are better priors for solving inverse problems in the high acceleration r
    
[^3]: LASER：神经符号学习语义视频表示

    LASER: Neuro-Symbolic Learning of Semantic Video Representations. (arXiv:2304.07647v1 [cs.CV])

    [http://arxiv.org/abs/2304.07647](http://arxiv.org/abs/2304.07647)

    LASER提出了一种神经符号学习方法来学习语义视频表示，通过逻辑规范捕捉视频数据中的时空属性，能够对齐原始视频和规范，有效地训练低级感知模型以提取符合所需高级规范的视频表示。

    

    现代涉及视频的AI应用（如视频-文本对齐、视频搜索和视频字幕）受益于对视频语义的细致理解。现有的视频理解方法要么需要大量注释，要么基于不可解释的通用嵌入，可能会忽略重要细节。我们提出了LASER，这是一种神经符号方法，通过利用能够捕捉视频数据中丰富的时空属性的逻辑规范来学习语义视频表示。特别地，我们通过原始视频与规范之间的对齐来公式化问题。对齐过程有效地训练了低层感知模型，以提取符合所需高层规范的细粒度视频表示。我们的流程可以端到端地训练，并可纳入从规范导出的对比和语义损失函数。我们在两个具有丰富空间和时间信息的数据集上评估了我们的方法。

    Modern AI applications involving video, such as video-text alignment, video search, and video captioning, benefit from a fine-grained understanding of video semantics. Existing approaches for video understanding are either data-hungry and need low-level annotation, or are based on general embeddings that are uninterpretable and can miss important details. We propose LASER, a neuro-symbolic approach that learns semantic video representations by leveraging logic specifications that can capture rich spatial and temporal properties in video data. In particular, we formulate the problem in terms of alignment between raw videos and specifications. The alignment process efficiently trains low-level perception models to extract a fine-grained video representation that conforms to the desired high-level specification. Our pipeline can be trained end-to-end and can incorporate contrastive and semantic loss functions derived from specifications. We evaluate our method on two datasets with rich sp
    

