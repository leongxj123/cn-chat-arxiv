# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Leveraging Foundation Models for Content-Based Medical Image Retrieval in Radiology](https://arxiv.org/abs/2403.06567) | 基于内容的医学图像检索中，利用基础模型作为特征提取器，无需微调即可取得与专门模型竞争的性能，尤其在检索病理特征方面具有较大困难。 |
| [^2] | [Open-world Machine Learning: A Review and New Outlooks](https://arxiv.org/abs/2403.01759) | 通过研究未知拒绝、新类别发现和类别增量学习，本文拓展了开放世界机器学习领域，提出了未来研究的多个潜在方向 |
| [^3] | [Disentangling representations of retinal images with generative models](https://arxiv.org/abs/2402.19186) | 引入一种新颖的视网膜底图像群体模型，有效解开患者属性与相机效果，实现可控且高度逼真的图像生成。 |
| [^4] | [Enhanced Distribution Alignment for Post-Training Quantization of Diffusion Models.](http://arxiv.org/abs/2401.04585) | 本文提出了一种扩展分布对齐方法以解决后训练量化对于弥散模型的分布不匹配问题，该方法在低延迟应用中具有较高的潜力，并且能有效提升性能。 |
| [^5] | [Latest Trends in Artificial Intelligence Technology: A Scoping Review.](http://arxiv.org/abs/2305.04532) | 本文对当前最先进的人工智能技术进行了范围评估，并要求对技术解决方案进行测试、使用公认数据集以及确保结果可复制。 |
| [^6] | [Indeterminate Probability Neural Network.](http://arxiv.org/abs/2303.11536) | 本文提出了一种新型通用模型——不定概率神经网络；它可以进行无监督聚类和使用很小的神经网络处理大规模分类，其理论优势体现在新的概率理论和神经网络框架中。 |

# 详细

[^1]: 利用基础模型进行放射学中基于内容的医学图像检索

    Leveraging Foundation Models for Content-Based Medical Image Retrieval in Radiology

    [https://arxiv.org/abs/2403.06567](https://arxiv.org/abs/2403.06567)

    基于内容的医学图像检索中，利用基础模型作为特征提取器，无需微调即可取得与专门模型竞争的性能，尤其在检索病理特征方面具有较大困难。

    

    Content-based image retrieval（CBIR）有望显著改善放射学中的诊断辅助和医学研究。我们提出利用视觉基础模型作为强大且多功能的现成特征提取器，用于基于内容的医学图像检索。通过在涵盖四种模态和161种病理学的160万张2D放射图像的全面数据集上对这些模型进行基准测试，我们发现弱监督模型表现优异，P@1可达0.594。这种性能不仅与专门化模型竞争，而且无需进行微调。我们的分析进一步探讨了检索病理学与解剖结构的挑战，表明准确检索病理特征更具挑战性。

    arXiv:2403.06567v1 Announce Type: cross  Abstract: Content-based image retrieval (CBIR) has the potential to significantly improve diagnostic aid and medical research in radiology. Current CBIR systems face limitations due to their specialization to certain pathologies, limiting their utility. In response, we propose using vision foundation models as powerful and versatile off-the-shelf feature extractors for content-based medical image retrieval. By benchmarking these models on a comprehensive dataset of 1.6 million 2D radiological images spanning four modalities and 161 pathologies, we identify weakly-supervised models as superior, achieving a P@1 of up to 0.594. This performance not only competes with a specialized model but does so without the need for fine-tuning. Our analysis further explores the challenges in retrieving pathological versus anatomical structures, indicating that accurate retrieval of pathological features presents greater difficulty. Despite these challenges, our
    
[^2]: 开放世界机器学习：回顾与新展望

    Open-world Machine Learning: A Review and New Outlooks

    [https://arxiv.org/abs/2403.01759](https://arxiv.org/abs/2403.01759)

    通过研究未知拒绝、新类别发现和类别增量学习，本文拓展了开放世界机器学习领域，提出了未来研究的多个潜在方向

    

    机器学习在许多应用中取得了显著成功。然而，现有研究主要基于封闭世界假设，即假定环境是静态的，模型一旦部署就是固定的。在许多现实应用中，这种基本且相当幼稚的假设可能不成立，因为开放环境复杂、动态且充满未知。在这种情况下，拒绝未知、发现新奇点，然后逐步学习，可以使模型像生物系统一样安全地并持续进化。本文通过研究未知拒绝、新类别发现和类别增量学习在统一范式中，提供了对开放世界机器学习的整体观点。详细讨论了当前方法的挑战、原则和局限性。最后，我们讨论了几个未来研究的潜在方向。本文旨在提供一份综述

    arXiv:2403.01759v1 Announce Type: new  Abstract: Machine learning has achieved remarkable success in many applications. However, existing studies are largely based on the closed-world assumption, which assumes that the environment is stationary, and the model is fixed once deployed. In many real-world applications, this fundamental and rather naive assumption may not hold because an open environment is complex, dynamic, and full of unknowns. In such cases, rejecting unknowns, discovering novelties, and then incrementally learning them, could enable models to be safe and evolve continually as biological systems do. This paper provides a holistic view of open-world machine learning by investigating unknown rejection, novel class discovery, and class-incremental learning in a unified paradigm. The challenges, principles, and limitations of current methodologies are discussed in detail. Finally, we discuss several potential directions for future research. This paper aims to provide a compr
    
[^3]: 用生成模型解开视网膜图像的表征

    Disentangling representations of retinal images with generative models

    [https://arxiv.org/abs/2402.19186](https://arxiv.org/abs/2402.19186)

    引入一种新颖的视网膜底图像群体模型，有效解开患者属性与相机效果，实现可控且高度逼真的图像生成。

    

    视网膜底图像在早期检测眼部疾病中起着至关重要的作用，最近的研究甚至表明，利用深度学习方法，这些图像还可以用于检测心血管风险因素和神经系统疾病。然而，这些图像受技术因素的影响可能对眼科领域可靠的人工智能应用构成挑战。例如，大型底图队列往往受到相机类型、图像质量或照明水平等因素的影响，存在学习快捷方式而不是图像生成过程背后因果关系的风险。在这里，我们提出了一个新颖的视网膜底图像群体模型，有效地解开了患者属性与相机效果，从而实现了可控且高度逼真的图像生成。为了实现这一目标，我们提出了一个基于距离相关性的新颖解开损失。通过定性和定量分析，我们展示了...

    arXiv:2402.19186v1 Announce Type: cross  Abstract: Retinal fundus images play a crucial role in the early detection of eye diseases and, using deep learning approaches, recent studies have even demonstrated their potential for detecting cardiovascular risk factors and neurological disorders. However, the impact of technical factors on these images can pose challenges for reliable AI applications in ophthalmology. For example, large fundus cohorts are often confounded by factors like camera type, image quality or illumination level, bearing the risk of learning shortcuts rather than the causal relationships behind the image generation process. Here, we introduce a novel population model for retinal fundus images that effectively disentangles patient attributes from camera effects, thus enabling controllable and highly realistic image generation. To achieve this, we propose a novel disentanglement loss based on distance correlation. Through qualitative and quantitative analyses, we demon
    
[^4]: 扩展分布对齐来实现弥散模型的后训练量化

    Enhanced Distribution Alignment for Post-Training Quantization of Diffusion Models. (arXiv:2401.04585v1 [cs.CV])

    [http://arxiv.org/abs/2401.04585](http://arxiv.org/abs/2401.04585)

    本文提出了一种扩展分布对齐方法以解决后训练量化对于弥散模型的分布不匹配问题，该方法在低延迟应用中具有较高的潜力，并且能有效提升性能。

    

    通过迭代噪声估计，扩散模型在图像生成任务中取得了巨大成功。然而，繁重的去噪过程和复杂的神经网络阻碍了它们在实际场景中的低延迟应用。量化可以有效降低模型复杂度，而后训练量化(PTQ)在加速去噪过程方面具有很高的潜力，并且不需要微调。不幸的是，我们发现由于不同去噪步骤中激活的高度动态分布，现有的扩散模型的PTQ方法在校准样本和重构输出两个层面上都存在分布不匹配的问题，导致性能远低于令人满意的水平，特别是在低位情况下。在本文中，我们提出了增强的分布对齐用于弥散模型的后训练量化(EDA-DM)来解决上述问题。具体来说，在校准样本层面，我们基于...[缺省]

    Diffusion models have achieved great success in image generation tasks through iterative noise estimation. However, the heavy denoising process and complex neural networks hinder their low-latency applications in real-world scenarios. Quantization can effectively reduce model complexity, and post-training quantization (PTQ), which does not require fine-tuning, is highly promising in accelerating the denoising process. Unfortunately, we find that due to the highly dynamic distribution of activations in different denoising steps, existing PTQ methods for diffusion models suffer from distribution mismatch issues at both calibration sample level and reconstruction output level, which makes the performance far from satisfactory, especially in low-bit cases. In this paper, we propose Enhanced Distribution Alignment for Post-Training Quantization of Diffusion Models (EDA-DM) to address the above issues. Specifically, at the calibration sample level, we select calibration samples based on the 
    
[^5]: 人工智能技术的最新趋势：一个范围评估研究

    Latest Trends in Artificial Intelligence Technology: A Scoping Review. (arXiv:2305.04532v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.04532](http://arxiv.org/abs/2305.04532)

    本文对当前最先进的人工智能技术进行了范围评估，并要求对技术解决方案进行测试、使用公认数据集以及确保结果可复制。

    

    人工智能技术已经广泛应用于多个领域。智能手机、社交媒体平台、搜索引擎和自主驾驶车辆等应用程序都利用了人工智能技术以提高其性能。本研究按照 PRISMA 框架对当前最先进的人工智能技术进行了范围评估。目标是寻找应用于不同领域人工智能技术研究的最先进技术。从人工智能和机器学习领域选取了三个知名期刊：《人工智能研究杂志》、《机器学习研究杂志》和《机器学习》，并观察了2022年发表的文章。对技术解决方案进行了一定的资格要求：技术必须针对可比较的解决方案进行测试，必须使用公认或其他充分证明的数据集进行应用，并确保结果可复制。

    Artificial intelligence is more ubiquitous in multiple domains. Smartphones, social media platforms, search engines, and autonomous vehicles are just a few examples of applications that utilize artificial intelligence technologies to enhance their performance. This study carries out a scoping review of the current state-of-the-art artificial intelligence technologies following the PRISMA framework. The goal was to find the most advanced technologies used in different domains of artificial intelligence technology research. Three recognized journals were used from artificial intelligence and machine learning domain: Journal of Artificial Intelligence Research, Journal of Machine Learning Research, and Machine Learning, and articles published in 2022 were observed. Certain qualifications were laid for the technological solutions: the technology must be tested against comparable solutions, commonly approved or otherwise well justified datasets must be used while applying, and results must 
    
[^6]: 不定概率神经网络

    Indeterminate Probability Neural Network. (arXiv:2303.11536v1 [cs.LG])

    [http://arxiv.org/abs/2303.11536](http://arxiv.org/abs/2303.11536)

    本文提出了一种新型通用模型——不定概率神经网络；它可以进行无监督聚类和使用很小的神经网络处理大规模分类，其理论优势体现在新的概率理论和神经网络框架中。

    

    本文提出了一个称为IPNN的新型通用模型，它将神经网络和概率论结合在一起。在传统概率论中，概率的计算是基于事件的发生，而这在当前的神经网络中几乎不使用。因此，我们提出了一种新的概率理论，它是经典概率论的扩展，并使经典概率论成为我们理论的一种特殊情况。此外，对于我们提出的神经网络框架，神经网络的输出被定义为概率事件，并基于这些事件的统计分析，推导出分类任务的推理模型。IPNN展现了新的特性：它在进行分类的同时可以执行无监督聚类。此外，IPNN能够使用非常小的神经网络进行非常大的分类，例如100个输出节点的模型可以分类10亿类别。理论优势体现在新的概率理论和神经网络框架中，并且实验结果展示了IPNN在各种应用中的潜力。

    We propose a new general model called IPNN - Indeterminate Probability Neural Network, which combines neural network and probability theory together. In the classical probability theory, the calculation of probability is based on the occurrence of events, which is hardly used in current neural networks. In this paper, we propose a new general probability theory, which is an extension of classical probability theory, and makes classical probability theory a special case to our theory. Besides, for our proposed neural network framework, the output of neural network is defined as probability events, and based on the statistical analysis of these events, the inference model for classification task is deduced. IPNN shows new property: It can perform unsupervised clustering while doing classification. Besides, IPNN is capable of making very large classification with very small neural network, e.g. model with 100 output nodes can classify 10 billion categories. Theoretical advantages are refl
    

