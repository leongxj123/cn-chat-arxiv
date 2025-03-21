# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CREMA: Multimodal Compositional Video Reasoning via Efficient Modular Adaptation and Fusion](https://arxiv.org/abs/2402.05889) | 该论文提出了一种名为CREMA的高效且模块化的模态融合框架，用于将任意新的模态注入视频推理。通过利用预训练模型增强多种信息模态，并引入查询转换器和融合模块，实现了灵活且有效的多模态组合推理。 |
| [^2] | [Karyotype AI for Precision Oncology.](http://arxiv.org/abs/2211.14312) | 本研究针对精准肿瘤学中的染色体分析问题，通过使用Fred Hutchinson癌症研究中心的大量数据，利用深度学习模型和拓扑视觉转换器(TopViTs)，成功开发出了一种自动识别染色体异常的方法。 |

# 详细

[^1]: CREMA: 通过有效的模块化适应和融合进行多模态组合视频推理

    CREMA: Multimodal Compositional Video Reasoning via Efficient Modular Adaptation and Fusion

    [https://arxiv.org/abs/2402.05889](https://arxiv.org/abs/2402.05889)

    该论文提出了一种名为CREMA的高效且模块化的模态融合框架，用于将任意新的模态注入视频推理。通过利用预训练模型增强多种信息模态，并引入查询转换器和融合模块，实现了灵活且有效的多模态组合推理。

    

    尽管在多模态组合推理方法方面取得了令人瞩目的进展，但由于处理固定模态输入并更新许多模型参数，仍然存在灵活性和效率方面的限制。本文解决了这些关键挑战，提出了CREMA，一种用于将任何新的模态注入视频推理的高效且模块化的模态融合框架。我们首先利用现有的预训练模型从给定的视频中增强多种信息模态（如光流、3D点云、音频），而无需额外的人工注释。接下来，我们引入了一个查询转换器，该转换器与每个可以访问的模态相关联，并具有多个参数高效的模块。它将多种模态特征投影到LLM令牌嵌入空间，使模型能够整合不同的数据类型以进行响应生成。此外，我们提出了一个融合模块，用于压缩多模态查询，在LLM中保持计算效率的同时进行融合组合。

    Despite impressive advancements in multimodal compositional reasoning approaches, they are still limited in their flexibility and efficiency by processing fixed modality inputs while updating a lot of model parameters. This paper tackles these critical challenges and proposes CREMA, an efficient and modular modality-fusion framework for injecting any new modality into video reasoning. We first augment multiple informative modalities (such as optical flow, 3D point cloud, audio) from given videos without extra human annotation by leveraging existing pre-trained models. Next, we introduce a query transformer with multiple parameter-efficient modules associated with each accessible modality. It projects diverse modality features to the LLM token embedding space, allowing the model to integrate different data types for response generation. Furthermore, we propose a fusion module designed to compress multimodal queries, maintaining computational efficiency in the LLM while combining additio
    
[^2]: 精准肿瘤学的染色体AI

    Karyotype AI for Precision Oncology. (arXiv:2211.14312v3 [q-bio.QM] UPDATED)

    [http://arxiv.org/abs/2211.14312](http://arxiv.org/abs/2211.14312)

    本研究针对精准肿瘤学中的染色体分析问题，通过使用Fred Hutchinson癌症研究中心的大量数据，利用深度学习模型和拓扑视觉转换器(TopViTs)，成功开发出了一种自动识别染色体异常的方法。

    

    染色体分析对于诊断遗传疾病至关重要。对于血液系统恶性肿瘤，通过染色体组型分析来发现体细胞突变是标准的护理方法。然而，染色体组型分析因为大部分是手动操作，且需要专业知识来识别和注释突变，所以昂贵且耗时。以Fred Hutchinson癌症研究中心过去五年的约10,000个患者标本和约50,000个染色体组型图片作为训练集，我们创建了一组代表单个染色体的标记图片。这些单个染色体用于训练和评估深度学习模型，以分类人类的24条染色体和识别染色体异常。具有最高准确性的模型使用了最近引入的拓扑视觉转换器(TopViTs)和二级块-托普利茨蒙版，以融入结构性归纳偏置。TopViT的性能优于CNN(Inc)

    Chromosome analysis is essential for diagnosing genetic disorders. For hematologic malignancies, identification of somatic clonal aberrations by karyotype analysis remains the standard of care. However, karyotyping is costly and time-consuming because of the largely manual process and the expertise required in identifying and annotating aberrations. Efforts to automate karyotype analysis to date fell short in aberration detection. Using a training set of ~10k patient specimens and ~50k karyograms from over 5 years from the Fred Hutchinson Cancer Center, we created a labeled set of images representing individual chromosomes. These individual chromosomes were used to train and assess deep learning models for classifying the 24 human chromosomes and identifying chromosomal aberrations. The top-accuracy models utilized the recently introduced Topological Vision Transformers (TopViTs) with 2-level-block-Toeplitz masking, to incorporate structural inductive bias. TopViT outperformed CNN (Inc
    

