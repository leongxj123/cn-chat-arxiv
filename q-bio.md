# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Karyotype AI for Precision Oncology.](http://arxiv.org/abs/2211.14312) | 本研究针对精准肿瘤学中的染色体分析问题，通过使用Fred Hutchinson癌症研究中心的大量数据，利用深度学习模型和拓扑视觉转换器(TopViTs)，成功开发出了一种自动识别染色体异常的方法。 |

# 详细

[^1]: 精准肿瘤学的染色体AI

    Karyotype AI for Precision Oncology. (arXiv:2211.14312v3 [q-bio.QM] UPDATED)

    [http://arxiv.org/abs/2211.14312](http://arxiv.org/abs/2211.14312)

    本研究针对精准肿瘤学中的染色体分析问题，通过使用Fred Hutchinson癌症研究中心的大量数据，利用深度学习模型和拓扑视觉转换器(TopViTs)，成功开发出了一种自动识别染色体异常的方法。

    

    染色体分析对于诊断遗传疾病至关重要。对于血液系统恶性肿瘤，通过染色体组型分析来发现体细胞突变是标准的护理方法。然而，染色体组型分析因为大部分是手动操作，且需要专业知识来识别和注释突变，所以昂贵且耗时。以Fred Hutchinson癌症研究中心过去五年的约10,000个患者标本和约50,000个染色体组型图片作为训练集，我们创建了一组代表单个染色体的标记图片。这些单个染色体用于训练和评估深度学习模型，以分类人类的24条染色体和识别染色体异常。具有最高准确性的模型使用了最近引入的拓扑视觉转换器(TopViTs)和二级块-托普利茨蒙版，以融入结构性归纳偏置。TopViT的性能优于CNN(Inc)

    Chromosome analysis is essential for diagnosing genetic disorders. For hematologic malignancies, identification of somatic clonal aberrations by karyotype analysis remains the standard of care. However, karyotyping is costly and time-consuming because of the largely manual process and the expertise required in identifying and annotating aberrations. Efforts to automate karyotype analysis to date fell short in aberration detection. Using a training set of ~10k patient specimens and ~50k karyograms from over 5 years from the Fred Hutchinson Cancer Center, we created a labeled set of images representing individual chromosomes. These individual chromosomes were used to train and assess deep learning models for classifying the 24 human chromosomes and identifying chromosomal aberrations. The top-accuracy models utilized the recently introduced Topological Vision Transformers (TopViTs) with 2-level-block-Toeplitz masking, to incorporate structural inductive bias. TopViT outperformed CNN (Inc
    

