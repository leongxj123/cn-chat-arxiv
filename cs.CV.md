# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RACR-MIL: Weakly Supervised Skin Cancer Grading using Rank-Aware Contextual Reasoning on Whole Slide Images.](http://arxiv.org/abs/2308.15618) | RACR-MIL是一个自动化的弱监督的皮肤癌分级方法，可以使用整张切片图像进行训练，无需细粒度的肿瘤注释，通过在切片图像中的瓦片章节中使用注意力机制的多实例学习，可以为切片图像分配分级。该方法主要创新包括使用空间和语义接近性定义切片图像图像以编码肿瘤区域的局部和非局部依赖关系以及使用序数排名约束保证注意力网络的性能 |

# 详细

[^1]: RACR-MIL：使用基于排名感知背景推理的整张切片图像进行弱监督的皮肤癌分级

    RACR-MIL: Weakly Supervised Skin Cancer Grading using Rank-Aware Contextual Reasoning on Whole Slide Images. (arXiv:2308.15618v1 [cs.CV])

    [http://arxiv.org/abs/2308.15618](http://arxiv.org/abs/2308.15618)

    RACR-MIL是一个自动化的弱监督的皮肤癌分级方法，可以使用整张切片图像进行训练，无需细粒度的肿瘤注释，通过在切片图像中的瓦片章节中使用注意力机制的多实例学习，可以为切片图像分配分级。该方法主要创新包括使用空间和语义接近性定义切片图像图像以编码肿瘤区域的局部和非局部依赖关系以及使用序数排名约束保证注意力网络的性能

    

    Cutaneous squamous cell cancer (cSCC) is the second most common skin cancer in the US. It is diagnosed by manual multi-class tumor grading using a tissue whole slide image (WSI), which is subjective and suffers from inter-pathologist variability. We propose an automated weakly-supervised grading approach for cSCC WSIs that is trained using WSI-level grade and does not require fine-grained tumor annotations. The proposed model, RACR-MIL, transforms each WSI into a bag of tiled patches and leverages attention-based multiple-instance learning to assign a WSI-level grade. We propose three key innovations to address general as well as cSCC-specific challenges in tumor grading. First, we leverage spatial and semantic proximity to define a WSI graph that encodes both local and non-local dependencies between tumor regions and leverage graph attention convolution to derive contextual patch features. Second, we introduce a novel ordinal ranking constraint on the patch attention network to ensure

    Cutaneous squamous cell cancer (cSCC) is the second most common skin cancer in the US. It is diagnosed by manual multi-class tumor grading using a tissue whole slide image (WSI), which is subjective and suffers from inter-pathologist variability. We propose an automated weakly-supervised grading approach for cSCC WSIs that is trained using WSI-level grade and does not require fine-grained tumor annotations. The proposed model, RACR-MIL, transforms each WSI into a bag of tiled patches and leverages attention-based multiple-instance learning to assign a WSI-level grade. We propose three key innovations to address general as well as cSCC-specific challenges in tumor grading. First, we leverage spatial and semantic proximity to define a WSI graph that encodes both local and non-local dependencies between tumor regions and leverage graph attention convolution to derive contextual patch features. Second, we introduce a novel ordinal ranking constraint on the patch attention network to ensure
    

