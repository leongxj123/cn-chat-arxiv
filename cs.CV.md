# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [HyperFusion: A Hypernetwork Approach to Multimodal Integration of Tabular and Medical Imaging Data for Predictive Modeling](https://arxiv.org/abs/2403.13319) | 提出一种基于超网络的新框架，通过将图像处理条件设置在EHR的值和测量上，以整合临床成像和表格数据，旨在利用这些数据中的互补信息。 |
| [^2] | [Recognize Any Regions.](http://arxiv.org/abs/2311.01373) | 本文提出了一种名为RegionSpot的新型、通用且高效的区域识别架构，旨在解决在计算机视觉中理解无约束图像中区域的语义的挑战。 |

# 详细

[^1]: HyperFusion：一种用于预测建模的多模态整合表格和医学成像数据的超网络方法

    HyperFusion: A Hypernetwork Approach to Multimodal Integration of Tabular and Medical Imaging Data for Predictive Modeling

    [https://arxiv.org/abs/2403.13319](https://arxiv.org/abs/2403.13319)

    提出一种基于超网络的新框架，通过将图像处理条件设置在EHR的值和测量上，以整合临床成像和表格数据，旨在利用这些数据中的互补信息。

    

    ARXIV: 2403.13319v1 公告类型: 交叉摘要: 整合各种临床模式，如医学成像和患者电子健康记录（EHR）获得的表格数据，是现代医疗保健的关键方面。多源数据的综合分析可以全面了解患者的状况，并可以增强诊断和治疗决策。深度神经网络（DNN）在医学领域的多模态任务中一直展示出出色的性能。然而，有效地将医学成像与以数字表格数据表示的临床、人口统计和遗传信息进行融合的复杂努力仍然是一个高度活跃的持续研究追求。

    arXiv:2403.13319v1 Announce Type: cross  Abstract: The integration of diverse clinical modalities such as medical imaging and the tabular data obtained by the patients' Electronic Health Records (EHRs) is a crucial aspect of modern healthcare. The integrative analysis of multiple sources can provide a comprehensive understanding of a patient's condition and can enhance diagnoses and treatment decisions. Deep Neural Networks (DNNs) consistently showcase outstanding performance in a wide range of multimodal tasks in the medical domain. However, the complex endeavor of effectively merging medical imaging with clinical, demographic and genetic information represented as numerical tabular data remains a highly active and ongoing research pursuit.   We present a novel framework based on hypernetworks to fuse clinical imaging and tabular data by conditioning the image processing on the EHR's values and measurements. This approach aims to leverage the complementary information present in these
    
[^2]: 认证任何区域

    Recognize Any Regions. (arXiv:2311.01373v1 [cs.CV])

    [http://arxiv.org/abs/2311.01373](http://arxiv.org/abs/2311.01373)

    本文提出了一种名为RegionSpot的新型、通用且高效的区域识别架构，旨在解决在计算机视觉中理解无约束图像中区域的语义的挑战。

    

    理解无约束图像中各个区域或块的语义，例如在开放世界物体检测中，代表了一项关键而具有挑战性的计算机视觉任务。在基于强大的图像级视觉语言（ViL）基础模型如CLIP的成功基础上，最近的努力要么通过使用广泛的区域-标签对集合从头开始训练对比模型，要么将检测模型的输出与区域建议的图像级表示对齐，以发挥它们的能力。尽管取得了显著进展，但这些方法都受到计算密集型的训练需求、数据噪声的影响以及环境信息的不足等限制。为了解决这些问题，我们探索了现成的基础模型的协同潜力，利用它们在定位和语义方面的各自优势。我们引入了一种新颖的、通用的、高效的区域识别架构，称为RegionSpot。

    Understanding the semantics of individual regions or patches within unconstrained images, such as in open-world object detection, represents a critical yet challenging task in computer vision. Building on the success of powerful image-level vision-language (ViL) foundation models like CLIP, recent efforts have sought to harness their capabilities by either training a contrastive model from scratch with an extensive collection of region-label pairs or aligning the outputs of a detection model with image-level representations of region proposals. Despite notable progress, these approaches are plagued by computationally intensive training requirements, susceptibility to data noise, and deficiency in contextual information. To address these limitations, we explore the synergistic potential of off-the-shelf foundation models, leveraging their respective strengths in localization and semantics. We introduce a novel, generic, and efficient region recognition architecture, named RegionSpot, de
    

