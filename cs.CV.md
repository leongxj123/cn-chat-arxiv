# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CHOSEN: Contrastive Hypothesis Selection for Multi-View Depth Refinement](https://arxiv.org/abs/2404.02225) | CHOSEN是一个用于多视角深度细化的对比假设选择框架，通过对比学习和精心设计的假设特征，能够在多视角立体匹配中实现较高质量的深度和法线精度。 |
| [^2] | [Exploring Challenges in Deep Learning of Single-Station Ground Motion Records](https://arxiv.org/abs/2403.07569) | 本研究旨在评估深度学习模型从场地运动记录中学习的效果，并探讨辅助信息对此过程的影响。 |
| [^3] | [Learning with Noisy Foundation Models](https://arxiv.org/abs/2403.06869) | 本文首次全面了解和分析了预训练数据集中的噪声性质，有效减轻其对下游任务影响。 |

# 详细

[^1]: CHOSEN：用于多视角深度细化的对比假设选择

    CHOSEN: Contrastive Hypothesis Selection for Multi-View Depth Refinement

    [https://arxiv.org/abs/2404.02225](https://arxiv.org/abs/2404.02225)

    CHOSEN是一个用于多视角深度细化的对比假设选择框架，通过对比学习和精心设计的假设特征，能够在多视角立体匹配中实现较高质量的深度和法线精度。

    

    我们提出了CHOSEN，这是一个简单而灵活、强大且有效的多视角深度细化框架。它可以应用于任何现有的多视角立体匹配流程中，具有针对不同多视角采集系统（如相机相对定位和镜头）的直观泛化能力。给定初始深度估计，CHOSEN迭代地重新采样并选择最佳假设，并自动适应由采集系统确定的不同度量或固有尺度。我们方法的关键在于在适当的解决方案空间中应用对比学习以及精心设计的假设特征，基于这些特征，可以有效地区分正负假设。将CHOSEN集成到简单的基线多视角立体匹配流程中，在深度和法线精度方面提供了令人印象深刻的质量，与许多当前基于深度学习的多视角立体匹配流程相比有所提高。

    arXiv:2404.02225v1 Announce Type: cross  Abstract: We propose CHOSEN, a simple yet flexible, robust and effective multi-view depth refinement framework. It can be employed in any existing multi-view stereo pipeline, with straightforward generalization capability for different multi-view capture systems such as camera relative positioning and lenses. Given an initial depth estimation, CHOSEN iteratively re-samples and selects the best hypotheses, and automatically adapts to different metric or intrinsic scales determined by the capture system. The key to our approach is the application of contrastive learning in an appropriate solution space and a carefully designed hypothesis feature, based on which positive and negative hypotheses can be effectively distinguished. Integrated in a simple baseline multi-view stereo pipeline, CHOSEN delivers impressive quality in terms of depth and normal accuracy compared to many current deep learning based multi-view stereo pipelines.
    
[^2]: 探究单台场地运动记录的深度学习挑战

    Exploring Challenges in Deep Learning of Single-Station Ground Motion Records

    [https://arxiv.org/abs/2403.07569](https://arxiv.org/abs/2403.07569)

    本研究旨在评估深度学习模型从场地运动记录中学习的效果，并探讨辅助信息对此过程的影响。

    

    当代的深度学习模型在地震学和地震工程的各种应用中展现出了令人期待的结果。这些模型主要依赖利用场地运动记录进行地震事件分类、定位、地震早期预警系统和结构健康监测等任务。然而，这些模型从这些复杂的时间序列信号中有效学习的程度尚未得到彻底分析。本研究的目标是评估辅助信息（如地震相到达时间或网络内地震台站分布）在从场地运动记录中进行深度学习过程中的主导程度，可能会影响其有效性。我们对两种深度学习模型进行超参数搜索，评估它们在从场地运动记录中进行深度学习的有效性，同时检查辅助信息的影响。

    arXiv:2403.07569v1 Announce Type: cross  Abstract: Contemporary deep learning models have demonstrated promising results across various applications within seismology and earthquake engineering. These models rely primarily on utilizing ground motion records for tasks such as earthquake event classification, localization, earthquake early warning systems, and structural health monitoring. However, the extent to which these models effectively learn from these complex time-series signals has not been thoroughly analyzed. In this study, our objective is to evaluate the degree to which auxiliary information, such as seismic phase arrival times or seismic station distribution within a network, dominates the process of deep learning from ground motion records, potentially hindering its effectiveness. We perform a hyperparameter search on two deep learning models to assess their effectiveness in deep learning from ground motion records while also examining the impact of auxiliary information o
    
[^3]: 在有噪声基础模型中学习

    Learning with Noisy Foundation Models

    [https://arxiv.org/abs/2403.06869](https://arxiv.org/abs/2403.06869)

    本文首次全面了解和分析了预训练数据集中的噪声性质，有效减轻其对下游任务影响。

    

    基础模型通常是在大规模数据集上进行预训练，然后通过调整来适应下游任务。然而，大规模预训练数据集往往无法获取或成本过高，可能包含标签噪声，这可能会对模型的泛化能力造成不利影响，并带来意想不到的风险。本文是首个全面了解和分析预训练数据集中噪声性质，并有效减轻其对下游任务影响的工作。具体而言，通过在合成有噪声的ImageNet-1K、YFCC15M和CC12M数据集上进行完全监督和图像-文本对比预训练的广泛实验，我们证明了，尽管预训练中的轻微噪声可以使同领域（ID）性能受益，即训练和测试数据共享类似分布，但它总是会破坏跨领域（OOD）性能，在那里训练和测试分布明显不同。

    arXiv:2403.06869v1 Announce Type: cross  Abstract: Foundation models are usually pre-trained on large-scale datasets and then adapted to downstream tasks through tuning. However, the large-scale pre-training datasets, often inaccessible or too expensive to handle, can contain label noise that may adversely affect the generalization of the model and pose unexpected risks. This paper stands out as the first work to comprehensively understand and analyze the nature of noise in pre-training datasets and then effectively mitigate its impacts on downstream tasks. Specifically, through extensive experiments of fully-supervised and image-text contrastive pre-training on synthetic noisy ImageNet-1K, YFCC15M, and CC12M datasets, we demonstrate that, while slight noise in pre-training can benefit in-domain (ID) performance, where the training and testing data share a similar distribution, it always deteriorates out-of-domain (OOD) performance, where training and testing distributions are signific
    

