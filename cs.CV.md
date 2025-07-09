# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TopCoW: Benchmarking Topology-Aware Anatomical Segmentation of the Circle of Willis (CoW) for CTA and MRA.](http://arxiv.org/abs/2312.17670) | 这项研究提出了TopCoW挑战，通过发布具有13种血管组分注释的Willis循环（CoW）数据集，并使用虚拟现实（VR）技术进行拓扑感知解剖分割，解决了手动和耗时的CoW表征问题。 |
| [^2] | [Annotating 8,000 Abdominal CT Volumes for Multi-Organ Segmentation in Three Weeks.](http://arxiv.org/abs/2305.09666) | 本文提出了一种高效方法在短时间内标记8000个腹部CT扫描中的8个器官，建立了迄今为止最大的多器官数据集。 |

# 详细

[^1]: TopCoW：基于拓扑感知解剖分割的Willis循环（CoW）在CTA和MRA中的基准测试

    TopCoW: Benchmarking Topology-Aware Anatomical Segmentation of the Circle of Willis (CoW) for CTA and MRA. (arXiv:2312.17670v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2312.17670](http://arxiv.org/abs/2312.17670)

    这项研究提出了TopCoW挑战，通过发布具有13种血管组分注释的Willis循环（CoW）数据集，并使用虚拟现实（VR）技术进行拓扑感知解剖分割，解决了手动和耗时的CoW表征问题。

    

    Willis循环（CoW）是连接大脑主要循环的重要动脉网络。其血管结构被认为影响着严重神经血管疾病的风险、严重程度和临床结果。然而，对高度变化的CoW解剖进行表征仍然是一项需要手动和耗时的专家任务。CoW通常通过两种血管造影成像模式进行成像，即磁共振血管成像（MRA）和计算机断层血管造影（CTA），但是关于CTA的CoW解剖的公共数据集及其注释非常有限。因此，我们在2023年组织了TopCoW挑战赛，并发布了一个带有注释的CoW数据集。TopCoW数据集是第一个具有13种可能的CoW血管组分的体素级注释的公共数据集，通过虚拟现实（VR）技术实现。它也是第一个带有来自同一患者的成对MRA和CTA的大型数据集。TopCoW挑战将CoW表征问题形式化为多类问题。

    The Circle of Willis (CoW) is an important network of arteries connecting major circulations of the brain. Its vascular architecture is believed to affect the risk, severity, and clinical outcome of serious neuro-vascular diseases. However, characterizing the highly variable CoW anatomy is still a manual and time-consuming expert task. The CoW is usually imaged by two angiographic imaging modalities, magnetic resonance angiography (MRA) and computed tomography angiography (CTA), but there exist limited public datasets with annotations on CoW anatomy, especially for CTA. Therefore we organized the TopCoW Challenge in 2023 with the release of an annotated CoW dataset. The TopCoW dataset was the first public dataset with voxel-level annotations for thirteen possible CoW vessel components, enabled by virtual-reality (VR) technology. It was also the first large dataset with paired MRA and CTA from the same patients. TopCoW challenge formalized the CoW characterization problem as a multiclas
    
[^2]: 在三周内为8,000个腹部CT扫描标注多器官分割

    Annotating 8,000 Abdominal CT Volumes for Multi-Organ Segmentation in Three Weeks. (arXiv:2305.09666v1 [eess.IV])

    [http://arxiv.org/abs/2305.09666](http://arxiv.org/abs/2305.09666)

    本文提出了一种高效方法在短时间内标记8000个腹部CT扫描中的8个器官，建立了迄今为止最大的多器官数据集。

    

    医学影像标注，特别是器官分割，是费时费力的。本文提出了一种系统高效的方法来加速器官分割的标注过程。我们标注了8,448个腹部CT扫描，标记了脾脏、肝脏、肾脏、胃、胆囊、胰腺、主动脉和下腔静脉。传统的标注方法需要一位经验丰富的标注员1600周，而我们的标注方法仅用了三周。

    Annotating medical images, particularly for organ segmentation, is laborious and time-consuming. For example, annotating an abdominal organ requires an estimated rate of 30-60 minutes per CT volume based on the expertise of an annotator and the size, visibility, and complexity of the organ. Therefore, publicly available datasets for multi-organ segmentation are often limited in data size and organ diversity. This paper proposes a systematic and efficient method to expedite the annotation process for organ segmentation. We have created the largest multi-organ dataset (by far) with the spleen, liver, kidneys, stomach, gallbladder, pancreas, aorta, and IVC annotated in 8,448 CT volumes, equating to 3.2 million slices. The conventional annotation methods would take an experienced annotator up to 1,600 weeks (or roughly 30.8 years) to complete this task. In contrast, our annotation method has accomplished this task in three weeks (based on an 8-hour workday, five days a week) while maintain
    

