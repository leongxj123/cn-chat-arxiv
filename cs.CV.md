# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TTA-Nav: Test-time Adaptive Reconstruction for Point-Goal Navigation under Visual Corruptions](https://arxiv.org/abs/2403.01977) | TTA-Nav提出了一种测试时自适应方法，通过引入自顶向下解码器，从损坏图像中重建出更清晰的图像，显著增强了点目标导航性能。 |
| [^2] | [TopCoW: Benchmarking Topology-Aware Anatomical Segmentation of the Circle of Willis (CoW) for CTA and MRA.](http://arxiv.org/abs/2312.17670) | 这项研究提出了TopCoW挑战，通过发布具有13种血管组分注释的Willis循环（CoW）数据集，并使用虚拟现实（VR）技术进行拓扑感知解剖分割，解决了手动和耗时的CoW表征问题。 |

# 详细

[^1]: TTA-Nav: 测试时自适应重建用于视觉损坏下的点目标导航

    TTA-Nav: Test-time Adaptive Reconstruction for Point-Goal Navigation under Visual Corruptions

    [https://arxiv.org/abs/2403.01977](https://arxiv.org/abs/2403.01977)

    TTA-Nav提出了一种测试时自适应方法，通过引入自顶向下解码器，从损坏图像中重建出更清晰的图像，显著增强了点目标导航性能。

    

    arXiv:2403.01977v1 公告类型: 跨  摘要: 在视觉损坏下的机器人导航是一个巨大的挑战。为了解决这一问题，我们提出了一种名为TTA-Nav的测试时自适应（TTA）方法，用于在视觉损坏下的点目标导航。我们的“即插即用”方法将自顶向下的解码器与预训练的导航模型相结合。首先，预训练的导航模型接收一个损坏的图像并提取特征。其次，自顶向下的解码器根据预训练模型提取的高级特征生成重建图像。然后，将损坏图像的重建图像馈送回预训练模型。最后，预训练模型再次进行前向传播以输出动作。尽管仅在清晰图像上训练，自顶向下的解码器可以从损坏图像中重建出更清晰的图像，无需基于梯度的自适应。具有我们自顶向下解码器的预训练导航模型显著提高了导航性能。

    arXiv:2403.01977v1 Announce Type: cross  Abstract: Robot navigation under visual corruption presents a formidable challenge. To address this, we propose a Test-time Adaptation (TTA) method, named as TTA-Nav, for point-goal navigation under visual corruptions. Our "plug-and-play" method incorporates a top-down decoder to a pre-trained navigation model. Firstly, the pre-trained navigation model gets a corrupted image and extracts features. Secondly, the top-down decoder produces the reconstruction given the high-level features extracted by the pre-trained model. Then, it feeds the reconstruction of a corrupted image back to the pre-trained model. Finally, the pre-trained model does forward pass again to output action. Despite being trained solely on clean images, the top-down decoder can reconstruct cleaner images from corrupted ones without the need for gradient-based adaptation. The pre-trained navigation model with our top-down decoder significantly enhances navigation performance acr
    
[^2]: TopCoW：基于拓扑感知解剖分割的Willis循环（CoW）在CTA和MRA中的基准测试

    TopCoW: Benchmarking Topology-Aware Anatomical Segmentation of the Circle of Willis (CoW) for CTA and MRA. (arXiv:2312.17670v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2312.17670](http://arxiv.org/abs/2312.17670)

    这项研究提出了TopCoW挑战，通过发布具有13种血管组分注释的Willis循环（CoW）数据集，并使用虚拟现实（VR）技术进行拓扑感知解剖分割，解决了手动和耗时的CoW表征问题。

    

    Willis循环（CoW）是连接大脑主要循环的重要动脉网络。其血管结构被认为影响着严重神经血管疾病的风险、严重程度和临床结果。然而，对高度变化的CoW解剖进行表征仍然是一项需要手动和耗时的专家任务。CoW通常通过两种血管造影成像模式进行成像，即磁共振血管成像（MRA）和计算机断层血管造影（CTA），但是关于CTA的CoW解剖的公共数据集及其注释非常有限。因此，我们在2023年组织了TopCoW挑战赛，并发布了一个带有注释的CoW数据集。TopCoW数据集是第一个具有13种可能的CoW血管组分的体素级注释的公共数据集，通过虚拟现实（VR）技术实现。它也是第一个带有来自同一患者的成对MRA和CTA的大型数据集。TopCoW挑战将CoW表征问题形式化为多类问题。

    The Circle of Willis (CoW) is an important network of arteries connecting major circulations of the brain. Its vascular architecture is believed to affect the risk, severity, and clinical outcome of serious neuro-vascular diseases. However, characterizing the highly variable CoW anatomy is still a manual and time-consuming expert task. The CoW is usually imaged by two angiographic imaging modalities, magnetic resonance angiography (MRA) and computed tomography angiography (CTA), but there exist limited public datasets with annotations on CoW anatomy, especially for CTA. Therefore we organized the TopCoW Challenge in 2023 with the release of an annotated CoW dataset. The TopCoW dataset was the first public dataset with voxel-level annotations for thirteen possible CoW vessel components, enabled by virtual-reality (VR) technology. It was also the first large dataset with paired MRA and CTA from the same patients. TopCoW challenge formalized the CoW characterization problem as a multiclas
    

