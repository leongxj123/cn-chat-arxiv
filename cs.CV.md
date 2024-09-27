# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ICON: Improving Inter-Report Consistency of Radiology Report Generation via Lesion-aware Mix-up Augmentation](https://arxiv.org/abs/2402.12844) | 本文提出的ICON方法旨在通过改善放射学报告生成的报告间一致性，提升系统捕捉语义等效病变相似性的能力。 |
| [^2] | [High-Quality Image Restoration Following Human Instructions.](http://arxiv.org/abs/2401.16468) | 本论文提出了一种使用人类编写的指令来指导图像恢复模型的方法，并在多个恢复任务上取得了最先进的结果，为基于文本指导的图像恢复和增强研究提供了一个新的基准。 |
| [^3] | [Enhanced Distribution Alignment for Post-Training Quantization of Diffusion Models.](http://arxiv.org/abs/2401.04585) | 本文提出了一种扩展分布对齐方法以解决后训练量化对于弥散模型的分布不匹配问题，该方法在低延迟应用中具有较高的潜力，并且能有效提升性能。 |
| [^4] | [Fast Inference Through The Reuse Of Attention Maps In Diffusion Models.](http://arxiv.org/abs/2401.01008) | 本文提出了一种无需训练的方法，通过重用注意力映射来实现Text-to-image diffusion models中的快速推理，以提高效率。 |
| [^5] | [EPTQ: Enhanced Post-Training Quantization via Label-Free Hessian.](http://arxiv.org/abs/2309.11531) | 本文提出了一种名为EPTQ的增强后训练量化方法，该方法通过自适应加权层和无标签Hessian近似技术实现了最先进的结果。 |
| [^6] | [Alternative Telescopic Displacement: An Efficient Multimodal Alignment Method.](http://arxiv.org/abs/2306.16950) | 备选的变焦位移是一种高效的多模态对齐方法，通过交替移动和扩展特征信息来融合多模态数据，可以稳健地捕捉不同模态特征之间的高级交互作用，从而显著提高多模态学习的性能，并在多个任务上优于其他流行的多模态方案。 |

# 详细

[^1]: ICON：通过病变感知混合增强改善放射学报告生成的报告间一致性

    ICON: Improving Inter-Report Consistency of Radiology Report Generation via Lesion-aware Mix-up Augmentation

    [https://arxiv.org/abs/2402.12844](https://arxiv.org/abs/2402.12844)

    本文提出的ICON方法旨在通过改善放射学报告生成的报告间一致性，提升系统捕捉语义等效病变相似性的能力。

    

    放射学报告生成的先前研究在增加生成报告的临床准确性方面取得了显著进展。本文强调了其应具备的另一个至关重要的特质，即报告间一致性，指的是对语义上等效的X射线照片生成一致性报告的能力。ICON提出了一种方法，它通过改善放射学报告生成的报告间一致性来解决这一问题。

    arXiv:2402.12844v1 Announce Type: cross  Abstract: Previous research on radiology report generation has made significant progress in terms of increasing the clinical accuracy of generated reports. In this paper, we emphasize another crucial quality that it should possess, i.e., inter-report consistency, which refers to the capability of generating consistent reports for semantically equivalent radiographs. This quality is even of greater significance than the overall report accuracy in terms of ensuring the system's credibility, as a system prone to providing conflicting results would severely erode users' trust. Regrettably, existing approaches struggle to maintain inter-report consistency, exhibiting biases towards common patterns and susceptibility to lesion variants. To address this issue, we propose ICON, which improves the inter-report consistency of radiology report generation. Aiming at enhancing the system's ability to capture the similarities in semantically equivalent lesion
    
[^2]: 遵循人类指令的高质量图像恢复

    High-Quality Image Restoration Following Human Instructions. (arXiv:2401.16468v1 [cs.CV])

    [http://arxiv.org/abs/2401.16468](http://arxiv.org/abs/2401.16468)

    本论文提出了一种使用人类编写的指令来指导图像恢复模型的方法，并在多个恢复任务上取得了最先进的结果，为基于文本指导的图像恢复和增强研究提供了一个新的基准。

    

    图像恢复是一个基本问题，涉及从退化观测中恢复出高质量的干净图像。全能图像恢复模型可以通过使用特定于退化类型的信息作为提示来有效地恢复各种类型和级别的退化图像，并引导恢复模型。我们提出了一种使用人类编写的指令来指导图像恢复模型的方法。在给定自然语言提示的情况下，我们的模型可以从退化图像中恢复出高质量的图像，并考虑多种退化类型。我们的方法InstructIR在图像去噪、雨水去除、去模糊、去雾和(低光)图像增强等多个恢复任务上取得了最先进的结果。InstructIR在之前的全能恢复方法上提高了1dB。此外，我们的数据集和结果为基于文本指导的图像恢复和增强的新研究提供了一个新的基准。我们提供了代码、数据集和模型。

    Image restoration is a fundamental problem that involves recovering a high-quality clean image from its degraded observation. All-In-One image restoration models can effectively restore images from various types and levels of degradation using degradation-specific information as prompts to guide the restoration model. In this work, we present the first approach that uses human-written instructions to guide the image restoration model. Given natural language prompts, our model can recover high-quality images from their degraded counterparts, considering multiple degradation types. Our method, InstructIR, achieves state-of-the-art results on several restoration tasks including image denoising, deraining, deblurring, dehazing, and (low-light) image enhancement. InstructIR improves +1dB over previous all-in-one restoration methods. Moreover, our dataset and results represent a novel benchmark for new research on text-guided image restoration and enhancement. Our code, datasets and models a
    
[^3]: 扩展分布对齐来实现弥散模型的后训练量化

    Enhanced Distribution Alignment for Post-Training Quantization of Diffusion Models. (arXiv:2401.04585v1 [cs.CV])

    [http://arxiv.org/abs/2401.04585](http://arxiv.org/abs/2401.04585)

    本文提出了一种扩展分布对齐方法以解决后训练量化对于弥散模型的分布不匹配问题，该方法在低延迟应用中具有较高的潜力，并且能有效提升性能。

    

    通过迭代噪声估计，扩散模型在图像生成任务中取得了巨大成功。然而，繁重的去噪过程和复杂的神经网络阻碍了它们在实际场景中的低延迟应用。量化可以有效降低模型复杂度，而后训练量化(PTQ)在加速去噪过程方面具有很高的潜力，并且不需要微调。不幸的是，我们发现由于不同去噪步骤中激活的高度动态分布，现有的扩散模型的PTQ方法在校准样本和重构输出两个层面上都存在分布不匹配的问题，导致性能远低于令人满意的水平，特别是在低位情况下。在本文中，我们提出了增强的分布对齐用于弥散模型的后训练量化(EDA-DM)来解决上述问题。具体来说，在校准样本层面，我们基于...[缺省]

    Diffusion models have achieved great success in image generation tasks through iterative noise estimation. However, the heavy denoising process and complex neural networks hinder their low-latency applications in real-world scenarios. Quantization can effectively reduce model complexity, and post-training quantization (PTQ), which does not require fine-tuning, is highly promising in accelerating the denoising process. Unfortunately, we find that due to the highly dynamic distribution of activations in different denoising steps, existing PTQ methods for diffusion models suffer from distribution mismatch issues at both calibration sample level and reconstruction output level, which makes the performance far from satisfactory, especially in low-bit cases. In this paper, we propose Enhanced Distribution Alignment for Post-Training Quantization of Diffusion Models (EDA-DM) to address the above issues. Specifically, at the calibration sample level, we select calibration samples based on the 
    
[^4]: Text-to-image diffusion models中通过重用注意力映射实现快速推理

    Fast Inference Through The Reuse Of Attention Maps In Diffusion Models. (arXiv:2401.01008v1 [cs.CV])

    [http://arxiv.org/abs/2401.01008](http://arxiv.org/abs/2401.01008)

    本文提出了一种无需训练的方法，通过重用注意力映射来实现Text-to-image diffusion models中的快速推理，以提高效率。

    

    文字到图像扩散模型在灵活和逼真的图像合成方面展示了前所未有的能力。然而，生成单个图像所需的迭代过程既昂贵又具有较高的延迟，促使研究人员进一步研究其效率。我们提出了一种无需调整采样步长的无需训练的方法。具体地说，我们发现重复计算注意力映射既耗时又冗余，因此我们建议在采样过程中结构化地重用注意力映射。我们的初步重用策略受到初级ODE理论的启发，该理论认为在采样过程的后期重用最合适。在注意到这种理论方法的一些局限性后，我们通过实验证明了一种更好的方法。

    Text-to-image diffusion models have demonstrated unprecedented abilities at flexible and realistic image synthesis. However, the iterative process required to produce a single image is costly and incurs a high latency, prompting researchers to further investigate its efficiency. Typically, improvements in latency have been achieved in two ways: (1) training smaller models through knowledge distillation (KD); and (2) adopting techniques from ODE-theory to facilitate larger step sizes. In contrast, we propose a training-free approach that does not alter the step-size of the sampler. Specifically, we find the repeated calculation of attention maps to be both costly and redundant; therefore, we propose a structured reuse of attention maps during sampling. Our initial reuse policy is motivated by rudimentary ODE-theory, which suggests that reuse is most suitable late in the sampling procedure. After noting a number of limitations in this theoretical approach, we empirically search for a bet
    
[^5]: EPTQ:通过无标签Hessian增强的后训练量化

    EPTQ: Enhanced Post-Training Quantization via Label-Free Hessian. (arXiv:2309.11531v1 [cs.CV])

    [http://arxiv.org/abs/2309.11531](http://arxiv.org/abs/2309.11531)

    本文提出了一种名为EPTQ的增强后训练量化方法，该方法通过自适应加权层和无标签Hessian近似技术实现了最先进的结果。

    

    深度神经网络的量化已成为将这些网络嵌入到最终用户设备上的关键要素。然而，当前的量化方法通常会导致准确性严重下降。本文提出了一种名为EPTQ的增强后训练量化方法。该方法基于知识蒸馏，并采用自适应加权层的方式。此外，我们提出了一种新的无标签Hessian近似技术，名为Label-Free Hessian。这种技术消除了计算Hessian所需的标记数据集的要求。自适应知识蒸馏利用Label-Free Hessian技术，在进行优化时更加关注模型的敏感部分。通过使用EPTQ，我们在各种模型、任务和数据集上实现了最先进的结果，包括ImageNet分类、COCO目标检测和用于语义分割的Pascal-VOC数据集。

    Quantization of deep neural networks (DNN) has become a key element in the efforts of embedding such networks on end-user devices. However, current quantization methods usually suffer from costly accuracy degradation. In this paper, we propose a new method for Enhanced Post Training Quantization named EPTQ. The method is based on knowledge distillation with an adaptive weighting of layers. In addition, we introduce a new label-free technique for approximating the Hessian trace of the task loss, named Label-Free Hessian. This technique removes the requirement of a labeled dataset for computing the Hessian. The adaptive knowledge distillation uses the Label-Free Hessian technique to give greater attention to the sensitive parts of the model while performing the optimization. Empirically, by employing EPTQ we achieve state-of-the-art results on a wide variety of models, tasks, and datasets, including ImageNet classification, COCO object detection, and Pascal-VOC for semantic segmentation.
    
[^6]: 备选的变焦位移：一种高效的多模态对齐方法

    Alternative Telescopic Displacement: An Efficient Multimodal Alignment Method. (arXiv:2306.16950v1 [cs.CV])

    [http://arxiv.org/abs/2306.16950](http://arxiv.org/abs/2306.16950)

    备选的变焦位移是一种高效的多模态对齐方法，通过交替移动和扩展特征信息来融合多模态数据，可以稳健地捕捉不同模态特征之间的高级交互作用，从而显著提高多模态学习的性能，并在多个任务上优于其他流行的多模态方案。

    

    特征对齐是融合多模态数据的主要方式。我们提出了一种特征对齐方法，可以完全融合多模态信息，通过在特征空间中交替移动和扩展来实现不同模态之间的一致表示。所提出的方法能够稳健地捕捉不同模态特征之间的高级交互作用，从而显著提高多模态学习的性能。我们还表明，所提出的方法在多个任务上优于其他流行的多模态方案。对ETT和MIT-BIH-Arrhythmia数据集的实验评估表明，所提出的方法达到了最先进的性能。

    Feature alignment is the primary means of fusing multimodal data. We propose a feature alignment method that fully fuses multimodal information, which alternately shifts and expands feature information from different modalities to have a consistent representation in a feature space. The proposed method can robustly capture high-level interactions between features of different modalities, thus significantly improving the performance of multimodal learning. We also show that the proposed method outperforms other popular multimodal schemes on multiple tasks. Experimental evaluation of ETT and MIT-BIH-Arrhythmia, datasets shows that the proposed method achieves state of the art performance.
    

