# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Low-Dose CT Image Reconstruction by Fine-Tuning a UNet Pretrained for Gaussian Denoising for the Downstream Task of Image Enhancement](https://arxiv.org/abs/2403.03551) | 提出了一种通过精调UNet进行低剂量CT图像重建的方法，其中第二阶段的训练策略为CT图像增强阶段。 |
| [^2] | [Feudal Networks for Visual Navigation](https://arxiv.org/abs/2402.12498) | 使用封建学习的视觉导航，通过高级管理者、中级管理者和工作代理的分层结构，在不同空间和时间尺度上操作，具有独特模块来实现自监督学习记忆代理地图。 |
| [^3] | [Solid Waste Detection in Remote Sensing Images: A Survey](https://arxiv.org/abs/2402.09066) | 本文调查了固体废物在遥感图像中的检测方法。研究者利用地球观测卫星提供的高分辨率数据，通过遥感图像实现了固体废物处置场地的识别、监测和评估。 |
| [^4] | [Pretraining Vision-Language Model for Difference Visual Question Answering in Longitudinal Chest X-rays](https://arxiv.org/abs/2402.08966) | 提出了一种名为PLURAL的预训练视觉-语言模型，用于纵向胸部X射线图中差异视觉问答任务。该模型通过在自然图像和文本上进行预训练，然后使用纵向胸部X射线数据进行训练，从而提高了模型的性能。 |
| [^5] | [Catch-Up Distillation: You Only Need to Train Once for Accelerating Sampling.](http://arxiv.org/abs/2305.10769) | 本文提出了一种名为“追赶蒸馏”的方法，通过调整传统采样算法，让速度估计模型的当前时刻输出与其先前时刻输出和地面真实标签对齐，从而实现只需一次训练便能加速采样的效果。 |
| [^6] | [Towards the Characterization of Representations Learned via Capsule-based Network Architectures.](http://arxiv.org/abs/2305.05349) | 本研究旨在评估胶囊网络架构学习的表示方法及其可解释性，发现其编码的表示可能与部分-整体关系并不严格相关。 |

# 详细

[^1]: 通过微调预先为高斯降噪而训练的UNet进行低剂量CT图像重建，用于图像增强的下游任务

    Low-Dose CT Image Reconstruction by Fine-Tuning a UNet Pretrained for Gaussian Denoising for the Downstream Task of Image Enhancement

    [https://arxiv.org/abs/2403.03551](https://arxiv.org/abs/2403.03551)

    提出了一种通过精调UNet进行低剂量CT图像重建的方法，其中第二阶段的训练策略为CT图像增强阶段。

    

    计算机断层扫描（CT）是一种广泛使用的医学成像模态，由于其基于电离辐射，因此希望尽量减少辐射剂量。然而，降低辐射剂量会导致图像质量下降，从低剂量CT（LDCT）数据重建仍然是一个具有挑战性的任务，值得进行研究。根据LoDoPaB-CT基准，许多最先进的方法使用涉及UNet型架构的流程。具体来说，排名第一的方法ItNet使用包括滤波反投影（FBP）、在CT数据上训练的UNet和迭代细化步骤的三阶段流程。在本文中，我们提出了一种更简单的两阶段方法。第一阶段也使用了FBP，而新颖之处在于第二阶段的训练策略，特点是CT图像增强阶段。我们方法的关键点在于神经网络是预训练的。

    arXiv:2403.03551v1 Announce Type: cross  Abstract: Computed Tomography (CT) is a widely used medical imaging modality, and as it is based on ionizing radiation, it is desirable to minimize the radiation dose. However, a reduced radiation dose comes with reduced image quality, and reconstruction from low-dose CT (LDCT) data is still a challenging task which is subject to research. According to the LoDoPaB-CT benchmark, a benchmark for LDCT reconstruction, many state-of-the-art methods use pipelines involving UNet-type architectures. Specifically the top ranking method, ItNet, employs a three-stage process involving filtered backprojection (FBP), a UNet trained on CT data, and an iterative refinement step. In this paper, we propose a less complex two-stage method. The first stage also employs FBP, while the novelty lies in the training strategy for the second stage, characterized as the CT image enhancement stage. The crucial point of our approach is that the neural network is pretrained
    
[^2]: 封建网络用于视觉导航

    Feudal Networks for Visual Navigation

    [https://arxiv.org/abs/2402.12498](https://arxiv.org/abs/2402.12498)

    使用封建学习的视觉导航，通过高级管理者、中级管理者和工作代理的分层结构，在不同空间和时间尺度上操作，具有独特模块来实现自监督学习记忆代理地图。

    

    视觉导航遵循人类可以在没有详细地图的情况下导航的直觉。一种常见方法是在建立包含可用于规划的图像节点的拓扑图的同时进行交互式探索。最近的变体从被动视频中学习，并可以利用复杂的社交和语义线索进行导航。然而，需要大量的训练视频，利用大型图并且由于使用了里程计，场景不是未知的。我们引入了一种使用封建学习的视觉导航的新方法，该方法采用了由工作代理、中级管理者和高级管理者组成的分层结构。封建学习范式的关键在于，每个级别的代理看到任务的不同方面，并且在不同的空间和时间尺度上运作。在此框架中开发了两个独特的模块。对于高级管理者，我们自监督地学习一个记忆代理地图以记录

    arXiv:2402.12498v1 Announce Type: cross  Abstract: Visual navigation follows the intuition that humans can navigate without detailed maps. A common approach is interactive exploration while building a topological graph with images at nodes that can be used for planning. Recent variations learn from passive videos and can navigate using complex social and semantic cues. However, a significant number of training videos are needed, large graphs are utilized, and scenes are not unseen since odometry is utilized. We introduce a new approach to visual navigation using feudal learning, which employs a hierarchical structure consisting of a worker agent, a mid-level manager, and a high-level manager. Key to the feudal learning paradigm, agents at each level see a different aspect of the task and operate at different spatial and temporal scales. Two unique modules are developed in this framework. For the high- level manager, we learn a memory proxy map in a self supervised manner to record prio
    
[^3]: 遥感图像中的固体废物检测：一项调查

    Solid Waste Detection in Remote Sensing Images: A Survey

    [https://arxiv.org/abs/2402.09066](https://arxiv.org/abs/2402.09066)

    本文调查了固体废物在遥感图像中的检测方法。研究者利用地球观测卫星提供的高分辨率数据，通过遥感图像实现了固体废物处置场地的识别、监测和评估。

    

    识别和表征非法固体废物处置场地对环境保护至关重要，特别是应对污染和健康危害。不当管理的垃圾填埋场通过雨水渗透污染土壤和地下水，对动物和人类构成威胁。传统的填埋场辨识方法，如现场检查，耗时且昂贵。遥感技术是用于识别和监测固体废物处置场地的一种经济有效的解决方案，可以实现广泛覆盖和多次获取。地球观测（EO）卫星配备了一系列传感器和成像能力，几十年来一直提供高分辨率的数据。研究人员提出了专门的技术，利用遥感图像执行一系列任务，如废物场地检测、倾倒场监测和适宜位置评估。

    arXiv:2402.09066v1 Announce Type: cross Abstract: The detection and characterization of illegal solid waste disposal sites are essential for environmental protection, particularly for mitigating pollution and health hazards. Improperly managed landfills contaminate soil and groundwater via rainwater infiltration, posing threats to both animals and humans. Traditional landfill identification approaches, such as on-site inspections, are time-consuming and expensive. Remote sensing is a cost-effective solution for the identification and monitoring of solid waste disposal sites that enables broad coverage and repeated acquisitions over time. Earth Observation (EO) satellites, equipped with an array of sensors and imaging capabilities, have been providing high-resolution data for several decades. Researchers proposed specialized techniques that leverage remote sensing imagery to perform a range of tasks such as waste site detection, dumping site monitoring, and assessment of suitable locati
    
[^4]: 用于纵向胸部X射线图中差异视觉问答的预训练视觉-语言模型

    Pretraining Vision-Language Model for Difference Visual Question Answering in Longitudinal Chest X-rays

    [https://arxiv.org/abs/2402.08966](https://arxiv.org/abs/2402.08966)

    提出了一种名为PLURAL的预训练视觉-语言模型，用于纵向胸部X射线图中差异视觉问答任务。该模型通过在自然图像和文本上进行预训练，然后使用纵向胸部X射线数据进行训练，从而提高了模型的性能。

    

    差异视觉问答(diff-VQA)是一个挑战性的任务，要求根据一对图像的差异回答复杂的问题。在读取胸部X射线图像中尤为重要，因为放射科医生通常会对同一患者在不同时间拍摄的多幅图像进行比较，以追踪疾病的进展和其临床实践中严重程度的变化。然而，之前的研究集中在为差异视觉问答任务设计特定的网络架构，错过了利用预训练的视觉-语言模型(VLM)提高模型性能的机会。在这里，我们介绍了一种名为PLURAL的新型VLM，它在自然图像和纵向胸部X射线数据上进行了差异视觉问答任务的预训练。该模型采用逐步的方法开发，从在自然图像和文本上进行预训练开始，然后使用纵向胸部X射线数据进行训练。纵向数据包括...

    arXiv:2402.08966v1 Announce Type: cross Abstract: Difference visual question answering (diff-VQA) is a challenging task that requires answering complex questions based on differences between a pair of images. This task is particularly important in reading chest X-ray images because radiologists often compare multiple images of the same patient taken at different times to track disease progression and changes in its severity in their clinical practice. However, previous works focused on designing specific network architectures for the diff-VQA task, missing opportunities to enhance the model's performance using a pretrained vision-language model (VLM). Here, we introduce a novel VLM called PLURAL, which is pretrained on natural and longitudinal chest X-ray data for the diff-VQA task. The model is developed using a step-by-step approach, starting with being pretrained on natural images and texts, followed by being trained using longitudinal chest X-ray data. The longitudinal data consist
    
[^5]: 追赶蒸馏：加速采样只需一次训练

    Catch-Up Distillation: You Only Need to Train Once for Accelerating Sampling. (arXiv:2305.10769v1 [cs.LG])

    [http://arxiv.org/abs/2305.10769](http://arxiv.org/abs/2305.10769)

    本文提出了一种名为“追赶蒸馏”的方法，通过调整传统采样算法，让速度估计模型的当前时刻输出与其先前时刻输出和地面真实标签对齐，从而实现只需一次训练便能加速采样的效果。

    

    扩散概率模型在各种机器学习领域取得了令人瞩目的进展。然而，为了实现高质量的合成样本，通常需要执行大量的采样步骤，这阻碍了实时样本合成的可能性。传统的通过知识蒸馏加速采样的算法依赖于预训练的模型权重和离散时间步骤场景，需要额外的培训课程才能实现他们的目标。为了解决这些问题，我们提出了追赶蒸馏（CUD），它鼓励速度估计模型的当前时刻输出“追赶”其先前时刻输出。具体而言，CUD调整了原始的常微分方程（ODE）训练目标，以使当前时刻输出与地面真实标签和先前时刻输出对齐，利用基于龙格-库塔的多步对齐蒸馏进行精确的ODE估计，同时防止异步更新。

    Diffusion Probability Models (DPMs) have made impressive advancements in various machine learning domains. However, achieving high-quality synthetic samples typically involves performing a large number of sampling steps, which impedes the possibility of real-time sample synthesis. Traditional accelerated sampling algorithms via knowledge distillation rely on pre-trained model weights and discrete time step scenarios, necessitating additional training sessions to achieve their goals. To address these issues, we propose the Catch-Up Distillation (CUD), which encourages the current moment output of the velocity estimation model ``catch up'' with its previous moment output. Specifically, CUD adjusts the original Ordinary Differential Equation (ODE) training objective to align the current moment output with both the ground truth label and the previous moment output, utilizing Runge-Kutta-based multi-step alignment distillation for precise ODE estimation while preventing asynchronous updates
    
[^6]: 旨在表征基于胶囊网络架构学习的表示方法

    Towards the Characterization of Representations Learned via Capsule-based Network Architectures. (arXiv:2305.05349v1 [cs.LG])

    [http://arxiv.org/abs/2305.05349](http://arxiv.org/abs/2305.05349)

    本研究旨在评估胶囊网络架构学习的表示方法及其可解释性，发现其编码的表示可能与部分-整体关系并不严格相关。

    

    胶囊网络作为标准深度神经网络的一种更为紧凑和可解释的替代方法而重新引入。尽管最近的研究证明了其压缩能力，但至今尚未完全评估其可解释性质。在这里，我们进行了一项系统而原则性的研究，以评估这种类型网络的可解释性。此外，我们特别注意分析所学到的表示中是否确实编码了部分-整体关系的水平。在MNIST、SVHN、PASCAL-part和CelebA数据集中的分析表明，在CapsNets中编码的表示可能既不像文献中通常所述的那样分离，也不是严格与部分-整体关系相关的。

    Capsule Networks (CapsNets) have been re-introduced as a more compact and interpretable alternative to standard deep neural networks. While recent efforts have proved their compression capabilities, to date, their interpretability properties have not been fully assessed. Here, we conduct a systematic and principled study towards assessing the interpretability of these types of networks. Moreover, we pay special attention towards analyzing the level to which part-whole relationships are indeed encoded within the learned representation. Our analysis in the MNIST, SVHN, PASCAL-part and CelebA datasets suggest that the representations encoded in CapsNets might not be as disentangled nor strictly related to parts-whole relationships as is commonly stated in the literature.
    

