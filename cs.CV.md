# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Revolutionizing Disease Diagnosis with simultaneous functional PET/MR and Deeply Integrated Brain Metabolic, Hemodynamic, and Perfusion Networks](https://arxiv.org/abs/2403.20058) | 提出了MX-ARM，一种基于AI的疾病诊断模型，利用同时功能PET/MR技术，能够在推理过程中同时接受单模态和多模态输入，具有创新的模态分离和重构功能。 |
| [^2] | [NIGHT -- Non-Line-of-Sight Imaging from Indirect Time of Flight Data](https://arxiv.org/abs/2403.19376) | 本文首次使用来自即插即用的间接飞行时间传感器的数据，引入了一个深度学习模型，能够将光线反射发生的表面重新构建为虚拟镜子，从而实现了获取隐藏场景深度信息的可行性。 |
| [^3] | [Neural Slot Interpreters: Grounding Object Semantics in Emergent Slot Representations](https://arxiv.org/abs/2403.07887) | 提出了神经槽解释器（NSI），通过槽表示学习接地和生成物体语义，实现了将现实世界的物体语义结合到抽象中。 |
| [^4] | [Entity-Aware Multimodal Alignment Framework for News Image Captioning](https://arxiv.org/abs/2402.19404) | 设计了面向实体的多模态对齐任务和对齐框架，提高了新闻图像字幕生成任务的性能表现。 |
| [^5] | [Disentangling representations of retinal images with generative models](https://arxiv.org/abs/2402.19186) | 引入一种新颖的视网膜底图像群体模型，有效解开患者属性与相机效果，实现可控且高度逼真的图像生成。 |
| [^6] | [Bridging the Projection Gap: Overcoming Projection Bias Through Parameterized Distance Learning](https://arxiv.org/abs/2309.01390) | 通过学习参数化的马氏距离度量，解决广义零样本学习中的投影偏差问题，提出了扩展VAEGAN架构和引入新损失函数以实现更稳健的距离学习 |
| [^7] | [High Perceptual Quality Wireless Image Delivery with Denoising Diffusion Models.](http://arxiv.org/abs/2309.15889) | 本论文研究了通过深度学习的联合源-信道编码和去噪扩散模型在噪声无线信道上进行图像传输的问题。通过利用范围-零空间分解和逐步优化零空间内容，实现了在失真和感知质量方面的显著改进。 |
| [^8] | [Alternative Telescopic Displacement: An Efficient Multimodal Alignment Method.](http://arxiv.org/abs/2306.16950) | 备选的变焦位移是一种高效的多模态对齐方法，通过交替移动和扩展特征信息来融合多模态数据，可以稳健地捕捉不同模态特征之间的高级交互作用，从而显著提高多模态学习的性能，并在多个任务上优于其他流行的多模态方案。 |
| [^9] | [Deep Single Image Camera Calibration by Heatmap Regression to Recover Fisheye Images Under ManhattanWorld AssumptionWithout Ambiguity.](http://arxiv.org/abs/2303.17166) | 本文提出一种基于学习的标定方法，使用热度图回归来消除曼哈顿世界假设下鱼眼图片中横向角度歧义，同时恢复旋转和消除鱼眼失真。该方法使用优化的对角线点缓解图像中缺乏消失点的情况，并在实验证明其性能优于现有技术。 |

# 详细

[^1]: 利用同时功能PET/MR和深度整合的脑代谢、血液动力学和灌注网络彻底改变疾病诊断

    Revolutionizing Disease Diagnosis with simultaneous functional PET/MR and Deeply Integrated Brain Metabolic, Hemodynamic, and Perfusion Networks

    [https://arxiv.org/abs/2403.20058](https://arxiv.org/abs/2403.20058)

    提出了MX-ARM，一种基于AI的疾病诊断模型，利用同时功能PET/MR技术，能够在推理过程中同时接受单模态和多模态输入，具有创新的模态分离和重构功能。

    

    同时功能PET/MR（sf-PET/MR）是一种尖端的多模式神经影像技术。它提供了一个前所未有的机会，可以同时监测和整合由时空协变代谢活动、神经活动和脑血流（灌注）构建的多方面大脑网络。虽然在科学/临床价值上很高，但PET/MR硬件的可及性不足阻碍了其应用，更不用说现代基于AI的PET/MR融合模型。我们的目标是开发一个基于AI的临床可行疾病诊断模型，该模型基于全面的sf-PET/MR数据进行训练，在推理过程中具有允许单模态输入（例如，仅PET）以及强制多模态准确性的能力。为此，我们提出了MX-ARM，一种多模态专家混合对齐和重构模型。它是模态可分离和可交换的，动态分配不同的多层感知器（"混合）

    arXiv:2403.20058v1 Announce Type: cross  Abstract: Simultaneous functional PET/MR (sf-PET/MR) presents a cutting-edge multimodal neuroimaging technique. It provides an unprecedented opportunity for concurrently monitoring and integrating multifaceted brain networks built by spatiotemporally covaried metabolic activity, neural activity, and cerebral blood flow (perfusion). Albeit high scientific/clinical values, short in hardware accessibility of PET/MR hinders its applications, let alone modern AI-based PET/MR fusion models. Our objective is to develop a clinically feasible AI-based disease diagnosis model trained on comprehensive sf-PET/MR data with the power of, during inferencing, allowing single modality input (e.g., PET only) as well as enforcing multimodal-based accuracy. To this end, we propose MX-ARM, a multimodal MiXture-of-experts Alignment and Reconstruction Model. It is modality detachable and exchangeable, allocating different multi-layer perceptrons dynamically ("mixture 
    
[^2]: NIGHT -- 间接飞行时间数据的非视距成像

    NIGHT -- Non-Line-of-Sight Imaging from Indirect Time of Flight Data

    [https://arxiv.org/abs/2403.19376](https://arxiv.org/abs/2403.19376)

    本文首次使用来自即插即用的间接飞行时间传感器的数据，引入了一个深度学习模型，能够将光线反射发生的表面重新构建为虚拟镜子，从而实现了获取隐藏场景深度信息的可行性。

    

    从非视角相机外部获取物体是一个非常引人注目但也极具挑战性的研究课题。最近的工作表明，利用定制的直接飞行时间传感器产生的瞬时成像数据，这个想法是可行的。在本文中，我们首次使用来自即插即用的间接飞行时间传感器的数据来解决这个问题，而不需要任何额外的硬件要求。我们引入了一个深度学习模型，能够将光线反射发生的表面重新构建为虚拟镜子。这种建模使得任务更容易处理，也有助于构建带有注释的训练数据。从获得的数据中，可以恢复隐藏场景的深度信息。我们还提供了一个首创的合成数据集用于这个任务，并展示了所提出的想法的可行性。

    arXiv:2403.19376v1 Announce Type: cross  Abstract: The acquisition of objects outside the Line-of-Sight of cameras is a very intriguing but also extremely challenging research topic. Recent works showed the feasibility of this idea exploiting transient imaging data produced by custom direct Time of Flight sensors. In this paper, for the first time, we tackle this problem using only data from an off-the-shelf indirect Time of Flight sensor without any further hardware requirement. We introduced a Deep Learning model able to re-frame the surfaces where light bounces happen as a virtual mirror. This modeling makes the task easier to handle and also facilitates the construction of annotated training data. From the obtained data it is possible to retrieve the depth information of the hidden scene. We also provide a first-in-its-kind synthetic dataset for the task and demonstrate the feasibility of the proposed idea over it.
    
[^3]: 神经槽解释器：在新兴的槽表示中接地对象语义

    Neural Slot Interpreters: Grounding Object Semantics in Emergent Slot Representations

    [https://arxiv.org/abs/2403.07887](https://arxiv.org/abs/2403.07887)

    提出了神经槽解释器（NSI），通过槽表示学习接地和生成物体语义，实现了将现实世界的物体语义结合到抽象中。

    

    物体中心方法在将原始感知无监督分解为丰富的类似物体的抽象方面取得了重大进展。然而，将现实世界的物体语义接地到学到的抽象中的能力有限，这阻碍了它们在下游理解应用中的采用。我们提出神经槽解释器（NSI），它通过槽表示学习接地和生成物体语义。NSI的核心是一种类似XML的编程语言，它使用简单的语法规则将场景的物体语义组织成以物体为中心的程序原语。然后，一个对齐模型学习通过共享嵌入空间上的双层对比学习目标将程序原语接地到槽。最后，我们构建NSI程序生成模型，利用对齐模型推断的密集关联从槽生成以物体为中心的程序。在双模式检索实验中，

    arXiv:2403.07887v1 Announce Type: cross  Abstract: Object-centric methods have seen significant progress in unsupervised decomposition of raw perception into rich object-like abstractions. However, limited ability to ground object semantics of the real world into the learned abstractions has hindered their adoption in downstream understanding applications. We present the Neural Slot Interpreter (NSI) that learns to ground and generate object semantics via slot representations. At the core of NSI is an XML-like programming language that uses simple syntax rules to organize the object semantics of a scene into object-centric program primitives. Then, an alignment model learns to ground program primitives into slots through a bi-level contrastive learning objective over a shared embedding space. Finally, we formulate the NSI program generator model to use the dense associations inferred from the alignment model to generate object-centric programs from slots. Experiments on bi-modal retrie
    
[^4]: 面向实体的多模态对齐框架用于新闻图像字幕生成

    Entity-Aware Multimodal Alignment Framework for News Image Captioning

    [https://arxiv.org/abs/2402.19404](https://arxiv.org/abs/2402.19404)

    设计了面向实体的多模态对齐任务和对齐框架，提高了新闻图像字幕生成任务的性能表现。

    

    新闻图像字幕生成任务是图像字幕生成任务的一个变体，要求模型生成一个更具信息性的字幕，其中包含新闻图像和相关新闻文章。近年来，多模态大型语言模型发展迅速，并在新闻图像字幕生成任务中表现出前景。然而，根据我们的实验，常见的多模态大型语言模型在零样本设定下生成实体方面表现不佳。即使在新闻图像字幕生成数据集上进行简单微调，它们处理实体信息的能力仍然有限。为了获得一个更强大的模型来处理多模态实体信息，我们设计了两个多模态实体感知对齐任务和一个对齐框架，以对齐模型并生成新闻图像字幕。我们的方法在GoodNews数据集上将CIDEr分数提高到86.29（从72.33），在NYTimes800k数据集上将其提高到85.61（从70.83），优于先前的最先进模型。

    arXiv:2402.19404v1 Announce Type: cross  Abstract: News image captioning task is a variant of image captioning task which requires model to generate a more informative caption with news image and the associated news article. Multimodal Large Language models have developed rapidly in recent years and is promising in news image captioning task. However, according to our experiments, common MLLMs are not good at generating the entities in zero-shot setting. Their abilities to deal with the entities information are still limited after simply fine-tuned on news image captioning dataset. To obtain a more powerful model to handle the multimodal entity information, we design two multimodal entity-aware alignment tasks and an alignment framework to align the model and generate the news image captions. Our method achieves better results than previous state-of-the-art models in CIDEr score (72.33 -> 86.29) on GoodNews dataset and (70.83 -> 85.61) on NYTimes800k dataset.
    
[^5]: 用生成模型解开视网膜图像的表征

    Disentangling representations of retinal images with generative models

    [https://arxiv.org/abs/2402.19186](https://arxiv.org/abs/2402.19186)

    引入一种新颖的视网膜底图像群体模型，有效解开患者属性与相机效果，实现可控且高度逼真的图像生成。

    

    视网膜底图像在早期检测眼部疾病中起着至关重要的作用，最近的研究甚至表明，利用深度学习方法，这些图像还可以用于检测心血管风险因素和神经系统疾病。然而，这些图像受技术因素的影响可能对眼科领域可靠的人工智能应用构成挑战。例如，大型底图队列往往受到相机类型、图像质量或照明水平等因素的影响，存在学习快捷方式而不是图像生成过程背后因果关系的风险。在这里，我们提出了一个新颖的视网膜底图像群体模型，有效地解开了患者属性与相机效果，从而实现了可控且高度逼真的图像生成。为了实现这一目标，我们提出了一个基于距离相关性的新颖解开损失。通过定性和定量分析，我们展示了...

    arXiv:2402.19186v1 Announce Type: cross  Abstract: Retinal fundus images play a crucial role in the early detection of eye diseases and, using deep learning approaches, recent studies have even demonstrated their potential for detecting cardiovascular risk factors and neurological disorders. However, the impact of technical factors on these images can pose challenges for reliable AI applications in ophthalmology. For example, large fundus cohorts are often confounded by factors like camera type, image quality or illumination level, bearing the risk of learning shortcuts rather than the causal relationships behind the image generation process. Here, we introduce a novel population model for retinal fundus images that effectively disentangles patient attributes from camera effects, thus enabling controllable and highly realistic image generation. To achieve this, we propose a novel disentanglement loss based on distance correlation. Through qualitative and quantitative analyses, we demon
    
[^6]: 弥合投影差距：通过参数化距离学习克服投影偏差

    Bridging the Projection Gap: Overcoming Projection Bias Through Parameterized Distance Learning

    [https://arxiv.org/abs/2309.01390](https://arxiv.org/abs/2309.01390)

    通过学习参数化的马氏距离度量，解决广义零样本学习中的投影偏差问题，提出了扩展VAEGAN架构和引入新损失函数以实现更稳健的距离学习

    

    广义零样本学习（GZSL）旨在仅利用已知类别样本训练来识别来自已知和未知类别的样本。然而，在推断过程中，由于投影函数是从已知类别中学习的，GZSL方法很容易偏向已知类别。大多数方法致力于学习准确的投影，但投影中的偏差是不可避免的。我们通过提出学习参数化的马氏距离度量来解决该投影偏差，关键洞察是尽管投影存在偏差，但在推断过程中距离计算至关重要。我们作出两个主要贡献 - (1)我们通过增加两个分支扩展了VAEGAN（变分自动编码器和生成对抗网络）架构，分别输出来自已知和未知类别的样本的投影，从而实现更稳健的距离学习。 (2)我们引入了一种新颖的损失函数来优化马氏距离

    arXiv:2309.01390v2 Announce Type: replace-cross  Abstract: Generalized zero-shot learning (GZSL) aims to recognize samples from both seen and unseen classes using only seen class samples for training. However, GZSL methods are prone to bias towards seen classes during inference due to the projection function being learned from seen classes. Most methods focus on learning an accurate projection, but bias in the projection is inevitable. We address this projection bias by proposing to learn a parameterized Mahalanobis distance metric for robust inference. Our key insight is that the distance computation during inference is critical, even with a biased projection. We make two main contributions - (1) We extend the VAEGAN (Variational Autoencoder \& Generative Adversarial Networks) architecture with two branches to separately output the projection of samples from seen and unseen classes, enabling more robust distance learning. (2) We introduce a novel loss function to optimize the Mahalano
    
[^7]: 使用去噪扩散模型实现高感知质量的无线图像传输

    High Perceptual Quality Wireless Image Delivery with Denoising Diffusion Models. (arXiv:2309.15889v1 [eess.IV])

    [http://arxiv.org/abs/2309.15889](http://arxiv.org/abs/2309.15889)

    本论文研究了通过深度学习的联合源-信道编码和去噪扩散模型在噪声无线信道上进行图像传输的问题。通过利用范围-零空间分解和逐步优化零空间内容，实现了在失真和感知质量方面的显著改进。

    

    我们考虑通过基于深度学习的联合源-信道编码（DeepJSCC）以及接收端的去噪扩散概率模型（DDPM）在噪声无线信道上进行图像传输。我们特别关注在实际有限块长度的情况下的感知失真权衡问题，这种情况下，分离的源编码和信道编码可能会高度不理想。我们引入了一种利用目标图像的范围-零空间分解的新方案。我们在编码后传输图像的范围空间，并使用DDPM逐步优化其零空间内容。通过广泛的实验证明，与标准的DeepJSCC和最先进的生成式学习方法相比，我们在重构图像的失真和感知质量方面实现了显著改进。为了促进进一步的研究和可重现性，我们将公开分享我们的源代码。

    We consider the image transmission problem over a noisy wireless channel via deep learning-based joint source-channel coding (DeepJSCC) along with a denoising diffusion probabilistic model (DDPM) at the receiver. Specifically, we are interested in the perception-distortion trade-off in the practical finite block length regime, in which separate source and channel coding can be highly suboptimal. We introduce a novel scheme that utilizes the range-null space decomposition of the target image. We transmit the range-space of the image after encoding and employ DDPM to progressively refine its null space contents. Through extensive experiments, we demonstrate significant improvements in distortion and perceptual quality of reconstructed images compared to standard DeepJSCC and the state-of-the-art generative learning-based method. We will publicly share our source code to facilitate further research and reproducibility.
    
[^8]: 备选的变焦位移：一种高效的多模态对齐方法

    Alternative Telescopic Displacement: An Efficient Multimodal Alignment Method. (arXiv:2306.16950v1 [cs.CV])

    [http://arxiv.org/abs/2306.16950](http://arxiv.org/abs/2306.16950)

    备选的变焦位移是一种高效的多模态对齐方法，通过交替移动和扩展特征信息来融合多模态数据，可以稳健地捕捉不同模态特征之间的高级交互作用，从而显著提高多模态学习的性能，并在多个任务上优于其他流行的多模态方案。

    

    特征对齐是融合多模态数据的主要方式。我们提出了一种特征对齐方法，可以完全融合多模态信息，通过在特征空间中交替移动和扩展来实现不同模态之间的一致表示。所提出的方法能够稳健地捕捉不同模态特征之间的高级交互作用，从而显著提高多模态学习的性能。我们还表明，所提出的方法在多个任务上优于其他流行的多模态方案。对ETT和MIT-BIH-Arrhythmia数据集的实验评估表明，所提出的方法达到了最先进的性能。

    Feature alignment is the primary means of fusing multimodal data. We propose a feature alignment method that fully fuses multimodal information, which alternately shifts and expands feature information from different modalities to have a consistent representation in a feature space. The proposed method can robustly capture high-level interactions between features of different modalities, thus significantly improving the performance of multimodal learning. We also show that the proposed method outperforms other popular multimodal schemes on multiple tasks. Experimental evaluation of ETT and MIT-BIH-Arrhythmia, datasets shows that the proposed method achieves state of the art performance.
    
[^9]: 使用热度图回归进行深度单张图片摄像机标定，在曼哈顿世界假设下在不模糊的情况下还原鱼眼图片

    Deep Single Image Camera Calibration by Heatmap Regression to Recover Fisheye Images Under ManhattanWorld AssumptionWithout Ambiguity. (arXiv:2303.17166v1 [cs.CV])

    [http://arxiv.org/abs/2303.17166](http://arxiv.org/abs/2303.17166)

    本文提出一种基于学习的标定方法，使用热度图回归来消除曼哈顿世界假设下鱼眼图片中横向角度歧义，同时恢复旋转和消除鱼眼失真。该方法使用优化的对角线点缓解图像中缺乏消失点的情况，并在实验证明其性能优于现有技术。

    

    在正交世界坐标系中，曼哈顿世界沿着长方体建筑物广泛用于各种计算机视觉任务。然而，曼哈顿世界需要改进，因为图像中的横向角度的原点是任意的，即具有四倍轮换对称的横向角度的歧义。为了解决这个问题，我们提出了一个基于摄像机和行驶方向的道路方向的平角定义。我们提出了一种基于学习的标定方法，它使用热度图回归来消除歧义，类似于姿态估计关键点。与此同时，我们的两个分支网络恢复旋转并从一般场景图像中消除鱼眼失真。为了缓解图像中缺乏消失点的情况，我们引入了具有空间均匀性最佳的对角线点。大量实验证明，我们的方法在曼哈顿世界假设下对鱼眼图像的深度单张图片摄像机标定优于现有技术，没有歧义。

    In orthogonal world coordinates, a Manhattan world lying along cuboid buildings is widely useful for various computer vision tasks. However, the Manhattan world has much room for improvement because the origin of pan angles from an image is arbitrary, that is, four-fold rotational symmetric ambiguity of pan angles. To address this problem, we propose a definition for the pan-angle origin based on the directions of the roads with respect to a camera and the direction of travel. We propose a learning-based calibration method that uses heatmap regression to remove the ambiguity by each direction of labeled image coordinates, similar to pose estimation keypoints. Simultaneously, our two-branched network recovers the rotation and removes fisheye distortion from a general scene image. To alleviate the lack of vanishing points in images, we introduce auxiliary diagonal points that have the optimal 3D arrangement of spatial uniformity. Extensive experiments demonstrated that our method outperf
    

