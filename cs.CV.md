# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Joint chest X-ray diagnosis and clinical visual attention prediction with multi-stage cooperative learning: enhancing interpretability](https://arxiv.org/abs/2403.16970) | 该论文引入了一种新的深度学习框架，用于联合疾病诊断和胸部X光扫描对应视觉显著性图的预测，通过设计新颖的双编码器多任务UNet并利用多尺度特征融合分类器来提高计算辅助诊断的可解释性和质量。 |
| [^2] | [Attention-based Class-Conditioned Alignment for Multi-Source Domain Adaptive Object Detection](https://arxiv.org/abs/2403.09918) | 提出了一种基于注意力的类别条件对齐方案，用于多源领域自适应目标检测，在跨领域对齐每个对象类别的实例。 |
| [^3] | [Low-Dose CT Image Reconstruction by Fine-Tuning a UNet Pretrained for Gaussian Denoising for the Downstream Task of Image Enhancement](https://arxiv.org/abs/2403.03551) | 提出了一种通过精调UNet进行低剂量CT图像重建的方法，其中第二阶段的训练策略为CT图像增强阶段。 |
| [^4] | [Flexible Physical Camouflage Generation Based on a Differential Approach](https://arxiv.org/abs/2402.13575) | 该研究引入了一种新颖的神经渲染方法，名为FPA，通过学习对抗模式并结合特殊设计的对抗损失和隐蔽约束损失，可以生成物理世界中具有对抗性和隐蔽性质的伪装。 |
| [^5] | [TorchCP: A Library for Conformal Prediction based on PyTorch](https://arxiv.org/abs/2402.12683) | TorchCP是一个基于PyTorch的Python工具包，为深度学习模型上的合拟常规预测研究提供了实现后验和训练方法的多种工具，包括分类和回归任务。En_Tdlr: TorchCP is a Python toolbox built on PyTorch for conformal prediction research on deep learning models, providing various implementations for posthoc and training methods for classification and regression tasks, including multi-dimension output. |
| [^6] | [AICAttack: Adversarial Image Captioning Attack with Attention-Based Optimization](https://arxiv.org/abs/2402.11940) | 提出了一种新的对抗攻击策略AICAttack，旨在通过微小的图像扰动来攻击图像字幕模型，在黑盒攻击情景下具有良好的效果。 |
| [^7] | [Respect the model: Fine-grained and Robust Explanation with Sharing Ratio Decomposition](https://arxiv.org/abs/2402.03348) | 本论文提出了一种称为共享比例分解(SRD)的新颖解释方法，真实地反映了模型的推理过程，并在解释方面显著提高了鲁棒性。通过采用向量视角和考虑滤波器之间的复杂非线性交互，以及引入仅激活模式预测(APOP)方法，可以重新定义相关性并强调非活跃神经元的重要性。 |
| [^8] | [Hierarchical Fashion Design with Multi-stage Diffusion Models.](http://arxiv.org/abs/2401.07450) | 本论文提出了一种名为HieraFashDiff的新型时尚设计方法，它使用多级扩散模型实现了从高级设计概念到低级服装属性的分层设计和编辑，解决了当前在时尚设计中的挑战。 |
| [^9] | [A Comprehensive Multi-scale Approach for Speech and Dynamics Synchrony in Talking Head Generation.](http://arxiv.org/abs/2307.03270) | 该论文提出了一种多尺度方法，通过使用多尺度音视同步损失和多尺度自回归生成对抗网络，实现了语音和头部动力学的同步生成。实验证明，在当前状态下取得了显著的改进。 |
| [^10] | [NeuralFuse: Learning to Improve the Accuracy of Access-Limited Neural Network Inference in Low-Voltage Regimes.](http://arxiv.org/abs/2306.16869) | NeuralFuse是一个新颖的附加模块，通过学习输入转换来生成抗误差的数据表示，解决了低电压环境下有限访问神经网络推断的准确性与能量之间的权衡问题。 |

# 详细

[^1]: 联合胸部X光诊断和临床视觉注意力预测的多阶段协作学习：增强可解释性

    Joint chest X-ray diagnosis and clinical visual attention prediction with multi-stage cooperative learning: enhancing interpretability

    [https://arxiv.org/abs/2403.16970](https://arxiv.org/abs/2403.16970)

    该论文引入了一种新的深度学习框架，用于联合疾病诊断和胸部X光扫描对应视觉显著性图的预测，通过设计新颖的双编码器多任务UNet并利用多尺度特征融合分类器来提高计算辅助诊断的可解释性和质量。

    

    随着深度学习成为计算辅助诊断的最新技术，自动决策的可解释性对临床部署至关重要。尽管在这一领域提出了各种方法，但在放射学筛查过程中临床医生的视觉注意力图为提供重要洞察提供了独特的资产，并有可能提高计算辅助诊断的质量。通过这篇论文，我们引入了一种新颖的深度学习框架，用于联合疾病诊断和胸部X光扫描对应视觉显著性图的预测。具体来说，我们设计了一种新颖的双编码器多任务UNet，利用了DenseNet201主干和基于残差和膨胀激励块的编码器来提取用于显著性图预测的多样特征，并使用多尺度特征融合分类器进行疾病分类。

    arXiv:2403.16970v1 Announce Type: cross  Abstract: As deep learning has become the state-of-the-art for computer-assisted diagnosis, interpretability of the automatic decisions is crucial for clinical deployment. While various methods were proposed in this domain, visual attention maps of clinicians during radiological screening offer a unique asset to provide important insights and can potentially enhance the quality of computer-assisted diagnosis. With this paper, we introduce a novel deep-learning framework for joint disease diagnosis and prediction of corresponding visual saliency maps for chest X-ray scans. Specifically, we designed a novel dual-encoder multi-task UNet, which leverages both a DenseNet201 backbone and a Residual and Squeeze-and-Excitation block-based encoder to extract diverse features for saliency map prediction, and a multi-scale feature-fusion classifier to perform disease classification. To tackle the issue of asynchronous training schedules of individual tasks
    
[^2]: 基于注意力的多源领域自适应目标检测的类别条件对齐

    Attention-based Class-Conditioned Alignment for Multi-Source Domain Adaptive Object Detection

    [https://arxiv.org/abs/2403.09918](https://arxiv.org/abs/2403.09918)

    提出了一种基于注意力的类别条件对齐方案，用于多源领域自适应目标检测，在跨领域对齐每个对象类别的实例。

    

    目标检测（OD）的领域自适应方法致力于通过促进源域和目标域之间的特征对齐来缓解分布转移的影响。多源领域自适应（MSDA）允许利用多个带注释的源数据集和未标记的目标数据来提高检测模型的准确性和鲁棒性。大多数最先进的OD MSDA方法以一种与类别无关的方式执行特征对齐。最近提出的基于原型的方法提出了一种按类别对齐的方法，但由于嘈杂的伪标签而导致错误积累，这可能会对不平衡数据的自适应产生负面影响。为克服这些限制，我们提出了一种基于注意力的类别条件对齐方案，用于MSDA，该方案在跨领域对齐每个对象类别的实例。

    arXiv:2403.09918v1 Announce Type: cross  Abstract: Domain adaptation methods for object detection (OD) strive to mitigate the impact of distribution shifts by promoting feature alignment across source and target domains. Multi-source domain adaptation (MSDA) allows leveraging multiple annotated source datasets, and unlabeled target data to improve the accuracy and robustness of the detection model. Most state-of-the-art MSDA methods for OD perform feature alignment in a class-agnostic manner. This is challenging since the objects have unique modal information due to variations in object appearance across domains. A recent prototype-based approach proposed a class-wise alignment, yet it suffers from error accumulation due to noisy pseudo-labels which can negatively affect adaptation with imbalanced data. To overcome these limitations, we propose an attention-based class-conditioned alignment scheme for MSDA that aligns instances of each object category across domains. In particular, an 
    
[^3]: 通过微调预先为高斯降噪而训练的UNet进行低剂量CT图像重建，用于图像增强的下游任务

    Low-Dose CT Image Reconstruction by Fine-Tuning a UNet Pretrained for Gaussian Denoising for the Downstream Task of Image Enhancement

    [https://arxiv.org/abs/2403.03551](https://arxiv.org/abs/2403.03551)

    提出了一种通过精调UNet进行低剂量CT图像重建的方法，其中第二阶段的训练策略为CT图像增强阶段。

    

    计算机断层扫描（CT）是一种广泛使用的医学成像模态，由于其基于电离辐射，因此希望尽量减少辐射剂量。然而，降低辐射剂量会导致图像质量下降，从低剂量CT（LDCT）数据重建仍然是一个具有挑战性的任务，值得进行研究。根据LoDoPaB-CT基准，许多最先进的方法使用涉及UNet型架构的流程。具体来说，排名第一的方法ItNet使用包括滤波反投影（FBP）、在CT数据上训练的UNet和迭代细化步骤的三阶段流程。在本文中，我们提出了一种更简单的两阶段方法。第一阶段也使用了FBP，而新颖之处在于第二阶段的训练策略，特点是CT图像增强阶段。我们方法的关键点在于神经网络是预训练的。

    arXiv:2403.03551v1 Announce Type: cross  Abstract: Computed Tomography (CT) is a widely used medical imaging modality, and as it is based on ionizing radiation, it is desirable to minimize the radiation dose. However, a reduced radiation dose comes with reduced image quality, and reconstruction from low-dose CT (LDCT) data is still a challenging task which is subject to research. According to the LoDoPaB-CT benchmark, a benchmark for LDCT reconstruction, many state-of-the-art methods use pipelines involving UNet-type architectures. Specifically the top ranking method, ItNet, employs a three-stage process involving filtered backprojection (FBP), a UNet trained on CT data, and an iterative refinement step. In this paper, we propose a less complex two-stage method. The first stage also employs FBP, while the novelty lies in the training strategy for the second stage, characterized as the CT image enhancement stage. The crucial point of our approach is that the neural network is pretrained
    
[^4]: 基于差异方法的灵活物理伪装生成

    Flexible Physical Camouflage Generation Based on a Differential Approach

    [https://arxiv.org/abs/2402.13575](https://arxiv.org/abs/2402.13575)

    该研究引入了一种新颖的神经渲染方法，名为FPA，通过学习对抗模式并结合特殊设计的对抗损失和隐蔽约束损失，可以生成物理世界中具有对抗性和隐蔽性质的伪装。

    

    这项研究介绍了一种新的神经渲染方法，专门针对对抗伪装，在广泛的三维渲染框架内进行了定制。我们的方法，名为FPA，通过忠实地模拟光照条件和材料变化，确保在三维目标上对纹理进行微妙而逼真的表现。为了实现这一目标，我们采用一种生成方法，从扩散模型中学习对抗模式。这涉及将一个特别设计的对抗损失和隐蔽约束损失结合在一起，以确保伪装在物理世界中的对抗性和隐蔽性质。此外，我们展示了所提出的伪装在贴纸模式下的有效性，展示了其覆盖目标而不影响对抗信息的能力。通过实证和物理实验，FPA在攻击成功率和可转移性方面表现出很强的性能。

    arXiv:2402.13575v1 Announce Type: cross  Abstract: This study introduces a novel approach to neural rendering, specifically tailored for adversarial camouflage, within an extensive 3D rendering framework. Our method, named FPA, goes beyond traditional techniques by faithfully simulating lighting conditions and material variations, ensuring a nuanced and realistic representation of textures on a 3D target. To achieve this, we employ a generative approach that learns adversarial patterns from a diffusion model. This involves incorporating a specially designed adversarial loss and covert constraint loss to guarantee the adversarial and covert nature of the camouflage in the physical world. Furthermore, we showcase the effectiveness of the proposed camouflage in sticker mode, demonstrating its ability to cover the target without compromising adversarial information. Through empirical and physical experiments, FPA exhibits strong performance in terms of attack success rate and transferabili
    
[^5]: TorchCP：基于PyTorch的一种适用于合拟常规预测的库

    TorchCP: A Library for Conformal Prediction based on PyTorch

    [https://arxiv.org/abs/2402.12683](https://arxiv.org/abs/2402.12683)

    TorchCP是一个基于PyTorch的Python工具包，为深度学习模型上的合拟常规预测研究提供了实现后验和训练方法的多种工具，包括分类和回归任务。En_Tdlr: TorchCP is a Python toolbox built on PyTorch for conformal prediction research on deep learning models, providing various implementations for posthoc and training methods for classification and regression tasks, including multi-dimension output.

    

    TorchCP是一个用于深度学习模型上的合拟常规预测研究的Python工具包。它包含了用于后验和训练方法的各种实现，用于分类和回归任务（包括多维输出）。TorchCP建立在PyTorch之上，并利用矩阵计算的优势，提供简洁高效的推理实现。该代码采用LGPL许可证，并在$\href{https://github.com/ml-stat-Sustech/TorchCP}{\text{this https URL}}$开源。

    arXiv:2402.12683v1 Announce Type: new  Abstract: TorchCP is a Python toolbox for conformal prediction research on deep learning models. It contains various implementations for posthoc and training methods for classification and regression tasks (including multi-dimension output). TorchCP is built on PyTorch (Paszke et al., 2019) and leverages the advantages of matrix computation to provide concise and efficient inference implementations. The code is licensed under the LGPL license and is open-sourced at $\href{https://github.com/ml-stat-Sustech/TorchCP}{\text{this https URL}}$.
    
[^6]: AICAttack：基于注意力优化的对抗性图像字幕攻击

    AICAttack: Adversarial Image Captioning Attack with Attention-Based Optimization

    [https://arxiv.org/abs/2402.11940](https://arxiv.org/abs/2402.11940)

    提出了一种新的对抗攻击策略AICAttack，旨在通过微小的图像扰动来攻击图像字幕模型，在黑盒攻击情景下具有良好的效果。

    

    最近深度学习研究取得了在计算机视觉（CV）和自然语言处理（NLP）等许多任务上显著的成就。CV和NLP交叉点上的图像字幕问题中，相关模型对抗攻击的稳健性尚未得到充分研究。本文提出了一种新颖的对抗攻击策略，称为AICAttack（基于注意力的图像字幕攻击），旨在通过对图像进行微小扰动来攻击图像字幕模型。在黑盒攻击环境中运行，我们的算法不需要访问目标模型的架构、参数或梯度信息。我们引入了基于注意力的候选选择机制，可识别最佳像素进行攻击，然后采用差分进化（DE）来扰乱像素的RGB值。通过对基准上的广泛实验，我们证明了AICAttack的有效性。

    arXiv:2402.11940v1 Announce Type: cross  Abstract: Recent advances in deep learning research have shown remarkable achievements across many tasks in computer vision (CV) and natural language processing (NLP). At the intersection of CV and NLP is the problem of image captioning, where the related models' robustness against adversarial attacks has not been well studied. In this paper, we present a novel adversarial attack strategy, which we call AICAttack (Attention-based Image Captioning Attack), designed to attack image captioning models through subtle perturbations on images. Operating within a black-box attack scenario, our algorithm requires no access to the target model's architecture, parameters, or gradient information. We introduce an attention-based candidate selection mechanism that identifies the optimal pixels to attack, followed by Differential Evolution (DE) for perturbing pixels' RGB values. We demonstrate AICAttack's effectiveness through extensive experiments on benchma
    
[^7]: 尊重模型: 细粒度且鲁棒的解释与共享比例分解

    Respect the model: Fine-grained and Robust Explanation with Sharing Ratio Decomposition

    [https://arxiv.org/abs/2402.03348](https://arxiv.org/abs/2402.03348)

    本论文提出了一种称为共享比例分解(SRD)的新颖解释方法，真实地反映了模型的推理过程，并在解释方面显著提高了鲁棒性。通过采用向量视角和考虑滤波器之间的复杂非线性交互，以及引入仅激活模式预测(APOP)方法，可以重新定义相关性并强调非活跃神经元的重要性。

    

    对现有的解释方法能否真实阐明模型决策过程的真实性提出了质疑。现有方法偏离了对模型的忠实表达，因此容易受到对抗性攻击的影响。为了解决这个问题，我们提出了一种新颖的可解释性人工智能(XAI)方法，称为SRD(共享比例分解)，它真实地反映了模型的推理过程，从而显著提高了解释的鲁棒性。与传统的神经元级别强调不同，我们采用向量视角来考虑滤波器之间复杂的非线性交互。我们还引入了一个有趣的观察，称为仅激活模式预测(APOP)，让我们强调非活跃神经元的重要性，并重新定义相关性，包括活跃和非活跃神经元的所有相关信息。我们的方法SRD允许递归分解一个点特征向量(PFV)。

    The truthfulness of existing explanation methods in authentically elucidating the underlying model's decision-making process has been questioned. Existing methods have deviated from faithfully representing the model, thus susceptible to adversarial attacks. To address this, we propose a novel eXplainable AI (XAI) method called SRD (Sharing Ratio Decomposition), which sincerely reflects the model's inference process, resulting in significantly enhanced robustness in our explanations. Different from the conventional emphasis on the neuronal level, we adopt a vector perspective to consider the intricate nonlinear interactions between filters. We also introduce an interesting observation termed Activation-Pattern-Only Prediction (APOP), letting us emphasize the importance of inactive neurons and redefine relevance encapsulating all relevant information including both active and inactive neurons. Our method, SRD, allows for the recursive decomposition of a Pointwise Feature Vector (PFV), pr
    
[^8]: 带有多级扩散模型的分层时尚设计

    Hierarchical Fashion Design with Multi-stage Diffusion Models. (arXiv:2401.07450v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2401.07450](http://arxiv.org/abs/2401.07450)

    本论文提出了一种名为HieraFashDiff的新型时尚设计方法，它使用多级扩散模型实现了从高级设计概念到低级服装属性的分层设计和编辑，解决了当前在时尚设计中的挑战。

    

    跨模态时尚合成和编辑通过自动生成和局部修改设计草图，为时尚设计师提供智能支持。尽管当前的扩散模型在图像合成方面表现出了可靠的稳定性和可控性，但在从抽象的设计元素中生成时尚设计和精细编辑方面仍面临重大挑战。高级设计概念，例如办公室、商务和派对，形成了抽象的感官表达方式，而袖长、领型和裤长等可衡量的方面被视为服装的低级属性。使用冗长的文字描述来控制和编辑时尚图像存在困难。在本文中，我们提出了一种名为HieraFashDiff的新型时尚设计方法，它使用共享的多级扩散模型，将高级设计概念和低级服装属性融入到分层结构中。具体而言，我们将输入文本分为不同的层次，并将其输入到多级扩散模型中。

    Cross-modal fashion synthesis and editing offer intelligent support to fashion designers by enabling the automatic generation and local modification of design drafts.While current diffusion models demonstrate commendable stability and controllability in image synthesis,they still face significant challenges in generating fashion design from abstract design elements and fine-grained editing.Abstract sensory expressions, \eg office, business, and party, form the high-level design concepts, while measurable aspects like sleeve length, collar type, and pant length are considered the low-level attributes of clothing.Controlling and editing fashion images using lengthy text descriptions poses a difficulty.In this paper, we propose HieraFashDiff,a novel fashion design method using the shared multi-stage diffusion model encompassing high-level design concepts and low-level clothing attributes in a hierarchical structure.Specifically, we categorized the input text into different levels and fed 
    
[^9]: 语音和动力学同步的全面多尺度方法在虚拟说话头生成中的应用

    A Comprehensive Multi-scale Approach for Speech and Dynamics Synchrony in Talking Head Generation. (arXiv:2307.03270v1 [cs.GR])

    [http://arxiv.org/abs/2307.03270](http://arxiv.org/abs/2307.03270)

    该论文提出了一种多尺度方法，通过使用多尺度音视同步损失和多尺度自回归生成对抗网络，实现了语音和头部动力学的同步生成。实验证明，在当前状态下取得了显著的改进。

    

    使用深度生成模型使用语音输入信号对静态面部图像进行动画化是一个活跃的研究课题，并且近期取得了重要的进展。然而，目前很大一部分工作都集中在嘴唇同步和渲染质量上，很少关注自然头部运动的生成，更不用说头部运动与语音的视听相关性了。本文提出了一种多尺度音视同步损失和多尺度自回归生成对抗网络，以更好地处理语音与头部和嘴唇动力学之间的短期和长期相关性。特别地，我们在多模态输入金字塔上训练了一堆同步模型，并将这些模型用作多尺度生成网络中的指导，以产生不同时间尺度上的音频对齐运动展开。我们的生成器在面部标志域中操作，这是一种标准的低维头部表示方法。实验证明，在头部运动动力学方面取得了显著的改进。

    Animating still face images with deep generative models using a speech input signal is an active research topic and has seen important recent progress. However, much of the effort has been put into lip syncing and rendering quality while the generation of natural head motion, let alone the audio-visual correlation between head motion and speech, has often been neglected. In this work, we propose a multi-scale audio-visual synchrony loss and a multi-scale autoregressive GAN to better handle short and long-term correlation between speech and the dynamics of the head and lips. In particular, we train a stack of syncer models on multimodal input pyramids and use these models as guidance in a multi-scale generator network to produce audio-aligned motion unfolding over diverse time scales. Our generator operates in the facial landmark domain, which is a standard low-dimensional head representation. The experiments show significant improvements over the state of the art in head motion dynamic
    
[^10]: NeuralFuse: 学习改善低电压环境下有限访问神经网络推断的准确性

    NeuralFuse: Learning to Improve the Accuracy of Access-Limited Neural Network Inference in Low-Voltage Regimes. (arXiv:2306.16869v1 [cs.LG])

    [http://arxiv.org/abs/2306.16869](http://arxiv.org/abs/2306.16869)

    NeuralFuse是一个新颖的附加模块，通过学习输入转换来生成抗误差的数据表示，解决了低电压环境下有限访问神经网络推断的准确性与能量之间的权衡问题。

    

    深度神经网络在机器学习中已经无处不在，但其能量消耗仍然是一个值得关注的问题。降低供电电压是降低能量消耗的有效策略。然而，过度降低供电电压可能会导致准确性降低，因为模型参数存储在静态随机存储器(SRAM)中，而SRAM中会发生随机位翻转。为了解决这个挑战，我们引入了NeuralFuse，这是一个新颖的附加模块，通过学习输入转换来生成抗误差的数据表示，以在低电压环境中解决准确性与能量之间的权衡。NeuralFuse在标称电压和低电压情况下都能保护DNN的准确性。此外，NeuralFuse易于实现，并可以轻松应用于有限访问的DNN，例如不可配置的硬件或云端API的远程访问。实验结果表明，在1%的位错误率下，NeuralFuse可以将SRAM内存访问能量降低高达24%，同时保持准确性。

    Deep neural networks (DNNs) have become ubiquitous in machine learning, but their energy consumption remains a notable issue. Lowering the supply voltage is an effective strategy for reducing energy consumption. However, aggressively scaling down the supply voltage can lead to accuracy degradation due to random bit flips in static random access memory (SRAM) where model parameters are stored. To address this challenge, we introduce NeuralFuse, a novel add-on module that addresses the accuracy-energy tradeoff in low-voltage regimes by learning input transformations to generate error-resistant data representations. NeuralFuse protects DNN accuracy in both nominal and low-voltage scenarios. Moreover, NeuralFuse is easy to implement and can be readily applied to DNNs with limited access, such as non-configurable hardware or remote access to cloud-based APIs. Experimental results demonstrate that, at a 1% bit error rate, NeuralFuse can reduce SRAM memory access energy by up to 24% while imp
    

