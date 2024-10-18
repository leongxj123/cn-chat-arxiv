# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Corrective Machine Unlearning](https://arxiv.org/abs/2402.14015) | 该论文通过形式化“修正机器消除”来解决受未知操纵影响的数据对训练模型的影响问题，可能仅知道一部分受影响样本。发现纠正消除问题与传统以隐私为导向的消除方法有显著不同的要求。 |
| [^2] | [FlashTex: Fast Relightable Mesh Texturing with LightControlNet](https://arxiv.org/abs/2402.13251) | 提出了FlashTex方法，基于LightControlNet实现了快速自动化3D网格纹理生成，实现了照明与表面材质的解耦，使得网格能够在任何照明环境下正确重照和渲染 |
| [^3] | [Multilinear Mixture of Experts: Scalable Expert Specialization through Factorization](https://arxiv.org/abs/2402.12550) | 多线性专家混合（MMoE）层通过因式分解针对视觉模型提供了一种可扩展的专家特化解决方案，避免了离散专家路由和过高推理时间成本。 |
| [^4] | [Tables as Images? Exploring the Strengths and Limitations of LLMs on Multimodal Representations of Tabular Data](https://arxiv.org/abs/2402.12424) | 本研究探讨了LLM在解释表格数据方面的有效性，比较了文本和图像表格表示对LLM性能的影响，为在表格相关任务上有效使用LLM提供了见解。 |
| [^5] | [Learning Contrastive Feature Representations for Facial Action Unit Detection](https://arxiv.org/abs/2402.06165) | 这项研究提出了一种对比学习框架，通过监督和自监督信号来增强面部动作单元检测模型的性能。采用正样本抽样和权衡重要性的损失函数来应对噪声AU标签和AU类型分布不平衡的挑战。 |
| [^6] | [Self-supervised learning of video representations from a child's perspective](https://arxiv.org/abs/2402.00300) | 本研究从儿童的视角进行自监督学习，通过长时间的头戴式摄像记录训练视频模型，结果表明这些模型在促进从少量样本中学习行动概念方面非常有效。 |
| [^7] | [Unveiling the Blind Spots: A Critical Examination of Fairness in Autonomous Driving Systems](https://arxiv.org/abs/2308.02935) | 该研究对当前深度学习行人检测器的公平性进行了全面评估，发现了与年龄相关的重要公平性问题。 |
| [^8] | [D$^3$Fields: Dynamic 3D Descriptor Fields for Zero-Shot Generalizable Robotic Manipulation.](http://arxiv.org/abs/2309.16118) | D$^3$Fields是一个动态的三维描述符场，将底层三维环境的动态特性以及语义特征和实例掩模编码起来。它可以灵活地使用不同背景、风格和实例的二维图像指定目标，实现零样本机器人操作任务的可泛化。 |
| [^9] | [HyperDreamBooth: HyperNetworks for Fast Personalization of Text-to-Image Models.](http://arxiv.org/abs/2307.06949) | HyperDreamBooth是一个超网络，可以从一个人的单张图片中快速生成个性化权重，从而实现在多种背景和风格下合成一个人的面部，保持高保真度并同时保留对多样化风格和语义修改的关键知识。 |
| [^10] | [Degraded Polygons Raise Fundamental Questions of Neural Network Perception.](http://arxiv.org/abs/2306.04955) | 本文研究了神经网络在识别具有不同程度边缘降解的规则多边形时的性能和行为，发现存在基本问题，揭示了人机视觉差距的另一个角度。 |
| [^11] | [In Defense of Pure 16-bit Floating-Point Neural Networks.](http://arxiv.org/abs/2305.10947) | 本文探讨了纯16位浮点神经网络的被忽视的效率，提供了理论分析来探讨16位和32位模型的差异，并可以定量解释16位模型与其32位对应物之间的条件。 |

# 详细

[^1]: 修正机器消除

    Corrective Machine Unlearning

    [https://arxiv.org/abs/2402.14015](https://arxiv.org/abs/2402.14015)

    该论文通过形式化“修正机器消除”来解决受未知操纵影响的数据对训练模型的影响问题，可能仅知道一部分受影响样本。发现纠正消除问题与传统以隐私为导向的消除方法有显著不同的要求。

    

    机器学习模型越来越面临数据完整性挑战，因为它们使用了大规模的从互联网中获取的训练数据集。本文研究了如果模型开发者发现某些数据被篡改或错误，他们可以采取什么措施。这些被篡改的数据会导致不利影响，如容易受到后门样本的攻击、系统性偏见，以及在某些输入领域的准确度降低。通常，并非所有被篡改的训练样本都是已知的，而只有一小部分代表性的受影响数据被标记。

    arXiv:2402.14015v1 Announce Type: cross  Abstract: Machine Learning models increasingly face data integrity challenges due to the use of large-scale training datasets drawn from the internet. We study what model developers can do if they detect that some data was manipulated or incorrect. Such manipulated data can cause adverse effects like vulnerability to backdoored samples, systematic biases, and in general, reduced accuracy on certain input domains. Often, all manipulated training samples are not known, and only a small, representative subset of the affected data is flagged.   We formalize "Corrective Machine Unlearning" as the problem of mitigating the impact of data affected by unknown manipulations on a trained model, possibly knowing only a subset of impacted samples. We demonstrate that the problem of corrective unlearning has significantly different requirements from traditional privacy-oriented unlearning. We find most existing unlearning methods, including the gold-standard
    
[^2]: FlashTex：具有LightControlNet的快速可重塑网格纹理

    FlashTex: Fast Relightable Mesh Texturing with LightControlNet

    [https://arxiv.org/abs/2402.13251](https://arxiv.org/abs/2402.13251)

    提出了FlashTex方法，基于LightControlNet实现了快速自动化3D网格纹理生成，实现了照明与表面材质的解耦，使得网格能够在任何照明环境下正确重照和渲染

    

    手动为3D网格创建纹理费时费力，即使对于专家视觉内容创建者也是如此。我们提出了一种快速方法，根据用户提供的文本提示自动为输入的3D网格着色。重要的是，我们的方法将照明与表面材质/反射在生成的纹理中解耦，以便网格可以在任何照明环境中正确重照和渲染。我们引入了LightControlNet，这是一种基于ControlNet架构的新文本到图像模型，允许将所需照明规格作为对模型的条件图像。我们的文本到纹理管道然后分两个阶段构建纹理。第一阶段使用LightControlNet生成网格的一组稀疏的视觉一致的参考视图。第二阶段应用基于分数蒸馏采样（SDS）的纹理优化，通过LightControlNet来提高纹理质量同时解耦表面材质

    arXiv:2402.13251v1 Announce Type: cross  Abstract: Manually creating textures for 3D meshes is time-consuming, even for expert visual content creators. We propose a fast approach for automatically texturing an input 3D mesh based on a user-provided text prompt. Importantly, our approach disentangles lighting from surface material/reflectance in the resulting texture so that the mesh can be properly relit and rendered in any lighting environment. We introduce LightControlNet, a new text-to-image model based on the ControlNet architecture, which allows the specification of the desired lighting as a conditioning image to the model. Our text-to-texture pipeline then constructs the texture in two stages. The first stage produces a sparse set of visually consistent reference views of the mesh using LightControlNet. The second stage applies a texture optimization based on Score Distillation Sampling (SDS) that works with LightControlNet to increase the texture quality while disentangling surf
    
[^3]: 多线性专家混合：通过因式分解实现可扩展的专家特化

    Multilinear Mixture of Experts: Scalable Expert Specialization through Factorization

    [https://arxiv.org/abs/2402.12550](https://arxiv.org/abs/2402.12550)

    多线性专家混合（MMoE）层通过因式分解针对视觉模型提供了一种可扩展的专家特化解决方案，避免了离散专家路由和过高推理时间成本。

    

    专家混合（MoE）范式提供了一种强大的方法，将难以理解的密集层分解为更小、模块化的计算，通常更易于人类解释、调试和编辑。然而，一个主要问题在于扩展专家数量的计算成本，以实现足够精细的专业化。本文提出了多线性专家混合（MMoE）层来解决这个问题，重点放在视觉模型上。MMoE层完全以因式化形式对庞大的权重张量进行隐式计算。因此，MMoEs既避免了在流行的“稀疏”MoE模型中离散专家路由所造成的问题，又不会引起“软”MoE替代方案中过高的推理时间成本。我们通过可视化和反事实干预，提供了定性和定量证据，证明了扩展MMoE层的效果。

    arXiv:2402.12550v1 Announce Type: cross  Abstract: The Mixture of Experts (MoE) paradigm provides a powerful way to decompose inscrutable dense layers into smaller, modular computations often more amenable to human interpretation, debugging, and editability. A major problem however lies in the computational cost of scaling the number of experts to achieve sufficiently fine-grained specialization. In this paper, we propose the Multilinear Mixutre of Experts (MMoE) layer to address this, focusing on vision models. MMoE layers perform an implicit computation on prohibitively large weight tensors entirely in factorized form. Consequently, MMoEs both (1) avoid the issues incurred through the discrete expert routing in the popular 'sparse' MoE models, yet (2) do not incur the restrictively high inference-time costs of 'soft' MoE alternatives. We present both qualitative and quantitative evidence (through visualization and counterfactual interventions respectively) that scaling MMoE layers wh
    
[^4]: 表格作为图片？探讨LLM在多模态表格数据表示上的优势和局限性

    Tables as Images? Exploring the Strengths and Limitations of LLMs on Multimodal Representations of Tabular Data

    [https://arxiv.org/abs/2402.12424](https://arxiv.org/abs/2402.12424)

    本研究探讨了LLM在解释表格数据方面的有效性，比较了文本和图像表格表示对LLM性能的影响，为在表格相关任务上有效使用LLM提供了见解。

    

    在本文中，我们通过不同的提示策略和数据格式研究了各种LLM在解释表格数据方面的有效性。我们的分析涵盖了六个针对与表格相关任务的基准，如问答和事实核查。我们首次介绍了LLM在基于图像的表格表示上的表现评估。具体地，我们比较了五种基于文本和三种基于图像的表格表示，展示了表示和提示对LLM性能的影响。我们的研究为在表格相关任务上有效使用LLM提供了见解。

    arXiv:2402.12424v1 Announce Type: cross  Abstract: In this paper, we investigate the effectiveness of various LLMs in interpreting tabular data through different prompting strategies and data formats. Our analysis extends across six benchmarks for table-related tasks such as question-answering and fact-checking. We introduce for the first time the assessment of LLMs' performance on image-based table representations. Specifically, we compare five text-based and three image-based table representations, demonstrating the influence of representation and prompting on LLM performance. Our study provides insights into the effective use of LLMs on table-related tasks.
    
[^5]: 学习对比特征表示来进行面部动作单元检测

    Learning Contrastive Feature Representations for Facial Action Unit Detection

    [https://arxiv.org/abs/2402.06165](https://arxiv.org/abs/2402.06165)

    这项研究提出了一种对比学习框架，通过监督和自监督信号来增强面部动作单元检测模型的性能。采用正样本抽样和权衡重要性的损失函数来应对噪声AU标签和AU类型分布不平衡的挑战。

    

    面部动作单元（AU）检测的主要方法涉及监督的多标签二进制分类问题。现有的方法常常对AU的像素级信息进行编码，从而对模型的复杂性和表达能力提出了很大的要求。此外，由于存在噪声AU标签，这种做法增加了过拟合的风险。在本研究中，我们引入了一个对比学习框架，通过监督和自监督信号增强。目标是在AU检测领域中摆脱传统的像素级学习范式，获得判别特征。为了应对噪声AU标签带来的挑战，我们通过引入自监督信号来增强监督信号。这种增强是通过正样本抽样实现的，包括三种不同类型的正样本对。另外，为了减轻每个AU类型的分布不平衡问题，我们采用了一种权衡重要性的损失函数。

    The predominant approach to facial action unit (AU) detection revolves around a supervised multi-label binary classification problem. Existing methodologies often encode pixel-level information of AUs, thereby imposing substantial demands on model complexity and expressiveness. Moreover, this practice elevates the susceptibility to overfitting due to the presence of noisy AU labels. In the present study, we introduce a contrastive learning framework enhanced by both supervised and self-supervised signals. The objective is to acquire discriminative features, deviating from the conventional pixel-level learning paradigm within the domain of AU detection. To address the challenge posed by noisy AU labels, we augment the supervised signal through the introduction of a self-supervised signal. This augmentation is achieved through positive sample sampling, encompassing three distinct types of positive sample pairs. Furthermore, to mitigate the imbalanced distribution of each AU type, we empl
    
[^6]: 从儿童视角进行自监督学习的视频表示

    Self-supervised learning of video representations from a child's perspective

    [https://arxiv.org/abs/2402.00300](https://arxiv.org/abs/2402.00300)

    本研究从儿童的视角进行自监督学习，通过长时间的头戴式摄像记录训练视频模型，结果表明这些模型在促进从少量样本中学习行动概念方面非常有效。

    

    儿童通过几年的自我视觉经验学习到了强大的世界内部模型。这些内部模型能否通过儿童的视觉体验和通用的自监督学习算法来学习，还是需要强大的归纳偏差？最近，在收集大规模、纵向的发展现实视频数据集以及通用的自监督学习算法的进展使我们能够开始探讨这个本质与养育之间的问题。然而，现有的工作通常关注基于图像的自监督学习算法和可以从静态图像中学习的视觉能力（例如目标识别），从而忽略了世界的时间性质。为了弥合这一差距，我们在一个儿童早期发展阶段（6-31个月）从儿童的头戴式摄像记录中训练自监督视频模型。所得到的模型在促进从少量样本中学习行动概念方面非常有效。

    Children learn powerful internal models of the world around them from a few years of egocentric visual experience. Can such internal models be learned from a child's visual experience with highly generic learning algorithms or do they require strong inductive biases? Recent advances in collecting large-scale, longitudinal, developmentally realistic video datasets and generic self-supervised learning (SSL) algorithms are allowing us to begin to tackle this nature vs. nurture question. However, existing work typically focuses on image-based SSL algorithms and visual capabilities that can be learned from static images (e.g. object recognition), thus ignoring temporal aspects of the world. To close this gap, here we train self-supervised video models on longitudinal, egocentric headcam recordings collected from a child over a two year period in their early development (6-31 months). The resulting models are highly effective at facilitating the learning of action concepts from a small numbe
    
[^7]: 揭示盲点：对自动驾驶系统中公平性的关键审查

    Unveiling the Blind Spots: A Critical Examination of Fairness in Autonomous Driving Systems

    [https://arxiv.org/abs/2308.02935](https://arxiv.org/abs/2308.02935)

    该研究对当前深度学习行人检测器的公平性进行了全面评估，发现了与年龄相关的重要公平性问题。

    

    自主驾驶系统已经扩展了智能车辆物联网的范围，并成为Web生态系统的重要组成部分。类似于传统的基于Web的应用程序，公平性对于确保自动驾驶系统的高质量是一个重要方面，特别是在其中的行人检测器的背景下。然而，目前关于当前基于深度学习（DL）的行人检测器公平性的综合评估在文献中尚未出现。为了填补这一空白，我们在大规模真实世界数据集上评估了八种被广泛探索的DL行人检测器在人口统计学群体之间的表现。为了实现彻底的公平性评估，我们为数据集提供了广泛的注释，共涉及8,311张图像，16,070个性别标签，20,115个年龄标签和3,513个肤色标签。我们的研究发现了与年龄相关的重要公平性问题。

    arXiv:2308.02935v2 Announce Type: replace-cross  Abstract: Autonomous driving systems have extended the spectrum of Web of Things for intelligent vehicles and have become an important component of the Web ecosystem. Similar to traditional Web-based applications, fairness is an essential aspect for ensuring the high quality of autonomous driving systems, particularly in the context of pedestrian detectors within them. However, there is an absence in the literature of a comprehensive assessment of the fairness of current Deep Learning (DL)-based pedestrian detectors. To fill the gap, we evaluate eight widely-explored DL-based pedestrian detectors across demographic groups on large-scale real-world datasets. To enable a thorough fairness evaluation, we provide extensive annotations for the datasets, resulting in 8,311 images with 16,070 gender labels, 20,115 age labels, and 3,513 skin tone labels. Our findings reveal significant fairness issues related to age. The undetected proportions f
    
[^8]: D$^3$Fields: 动态三维描述符场用于零样本可泛化机器人操作

    D$^3$Fields: Dynamic 3D Descriptor Fields for Zero-Shot Generalizable Robotic Manipulation. (arXiv:2309.16118v1 [cs.RO])

    [http://arxiv.org/abs/2309.16118](http://arxiv.org/abs/2309.16118)

    D$^3$Fields是一个动态的三维描述符场，将底层三维环境的动态特性以及语义特征和实例掩模编码起来。它可以灵活地使用不同背景、风格和实例的二维图像指定目标，实现零样本机器人操作任务的可泛化。

    

    场景表示是机器人操作系统中一个关键的设计选择。一个理想的表示应该是三维的、动态的和语义化的，以满足不同操作任务的需求。然而，先前的工作往往同时缺乏这三个属性。在这项工作中，我们介绍了D$^3$Fields动态三维描述符场。这些场捕捉了底层三维环境的动态特性，编码了语义特征和实例掩模。具体而言，我们将工作区域中的任意三维点投影到多视角的二维视觉观察中，并插值从基本模型中得到的特征。由此得到的融合描述符场可以使用具有不同背景、风格和实例的二维图像灵活地指定目标。为了评估这些描述符场的有效性，我们以零样本方式将我们的表示应用于各种机器人操作任务。通过在真实场景和模拟中的广泛评估，我们展示了该方法的有效性。

    Scene representation has been a crucial design choice in robotic manipulation systems. An ideal representation should be 3D, dynamic, and semantic to meet the demands of diverse manipulation tasks. However, previous works often lack all three properties simultaneously. In this work, we introduce D$^3$Fields dynamic 3D descriptor fields. These fields capture the dynamics of the underlying 3D environment and encode both semantic features and instance masks. Specifically, we project arbitrary 3D points in the workspace onto multi-view 2D visual observations and interpolate features derived from foundational models. The resulting fused descriptor fields allow for flexible goal specifications using 2D images with varied contexts, styles, and instances. To evaluate the effectiveness of these descriptor fields, we apply our representation to a wide range of robotic manipulation tasks in a zero-shot manner. Through extensive evaluation in both real-world scenarios and simulations, we demonst
    
[^9]: HyperDreamBooth：用于快速个性化文本到图像模型的超网络

    HyperDreamBooth: HyperNetworks for Fast Personalization of Text-to-Image Models. (arXiv:2307.06949v1 [cs.CV])

    [http://arxiv.org/abs/2307.06949](http://arxiv.org/abs/2307.06949)

    HyperDreamBooth是一个超网络，可以从一个人的单张图片中快速生成个性化权重，从而实现在多种背景和风格下合成一个人的面部，保持高保真度并同时保留对多样化风格和语义修改的关键知识。

    

    个性化已经成为生成式人工智能领域中的一个重要方面，使得在不同背景和风格下合成个体成为可能，同时保持高保真度。然而，个性化过程在时间和内存需求方面存在困难。每个个性化模型的微调需要大量的GPU时间投入，为每个主题存储一个个性化模型会对存储容量提出要求。为了克服这些挑战，我们提出了HyperDreamBooth-一种能够从一个人的单张图片有效生成一组个性化权重的超网络。通过将这些权重组合到扩散模型中，并搭配快速微调，HyperDreamBooth能够以多种背景和风格生成一个人的面部，保持高主题细节同时也保持模型对多样化风格和语义修改的关键知识。我们的方法在大约50倍体现了面部个性化。

    Personalization has emerged as a prominent aspect within the field of generative AI, enabling the synthesis of individuals in diverse contexts and styles, while retaining high-fidelity to their identities. However, the process of personalization presents inherent challenges in terms of time and memory requirements. Fine-tuning each personalized model needs considerable GPU time investment, and storing a personalized model per subject can be demanding in terms of storage capacity. To overcome these challenges, we propose HyperDreamBooth-a hypernetwork capable of efficiently generating a small set of personalized weights from a single image of a person. By composing these weights into the diffusion model, coupled with fast finetuning, HyperDreamBooth can generate a person's face in various contexts and styles, with high subject details while also preserving the model's crucial knowledge of diverse styles and semantic modifications. Our method achieves personalization on faces in roughly 
    
[^10]: 论神经网络对降解多边形的感知存在的基本问题

    Degraded Polygons Raise Fundamental Questions of Neural Network Perception. (arXiv:2306.04955v1 [cs.CV])

    [http://arxiv.org/abs/2306.04955](http://arxiv.org/abs/2306.04955)

    本文研究了神经网络在识别具有不同程度边缘降解的规则多边形时的性能和行为，发现存在基本问题，揭示了人机视觉差距的另一个角度。

    

    现代计算机视觉系统往往表现出与人类不一致的行为：从对抗攻击到图像损坏，深度学习视觉模型在各种环境中都表现不佳，然而人类却能够很好地解决这些问题。本文从另一个角度研究了人机视觉差距。我们重新审视了恢复受损图像的任务，该任务在人类视觉的“识别组件”理论中首次引入，研究了神经网络在分类具有不同程度边缘降解的规则多边形时的性能和行为。为此，我们使用了自动化形状可恢复性测试，快速生成了大规模数据集，将历史上手动创建图像可恢复性实验的方法进行了现代化改进。我们进一步研究了神经网络识别多边形的能力以及其相关问题。

    It is well-known that modern computer vision systems often exhibit behaviors misaligned with those of humans: from adversarial attacks to image corruptions, deep learning vision models suffer in a variety of settings that humans capably handle. In light of these phenomena, here we introduce another, orthogonal perspective studying the human-machine vision gap. We revisit the task of recovering images under degradation, first introduced over 30 years ago in the Recognition-by-Components theory of human vision. Specifically, we study the performance and behavior of neural networks on the seemingly simple task of classifying regular polygons at varying orders of degradation along their perimeters. To this end, we implement the Automated Shape Recoverability Test for rapidly generating large-scale datasets of perimeter-degraded regular polygons, modernizing the historically manual creation of image recoverability experiments. We then investigate the capacity of neural networks to recognize
    
[^11]: 关于纯16位浮点神经网络的辩护

    In Defense of Pure 16-bit Floating-Point Neural Networks. (arXiv:2305.10947v1 [cs.LG])

    [http://arxiv.org/abs/2305.10947](http://arxiv.org/abs/2305.10947)

    本文探讨了纯16位浮点神经网络的被忽视的效率，提供了理论分析来探讨16位和32位模型的差异，并可以定量解释16位模型与其32位对应物之间的条件。

    

    减少编码神经网络权重和激活所需的位数是非常可取的，因为它可以加快神经网络的训练和推理时间，同时减少内存消耗。因此，这一领域的研究引起了广泛关注，以开发利用更低精度计算的神经网络，比如混合精度训练。有趣的是，目前不存在纯16位浮点设置的方法。本文揭示了纯16位浮点神经网络被忽视的效率。我们通过提供全面的理论分析来探讨造成16位和32位模型的差异的因素。我们规范化了浮点误差和容忍度的概念，从而可以定量解释16位模型与其32位对应物之间密切逼近结果的条件。这种理论探索提供了新的视角。

    Reducing the number of bits needed to encode the weights and activations of neural networks is highly desirable as it speeds up their training and inference time while reducing memory consumption. For these reasons, research in this area has attracted significant attention toward developing neural networks that leverage lower-precision computing, such as mixed-precision training. Interestingly, none of the existing approaches has investigated pure 16-bit floating-point settings. In this paper, we shed light on the overlooked efficiency of pure 16-bit floating-point neural networks. As such, we provide a comprehensive theoretical analysis to investigate the factors contributing to the differences observed between 16-bit and 32-bit models. We formalize the concepts of floating-point error and tolerance, enabling us to quantitatively explain the conditions under which a 16-bit model can closely approximate the results of its 32-bit counterpart. This theoretical exploration offers perspect
    

