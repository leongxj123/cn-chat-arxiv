# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generation and Detection of Sign Language Deepfakes - A Linguistic and Visual Analysis](https://arxiv.org/abs/2404.01438) | 通过在上半身生成手语深度伪造视频，并由手语专家审核，本研究探讨了在深度伪造技术中的积极应用，为聋哑和听障社区带来潜在的健康和教育益处。 |
| [^2] | [Joint chest X-ray diagnosis and clinical visual attention prediction with multi-stage cooperative learning: enhancing interpretability](https://arxiv.org/abs/2403.16970) | 该论文引入了一种新的深度学习框架，用于联合疾病诊断和胸部X光扫描对应视觉显著性图的预测，通过设计新颖的双编码器多任务UNet并利用多尺度特征融合分类器来提高计算辅助诊断的可解释性和质量。 |
| [^3] | [Reasoning-Enhanced Object-Centric Learning for Videos](https://arxiv.org/abs/2403.15245) | 设计了一种新颖的推理模块STATM，利用记忆缓冲区增强模型在复杂场景中的感知能力。 |
| [^4] | [HyperFusion: A Hypernetwork Approach to Multimodal Integration of Tabular and Medical Imaging Data for Predictive Modeling](https://arxiv.org/abs/2403.13319) | 提出一种基于超网络的新框架，通过将图像处理条件设置在EHR的值和测量上，以整合临床成像和表格数据，旨在利用这些数据中的互补信息。 |
| [^5] | [Removing Undesirable Concepts in Text-to-Image Generative Models with Learnable Prompts](https://arxiv.org/abs/2403.12326) | 通过引入可学习提示到交叉注意力模块中，本文提出了一种新方法，用于从文本到图像生成模型中去除不良概念，实现了对模型效果的提升。 |
| [^6] | [Neural Slot Interpreters: Grounding Object Semantics in Emergent Slot Representations](https://arxiv.org/abs/2403.07887) | 提出了神经槽解释器（NSI），通过槽表示学习接地和生成物体语义，实现了将现实世界的物体语义结合到抽象中。 |
| [^7] | [You Need to Pay Better Attention](https://arxiv.org/abs/2403.01643) | 提出了三种新的注意力机制，在效率和学习能力方面优于标准的多头注意力，提高了Transformer模型的性能和更广泛的部署能力。 |
| [^8] | [Learning to Learn from APIs: Black-Box Data-Free Meta-Learning.](http://arxiv.org/abs/2305.18413) | 该论文提出了一个BiDf-MKD框架，可以从一组API库中无需访问训练数据，直接进行元学习；能够在更广泛的黑盒API上进行元学习，提高了元模型的泛化性能和应用范围。 |
| [^9] | [GPT-NAS: Neural Architecture Search with the Generative Pre-Trained Model.](http://arxiv.org/abs/2305.05351) | GPT-NAS使用生成式预训练模型优化神经架构搜索，通过提出近似的架构组件减小搜索空间，并明显优于其他NAS方法。 |
| [^10] | [Architecture, Dataset and Model-Scale Agnostic Data-free Meta-Learning.](http://arxiv.org/abs/2303.11183) | 不受体系结构、数据集和模型规模限制的无数据元学习框架PURER，通过ECI执行伪周期训练以适应新的任务，通过ICFIL对反演梯度进行校准来优化反演过程，并在各种任务中显著优于现有方法。 |
| [^11] | [Human alignment of neural network representations.](http://arxiv.org/abs/2211.01201) | 本文研究神经网络表示与人类心理表示之间的对齐问题，发现模型规模和体系结构对对齐几乎没有影响，而训练数据集和目标函数都对对齐有很大的影响。从一个数据集中学习的神经网络表示的线性变换能显著提高对另外两个数据集中人类相似性判断的对齐性。 |

# 详细

[^1]: 深度伪造手语的生成与检测-语言和视觉分析

    Generation and Detection of Sign Language Deepfakes - A Linguistic and Visual Analysis

    [https://arxiv.org/abs/2404.01438](https://arxiv.org/abs/2404.01438)

    通过在上半身生成手语深度伪造视频，并由手语专家审核，本研究探讨了在深度伪造技术中的积极应用，为聋哑和听障社区带来潜在的健康和教育益处。

    

    深度伪造领域中一个逐渐出现的问题是我们是否可以超越面部深度伪造，以及这对社会是否有益。因此，这项研究提出了在上半身生成中深度伪造技术的积极应用，同时为聋哑和听障（DHoH）社区执行手语。随后，通过一位手语专家对生成的视频进行审核。鉴于手语的复杂性、手语专家的匮乏以及对健康和教育的潜在益处，这种做法尤为有益。本研究的目标包括构建可靠的深度伪造数据集，通过计算机视觉和自然语言处理模型评估其技术和视觉可信度，以及评估生成内容的可信度。使用1200多个视频，模型包含以前见过和未见过的个体，借助一位手语专家的帮助进行生成。

    arXiv:2404.01438v1 Announce Type: cross  Abstract: A question in the realm of deepfakes is slowly emerging pertaining to whether we can go beyond facial deepfakes and whether it would be beneficial to society. Therefore, this research presents a positive application of deepfake technology in upper body generation, while performing sign-language for the Deaf and Hard of Hearing (DHoH) community. The resulting videos are later vetted with a sign language expert. This is particularly helpful, given the intricate nature of sign language, a scarcity of sign language experts, and potential benefits for health and education. The objectives of this work encompass constructing a reliable deepfake dataset, evaluating its technical and visual credibility through computer vision and natural language processing models, and assessing the plausibility of the generated content. With over 1200 videos, featuring both previously seen and unseen individuals for the generation model, using the help of a si
    
[^2]: 联合胸部X光诊断和临床视觉注意力预测的多阶段协作学习：增强可解释性

    Joint chest X-ray diagnosis and clinical visual attention prediction with multi-stage cooperative learning: enhancing interpretability

    [https://arxiv.org/abs/2403.16970](https://arxiv.org/abs/2403.16970)

    该论文引入了一种新的深度学习框架，用于联合疾病诊断和胸部X光扫描对应视觉显著性图的预测，通过设计新颖的双编码器多任务UNet并利用多尺度特征融合分类器来提高计算辅助诊断的可解释性和质量。

    

    随着深度学习成为计算辅助诊断的最新技术，自动决策的可解释性对临床部署至关重要。尽管在这一领域提出了各种方法，但在放射学筛查过程中临床医生的视觉注意力图为提供重要洞察提供了独特的资产，并有可能提高计算辅助诊断的质量。通过这篇论文，我们引入了一种新颖的深度学习框架，用于联合疾病诊断和胸部X光扫描对应视觉显著性图的预测。具体来说，我们设计了一种新颖的双编码器多任务UNet，利用了DenseNet201主干和基于残差和膨胀激励块的编码器来提取用于显著性图预测的多样特征，并使用多尺度特征融合分类器进行疾病分类。

    arXiv:2403.16970v1 Announce Type: cross  Abstract: As deep learning has become the state-of-the-art for computer-assisted diagnosis, interpretability of the automatic decisions is crucial for clinical deployment. While various methods were proposed in this domain, visual attention maps of clinicians during radiological screening offer a unique asset to provide important insights and can potentially enhance the quality of computer-assisted diagnosis. With this paper, we introduce a novel deep-learning framework for joint disease diagnosis and prediction of corresponding visual saliency maps for chest X-ray scans. Specifically, we designed a novel dual-encoder multi-task UNet, which leverages both a DenseNet201 backbone and a Residual and Squeeze-and-Excitation block-based encoder to extract diverse features for saliency map prediction, and a multi-scale feature-fusion classifier to perform disease classification. To tackle the issue of asynchronous training schedules of individual tasks
    
[^3]: 视频的增强推理对象中心学习

    Reasoning-Enhanced Object-Centric Learning for Videos

    [https://arxiv.org/abs/2403.15245](https://arxiv.org/abs/2403.15245)

    设计了一种新颖的推理模块STATM，利用记忆缓冲区增强模型在复杂场景中的感知能力。

    

    物体中心学习旨在将复杂的视觉场景分解为更易处理的物体表示，提升机器学习系统对物理世界的理解和推理能力。最近，基于槽位的视频模型展现出在分割和跟踪物体方面出色的能力，但忽视了有效推理模块的重要性。为了增强模型在复杂场景中的感知能力，我们设计了一种名为具有记忆缓冲区的基于槽位的时空变换器（STATM）的新型推理模块。记忆缓冲区主要用于存储来自上游模块的槽位信息，基于槽位的时空变换器通过槽位为基础进行预测。

    arXiv:2403.15245v1 Announce Type: cross  Abstract: Object-centric learning aims to break down complex visual scenes into more manageable object representations, enhancing the understanding and reasoning abilities of machine learning systems toward the physical world. Recently, slot-based video models have demonstrated remarkable proficiency in segmenting and tracking objects, but they overlook the importance of the effective reasoning module. In the real world, reasoning and predictive abilities play a crucial role in human perception and object tracking; in particular, these abilities are closely related to human intuitive physics. Inspired by this, we designed a novel reasoning module called the Slot-based Time-Space Transformer with Memory buffer (STATM) to enhance the model's perception ability in complex scenes. The memory buffer primarily serves as storage for slot information from upstream modules, the Slot-based Time-Space Transformer makes predictions through slot-based spatio
    
[^4]: HyperFusion：一种用于预测建模的多模态整合表格和医学成像数据的超网络方法

    HyperFusion: A Hypernetwork Approach to Multimodal Integration of Tabular and Medical Imaging Data for Predictive Modeling

    [https://arxiv.org/abs/2403.13319](https://arxiv.org/abs/2403.13319)

    提出一种基于超网络的新框架，通过将图像处理条件设置在EHR的值和测量上，以整合临床成像和表格数据，旨在利用这些数据中的互补信息。

    

    ARXIV: 2403.13319v1 公告类型: 交叉摘要: 整合各种临床模式，如医学成像和患者电子健康记录（EHR）获得的表格数据，是现代医疗保健的关键方面。多源数据的综合分析可以全面了解患者的状况，并可以增强诊断和治疗决策。深度神经网络（DNN）在医学领域的多模态任务中一直展示出出色的性能。然而，有效地将医学成像与以数字表格数据表示的临床、人口统计和遗传信息进行融合的复杂努力仍然是一个高度活跃的持续研究追求。

    arXiv:2403.13319v1 Announce Type: cross  Abstract: The integration of diverse clinical modalities such as medical imaging and the tabular data obtained by the patients' Electronic Health Records (EHRs) is a crucial aspect of modern healthcare. The integrative analysis of multiple sources can provide a comprehensive understanding of a patient's condition and can enhance diagnoses and treatment decisions. Deep Neural Networks (DNNs) consistently showcase outstanding performance in a wide range of multimodal tasks in the medical domain. However, the complex endeavor of effectively merging medical imaging with clinical, demographic and genetic information represented as numerical tabular data remains a highly active and ongoing research pursuit.   We present a novel framework based on hypernetworks to fuse clinical imaging and tabular data by conditioning the image processing on the EHR's values and measurements. This approach aims to leverage the complementary information present in these
    
[^5]: 使用可学习提示从文本到图像生成模型中去除不良概念

    Removing Undesirable Concepts in Text-to-Image Generative Models with Learnable Prompts

    [https://arxiv.org/abs/2403.12326](https://arxiv.org/abs/2403.12326)

    通过引入可学习提示到交叉注意力模块中，本文提出了一种新方法，用于从文本到图像生成模型中去除不良概念，实现了对模型效果的提升。

    

    生成模型已经展示出在从文本描述中生成视觉上令人印象深刻的内容方面具有显著潜力。然而，在未经筛选的互联网数据上训练这些模型存在学习和随后传播不良概念（如受版权保护或不道德内容）的风险。在本文中，我们提出了一种新方法，通过将可学习提示结合到交叉注意力模块中，从文本到图像生成模型中去除不良概念。这可学习提示充当附加内存，将不良概念的知识转移到其中，并减少这些概念对模型参数和相应文本输入的依赖。由于这种知识转移到提示中，消除这些不良概念更加稳定，并对其他概念影响最小。我们在稳定扩散模型上展示了我们方法的有效性，展示了其优势。

    arXiv:2403.12326v1 Announce Type: new  Abstract: Generative models have demonstrated remarkable potential in generating visually impressive content from textual descriptions. However, training these models on unfiltered internet data poses the risk of learning and subsequently propagating undesirable concepts, such as copyrighted or unethical content. In this paper, we propose a novel method to remove undesirable concepts from text-to-image generative models by incorporating a learnable prompt into the cross-attention module. This learnable prompt acts as additional memory to transfer the knowledge of undesirable concepts into it and reduce the dependency of these concepts on the model parameters and corresponding textual inputs. Because of this knowledge transfer into the prompt, erasing these undesirable concepts is more stable and has minimal negative impact on other concepts. We demonstrate the effectiveness of our method on the Stable Diffusion model, showcasing its superiority ov
    
[^6]: 神经槽解释器：在新兴的槽表示中接地对象语义

    Neural Slot Interpreters: Grounding Object Semantics in Emergent Slot Representations

    [https://arxiv.org/abs/2403.07887](https://arxiv.org/abs/2403.07887)

    提出了神经槽解释器（NSI），通过槽表示学习接地和生成物体语义，实现了将现实世界的物体语义结合到抽象中。

    

    物体中心方法在将原始感知无监督分解为丰富的类似物体的抽象方面取得了重大进展。然而，将现实世界的物体语义接地到学到的抽象中的能力有限，这阻碍了它们在下游理解应用中的采用。我们提出神经槽解释器（NSI），它通过槽表示学习接地和生成物体语义。NSI的核心是一种类似XML的编程语言，它使用简单的语法规则将场景的物体语义组织成以物体为中心的程序原语。然后，一个对齐模型学习通过共享嵌入空间上的双层对比学习目标将程序原语接地到槽。最后，我们构建NSI程序生成模型，利用对齐模型推断的密集关联从槽生成以物体为中心的程序。在双模式检索实验中，

    arXiv:2403.07887v1 Announce Type: cross  Abstract: Object-centric methods have seen significant progress in unsupervised decomposition of raw perception into rich object-like abstractions. However, limited ability to ground object semantics of the real world into the learned abstractions has hindered their adoption in downstream understanding applications. We present the Neural Slot Interpreter (NSI) that learns to ground and generate object semantics via slot representations. At the core of NSI is an XML-like programming language that uses simple syntax rules to organize the object semantics of a scene into object-centric program primitives. Then, an alignment model learns to ground program primitives into slots through a bi-level contrastive learning objective over a shared embedding space. Finally, we formulate the NSI program generator model to use the dense associations inferred from the alignment model to generate object-centric programs from slots. Experiments on bi-modal retrie
    
[^7]: 您需要更好地关注付费

    You Need to Pay Better Attention

    [https://arxiv.org/abs/2403.01643](https://arxiv.org/abs/2403.01643)

    提出了三种新的注意力机制，在效率和学习能力方面优于标准的多头注意力，提高了Transformer模型的性能和更广泛的部署能力。

    

    我们引入了三种新的注意力机制，这些机制在效率和学习能力方面胜过标准的多头注意力，从而提高了Transformer模型的性能和更广泛的部署能力。我们的第一个贡献是优化注意力，其性能与标准注意力相似，但参数数量少了四分之三，每个头部少了一个矩阵乘法。接下来，我们引入了高效注意力，其性能与标准注意力相当，但参数数量减少了一半，每个头部减少了两个矩阵乘法，并且比标准注意力快两倍。最后，我们介绍了超级注意力，在视觉和自然语言处理任务中明显超越了标准注意力，同时具有更少的参数和矩阵乘法。除了提供严格的数学比较，我们在MN中评估了所提出的注意力机制

    arXiv:2403.01643v1 Announce Type: cross  Abstract: We introduce three new attention mechanisms that outperform standard multi-head attention in terms of efficiency and learning capabilities, thereby improving the performance and broader deployability of Transformer models. Our first contribution is Optimised Attention, which performs similarly to standard attention, but has 3/4 as many parameters and one matrix multiplication fewer per head. Next, we introduce Efficient Attention, which performs on par with standard attention with only 1/2 as many parameters as many parameters and two matrix multiplications fewer per head and is up to twice as fast as standard attention. Lastly, we introduce Super Attention, which surpasses standard attention by a significant margin in both vision and natural language processing tasks while having fewer parameters and matrix multiplications. In addition to providing rigorous mathematical comparisons, we evaluate the presented attention mechanisms on MN
    
[^8]: 从API学习学习：黑盒数据无关元学习

    Learning to Learn from APIs: Black-Box Data-Free Meta-Learning. (arXiv:2305.18413v1 [cs.LG])

    [http://arxiv.org/abs/2305.18413](http://arxiv.org/abs/2305.18413)

    该论文提出了一个BiDf-MKD框架，可以从一组API库中无需访问训练数据，直接进行元学习；能够在更广泛的黑盒API上进行元学习，提高了元模型的泛化性能和应用范围。

    

    无数据元学习（DFML）旨在通过从一组预训练模型进行元学习而无需访问训练数据，从而实现高效学习新任务。现有的DFML工作仅能从（i）白盒和（ii）小规模预训练模型（iii）相同的架构中元学习，忽略了更实际的设置，即用户仅能通过任意模型架构和规模的API进行推断。为解决这个问题，我们提出了一个双层数据无关元知识蒸馏（BiDf-MKD）框架，将更通用的元知识从一组黑盒API转移到一个单一的元模型中。

    Data-free meta-learning (DFML) aims to enable efficient learning of new tasks by meta-learning from a collection of pre-trained models without access to the training data. Existing DFML work can only meta-learn from (i) white-box and (ii) small-scale pre-trained models (iii) with the same architecture, neglecting the more practical setting where the users only have inference access to the APIs with arbitrary model architectures and model scale inside. To solve this issue, we propose a Bi-level Data-free Meta Knowledge Distillation (BiDf-MKD) framework to transfer more general meta knowledge from a collection of black-box APIs to one single meta model. Specifically, by just querying APIs, we inverse each API to recover its training data via a zero-order gradient estimator and then perform meta-learning via a novel bi-level meta knowledge distillation structure, in which we design a boundary query set recovery technique to recover a more informative query set near the decision boundary. 
    
[^9]: GPT-NAS: 以生成式预训练模型为基础的神经架构搜索

    GPT-NAS: Neural Architecture Search with the Generative Pre-Trained Model. (arXiv:2305.05351v1 [cs.CV] CROSS LISTED)

    [http://arxiv.org/abs/2305.05351](http://arxiv.org/abs/2305.05351)

    GPT-NAS使用生成式预训练模型优化神经架构搜索，通过提出近似的架构组件减小搜索空间，并明显优于其他NAS方法。

    

    神经架构搜索(NAS)已经成为了一种自动设计最优神经网络架构的有效方法之一。虽然一些人工设计的神经网络已经在多项任务中取得了人类水平的表现，但在NAS方法中很少出现这类成果，主要原因在于神经架构的搜索空间太大了，导致NAS算法效率低下。这项工作提出了一种新的架构搜索算法，称为GPT-NAS，通过生成式预训练模型来优化神经架构。在GPT-NAS中，我们假设一个在大规模语料库上预训练的生成模型能够学习构建神经架构的基本规律。因此，GPT-NAS利用生成式预训练模型来提出合理的架构组件，从而大大减少了搜索空间，引入了搜索过程中的先验知识。广泛的实验结果表明，我们的GPT-NAS方法明显优于其他NAS方法。

    Neural Architecture Search (NAS) has emerged as one of the effective methods to design the optimal neural network architecture automatically. Although neural architectures have achieved human-level performances in several tasks, few of them are obtained from the NAS method. The main reason is the huge search space of neural architectures, making NAS algorithms inefficient. This work presents a novel architecture search algorithm, called GPT-NAS, that optimizes neural architectures by Generative Pre-Trained (GPT) model. In GPT-NAS, we assume that a generative model pre-trained on a large-scale corpus could learn the fundamental law of building neural architectures. Therefore, GPT-NAS leverages the generative pre-trained (GPT) model to propose reasonable architecture components given the basic one. Such an approach can largely reduce the search space by introducing prior knowledge in the search process. Extensive experimental results show that our GPT-NAS method significantly outperforms
    
[^10]: 不受体系结构、数据集和模型规模限制的无数据元学习

    Architecture, Dataset and Model-Scale Agnostic Data-free Meta-Learning. (arXiv:2303.11183v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.11183](http://arxiv.org/abs/2303.11183)

    不受体系结构、数据集和模型规模限制的无数据元学习框架PURER，通过ECI执行伪周期训练以适应新的任务，通过ICFIL对反演梯度进行校准来优化反演过程，并在各种任务中显著优于现有方法。

    

    无数据元学习的目的是从一组经过预训练的模型中学习有用的先验知识，而无需访问其训练数据。然而，现有的研究仅在参数空间中解决了该问题，忽略了预训练模型中蕴含的丰富数据知识，无法扩展到大规模预训练模型，只能元学习具有相同网络架构的预训练模型。为了解决这些问题，我们提出了一个统一的框架——PURER，其中包含：（1）数据无关的元训练期间的节目课程反转（ECI）；（2）元测试期间内部循环后的反演校准（ICFIL）。在元训练期间，我们提出了ECI来执行伪周期训练，以便快速适应新的看不见的任务。在元测试期间，我们提出了ICFIL来校准反演梯度，以减少基于反演的优化的负面影响。广泛的实验结果表明，所提出的PURER可以有效地元学习来自具有不同网络架构、数据集域甚至不同大小的预训练模型，并在各种任务中显著优于现有方法。

    The goal of data-free meta-learning is to learn useful prior knowledge from a collection of pre-trained models without accessing their training data. However, existing works only solve the problem in parameter space, which (i) ignore the fruitful data knowledge contained in the pre-trained models; (ii) can not scale to large-scale pre-trained models; (iii) can only meta-learn pre-trained models with the same network architecture. To address those issues, we propose a unified framework, dubbed PURER, which contains: (1) ePisode cUrriculum inveRsion (ECI) during data-free meta training; and (2) invErsion calibRation following inner loop (ICFIL) during meta testing. During meta training, we propose ECI to perform pseudo episode training for learning to adapt fast to new unseen tasks. Specifically, we progressively synthesize a sequence of pseudo episodes by distilling the training data from each pre-trained model. The ECI adaptively increases the difficulty level of pseudo episodes accord
    
[^11]: 人类对神经网络表示的对齐

    Human alignment of neural network representations. (arXiv:2211.01201v4 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2211.01201](http://arxiv.org/abs/2211.01201)

    本文研究神经网络表示与人类心理表示之间的对齐问题，发现模型规模和体系结构对对齐几乎没有影响，而训练数据集和目标函数都对对齐有很大的影响。从一个数据集中学习的神经网络表示的线性变换能显著提高对另外两个数据集中人类相似性判断的对齐性。

    

    当今的计算机视觉模型在各种视觉任务上实现了人类或接近人类水平的性能。然而，它们的体系结构、数据和学习算法与导致人类视觉的方式存在许多不同之处。本文研究影响神经网络所学习的表示与通过行为反应推断出的人类心理表示之间对齐的因素。我们发现，模型的规模和体系结构对与人类行为反应的对齐基本上没有影响，而训练数据集和目标函数则具有更大的影响。这些发现在使用两种不同任务收集的三个人类相似度判断数据集中保持一致。从一个数据集中学习的神经网络表示的线性变换显著提高了对另外两个数据集中的人类相似度判断的对齐性。此外，我们发现，一些人类概念...

    Today's computer vision models achieve human or near-human level performance across a wide variety of vision tasks. However, their architectures, data, and learning algorithms differ in numerous ways from those that give rise to human vision. In this paper, we investigate the factors that affect the alignment between the representations learned by neural networks and human mental representations inferred from behavioral responses. We find that model scale and architecture have essentially no effect on the alignment with human behavioral responses, whereas the training dataset and objective function both have a much larger impact. These findings are consistent across three datasets of human similarity judgments collected using two different tasks. Linear transformations of neural network representations learned from behavioral responses from one dataset substantially improve alignment with human similarity judgments on the other two datasets. In addition, we find that some human concept
    

