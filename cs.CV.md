# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Taming Cross-Domain Representation Variance in Federated Prototype Learning with Heterogeneous Data Domains](https://arxiv.org/abs/2403.09048) | 引入FedPLVM通过建立方差感知的双层原型聚类和使用新型$\alpha$-稀疏原型损失，以减少跨领域特征表示差异。 |
| [^2] | [Veagle: Advancements in Multimodal Representation Learning](https://arxiv.org/abs/2403.08773) | 本文介绍了一种新颖的方法，通过在当前视觉语言模型（VLMs）和多模态大语言模型（MLLMs）的基础上融合独特的机制，以增强现有模型的多模态能力。 |
| [^3] | [Evaluating Text-to-Image Generative Models: An Empirical Study on Human Image Synthesis](https://arxiv.org/abs/2403.05125) | 本文提出了一个细致的评估框架，用于评估文本到图像生成模型，针对人类图像合成。我们引入了一个创新的美学分数预测模型，评估生成图像的视觉吸引力，并展示了第一个标记有生成的人类图像中低质量区域的数据集，以促进自动缺陷检测，同时也研究了模型对概念覆盖度和公平性的影响。 |
| [^4] | [Effectiveness Assessment of Recent Large Vision-Language Models](https://arxiv.org/abs/2403.04306) | 本文评估了最近出现的大型视觉-语言模型在专业和通用任务中的表现，旨在全面了解这些创新方法的能力。 |
| [^5] | [Leveraging Self-Supervised Learning for Scene Recognition in Child Sexual Abuse Imagery](https://arxiv.org/abs/2403.01183) | 利用自监督学习技术，本文提出了一种能够安全高效处理儿童性虐待图像数据的场景识别方法。 |
| [^6] | [Adversarial Robustness Through Artifact Design](https://arxiv.org/abs/2402.04660) | 该研究提出了一种通过艺术设计实现对抗性鲁棒性的方法，通过微小更改现有规范来抵御对抗性示例的影响。 |
| [^7] | [On f-Divergence Principled Domain Adaptation: An Improved Framework](https://arxiv.org/abs/2402.01887) | 本文改进了基于f-散度的无监督领域自适应（UDA）框架，引入了f-领域差异度量指标，并通过去除绝对值函数和引入缩放参数，提出了新的目标误差和样本复杂度界限，从而使得我们能够恢复以前的KL结果，将算法和理论之间的差距缩小，并通过定位技术开发了快速率的泛化界限。实验结果证明了基于f-DD的领域学习算法在流行的UDA基准测试中表现出了卓越的性能。 |
| [^8] | [Immunohistochemistry guided segmentation of benign epithelial cells, in situ lesions, and invasive epithelial cells in breast cancer slides](https://arxiv.org/abs/2311.13261) | 该研究旨在开发一个用于乳腺癌切片中上皮细胞分割的AI模型，通过免疫组织化学引导分割出良性上皮细胞、原位病变和浸润性上皮细胞。 |
| [^9] | [Robust Image Watermarking using Stable Diffusion.](http://arxiv.org/abs/2401.04247) | 本研究提出了一种名为ZoDiac的方法，利用稳定扩散模型在可训练的潜在空间中注入水印，从而使水印能够在受攻击时可靠检测到，对最先进的水印攻击具有很强的鲁棒性，优于现有的水印方法。 |
| [^10] | [Decoupled Kullback-Leibler Divergence Loss.](http://arxiv.org/abs/2305.13948) | 本文提出了改进的KL散度损失函数，通过解决解耦式KL散度损失函数的对称性限制和引入全局信息来提升性能，在CIFAR-10/100和ImageNet数据集上展示了其在对抗训练和知识蒸馏任务中的优越表现。 |
| [^11] | [Adversarial robustness of VAEs through the lens of local geometry.](http://arxiv.org/abs/2208.03923) | 本文证明了对手攻击VAEs的最佳方法是利用由编码器和解码器网络引起的随机回溯度规张量的方向偏差。 |

# 详细

[^1]: 驯服异构数据域中联邦原型学习中的跨领域表示差异

    Taming Cross-Domain Representation Variance in Federated Prototype Learning with Heterogeneous Data Domains

    [https://arxiv.org/abs/2403.09048](https://arxiv.org/abs/2403.09048)

    引入FedPLVM通过建立方差感知的双层原型聚类和使用新型$\alpha$-稀疏原型损失，以减少跨领域特征表示差异。

    

    联邦学习（FL）允许在不共享私人数据的情况下进行协作机器学习训练。虽然大多数FL方法假设客户端之间具有相同的数据领域，但现实场景中通常涉及异构数据领域。联邦原型学习（FedPL）解决了这个问题，使用平均特征向量作为原型来增强模型泛化能力。然而，现有的FedPL方法为每个客户端创建相同数量的原型，导致跨领域性能差距，并使数据分布不同的客户端存在差异。为了减轻跨领域特征表示差异，我们引入了FedPLVM，它建立了方差感知的双层原型聚类，并采用了一种新颖的$\alpha$-稀疏原型损失。

    arXiv:2403.09048v1 Announce Type: new  Abstract: Federated learning (FL) allows collaborative machine learning training without sharing private data. While most FL methods assume identical data domains across clients, real-world scenarios often involve heterogeneous data domains. Federated Prototype Learning (FedPL) addresses this issue, using mean feature vectors as prototypes to enhance model generalization. However, existing FedPL methods create the same number of prototypes for each client, leading to cross-domain performance gaps and disparities for clients with varied data distributions. To mitigate cross-domain feature representation variance, we introduce FedPLVM, which establishes variance-aware dual-level prototypes clustering and employs a novel $\alpha$-sparsity prototype loss. The dual-level prototypes clustering strategy creates local clustered prototypes based on private data features, then performs global prototypes clustering to reduce communication complexity and pres
    
[^2]: Veagle: 多模态表示学习的进展

    Veagle: Advancements in Multimodal Representation Learning

    [https://arxiv.org/abs/2403.08773](https://arxiv.org/abs/2403.08773)

    本文介绍了一种新颖的方法，通过在当前视觉语言模型（VLMs）和多模态大语言模型（MLLMs）的基础上融合独特的机制，以增强现有模型的多模态能力。

    

    最近，人工智能领域的研究人员对语言和视觉如何结合产生了浓厚兴趣，从而催生了旨在无缝整合文本和视觉信息的多模态模型的发展。多模态模型是大型语言模型（LLMs）的延伸，在解决各种任务方面展现出了显著的能力，范围从图像字幕和视觉问答（VQA）到视觉定位。虽然这些模型展示了显著的进展，但在准确解释图像并回答问题方面仍存在挑战，在现实场景中经常发生。本文介绍了一种增强现有模型多模态能力的新方法。针对当前视觉语言模型（VLMs）和多模态大语言模型（MLLMs）中观察到的局限性，我们提出的模型Veagle，融合了受...

    arXiv:2403.08773v1 Announce Type: cross  Abstract: Lately, researchers in artificial intelligence have been really interested in how language and vision come together, giving rise to the development of multimodal models that aim to seamlessly integrate textual and visual information. Multimodal models, an extension of Large Language Models (LLMs), have exhibited remarkable capabilities in addressing a diverse array of tasks, ranging from image captioning and visual question answering (VQA) to visual grounding. While these models have showcased significant advancements, challenges persist in accurately interpreting images and answering the question, a common occurrence in real-world scenarios. This paper introduces a novel approach to enhance the multimodal capabilities of existing models. In response to the limitations observed in current Vision Language Models (VLMs) and Multimodal Large Language Models (MLLMs), our proposed model Veagle, incorporates a unique mechanism inspired by th
    
[^3]: 评估文本到图像生成模型：关于人类图像合成的经验性研究

    Evaluating Text-to-Image Generative Models: An Empirical Study on Human Image Synthesis

    [https://arxiv.org/abs/2403.05125](https://arxiv.org/abs/2403.05125)

    本文提出了一个细致的评估框架，用于评估文本到图像生成模型，针对人类图像合成。我们引入了一个创新的美学分数预测模型，评估生成图像的视觉吸引力，并展示了第一个标记有生成的人类图像中低质量区域的数据集，以促进自动缺陷检测，同时也研究了模型对概念覆盖度和公平性的影响。

    

    在本文中，我们提出了一个细致的评估框架，用于评估文本到图像（T2I）生成模型，应用于人类图像合成。我们的框架将评估分为两个不同的方面：第一，专注于图像质量，如美学和逼真度；第二，通过概念覆盖度和公平性来检查文本条件。我们引入了一种创新的美学分数预测模型，评估生成图像的视觉吸引力，并展示了第一个标记有生成的人类图像中低质量区域的数据集，以促进自动缺陷检测。我们对概念覆盖范围的探索调查了模型在准确解释和呈现基于文本的概念方面的有效性，而我们对公平性的分析揭示了模型输出中的偏见，重点关注性别、种族和年龄。虽然我们的研究基于人类图像，但这种双重方面的方法是为了

    arXiv:2403.05125v1 Announce Type: cross  Abstract: In this paper, we present an empirical study introducing a nuanced evaluation framework for text-to-image (T2I) generative models, applied to human image synthesis. Our framework categorizes evaluations into two distinct groups: first, focusing on image qualities such as aesthetics and realism, and second, examining text conditions through concept coverage and fairness. We introduce an innovative aesthetic score prediction model that assesses the visual appeal of generated images and unveils the first dataset marked with low-quality regions in generated human images to facilitate automatic defect detection. Our exploration into concept coverage probes the model's effectiveness in interpreting and rendering text-based concepts accurately, while our analysis of fairness reveals biases in model outputs, with an emphasis on gender, race, and age. While our study is grounded in human imagery, this dual-faceted approach is designed with the 
    
[^4]: 最近大型视觉-语言模型的有效性评估

    Effectiveness Assessment of Recent Large Vision-Language Models

    [https://arxiv.org/abs/2403.04306](https://arxiv.org/abs/2403.04306)

    本文评估了最近出现的大型视觉-语言模型在专业和通用任务中的表现，旨在全面了解这些创新方法的能力。

    

    大型视觉-语言模型(LVLMs)的出现代表着迈向人工通用智能的重要进步。然而，它们在专业和通用任务中的有效性程度需要进一步调查。本文旨在评估流行的LVLMs在专业和通用任务中的能力，旨在提供对这些创新方法的全面理解。为了评估它们在专业任务中的有效性，我们量身定制了一个包含自然、医疗和工业三种不同场景的全面测试平台，涵盖六项具有挑战性的任务。这些任务包括显著、伪装和透明物体检测，以及息肉和皮肤病变检测，以及工业异常检测。我们检验了最近三种开源LVLMs--MiniGPT-v2、LLaVA-1.5和Shikra--在视觉识别和定位领域的表现。

    arXiv:2403.04306v1 Announce Type: cross  Abstract: The advent of large vision-language models (LVLMs) represents a noteworthy advancement towards the pursuit of artificial general intelligence. However, the extent of their efficacy across both specialized and general tasks warrants further investigation. This article endeavors to evaluate the competency of popular LVLMs in specialized and general tasks, respectively, aiming to offer a comprehensive comprehension of these innovative methodologies. To gauge their efficacy in specialized tasks, we tailor a comprehensive testbed comprising three distinct scenarios: natural, healthcare, and industrial, encompassing six challenging tasks. These tasks include salient, camouflaged, and transparent object detection, as well as polyp and skin lesion detection, alongside industrial anomaly detection. We examine the performance of three recent open-source LVLMs -- MiniGPT-v2, LLaVA-1.5, and Shikra -- in the realm of visual recognition and localiza
    
[^5]: 利用自监督学习进行儿童性虐待图像场景识别

    Leveraging Self-Supervised Learning for Scene Recognition in Child Sexual Abuse Imagery

    [https://arxiv.org/abs/2403.01183](https://arxiv.org/abs/2403.01183)

    利用自监督学习技术，本文提出了一种能够安全高效处理儿童性虐待图像数据的场景识别方法。

    

    21世纪的犯罪分为虚拟和真实世界。然而，前者已经成为对后者人们福祉和安全构成全球威胁。它提出的挑战必须通过统一的全球合作来面对，我们必须比以往更加依赖自动化但值得信赖的工具来应对网络犯罪日益增长的本质。每年有超过1000万起儿童性虐待报告提交给美国国家失踪和被剥削儿童中心，超过80%来自网络来源。因此，调查中心和清除中心无法手动处理和正确调查所有图像。基于此，能够安全高效处理这些数据的可靠自动化工具至关重要。在这方面，场景识别任务寻找环境中的上下文线索，能够组织和分类儿童性虐待数据，而无需在敏感数据上进行训练。

    arXiv:2403.01183v1 Announce Type: cross  Abstract: Crime in the 21st century is split into a virtual and real world. However, the former has become a global menace to people's well-being and security in the latter. The challenges it presents must be faced with unified global cooperation, and we must rely more than ever on automated yet trustworthy tools to combat the ever-growing nature of online offenses. Over 10 million child sexual abuse reports are submitted to the US National Center for Missing & Exploited Children every year, and over 80% originated from online sources. Therefore, investigation centers and clearinghouses cannot manually process and correctly investigate all imagery. In light of that, reliable automated tools that can securely and efficiently deal with this data are paramount. In this sense, the scene recognition task looks for contextual cues in the environment, being able to group and classify child sexual abuse data without requiring to be trained on sensitive 
    
[^6]: 通过艺术设计提高对抗性鲁棒性

    Adversarial Robustness Through Artifact Design

    [https://arxiv.org/abs/2402.04660](https://arxiv.org/abs/2402.04660)

    该研究提出了一种通过艺术设计实现对抗性鲁棒性的方法，通过微小更改现有规范来抵御对抗性示例的影响。

    

    对抗性示例的出现给机器学习带来了挑战。为了阻碍对抗性示例，大多数防御方法都改变了模型的训练方式（如对抗性训练）或推理过程（如随机平滑）。尽管这些方法显著提高了模型的对抗性鲁棒性，但模型仍然极易受到对抗性示例的影响。在某些领域如交通标志识别中，我们发现对象是按照规范来设计（如标志规范）。为了改善对抗性鲁棒性，我们提出了一种新颖的方法。具体来说，我们提供了一种重新定义规范的方法，对现有规范进行微小的更改，以防御对抗性示例。我们将艺术设计问题建模为一个鲁棒优化问题，并提出了基于梯度和贪婪搜索的方法来解决它。我们在交通标志识别领域对我们的方法进行了评估，使其能够改变交通标志中的象形图标（即标志内的符号）。

    Adversarial examples arose as a challenge for machine learning. To hinder them, most defenses alter how models are trained (e.g., adversarial training) or inference is made (e.g., randomized smoothing). Still, while these approaches markedly improve models' adversarial robustness, models remain highly susceptible to adversarial examples. Identifying that, in certain domains such as traffic-sign recognition, objects are implemented per standards specifying how artifacts (e.g., signs) should be designed, we propose a novel approach for improving adversarial robustness. Specifically, we offer a method to redefine standards, making minor changes to existing ones, to defend against adversarial examples. We formulate the problem of artifact design as a robust optimization problem, and propose gradient-based and greedy search methods to solve it. We evaluated our approach in the domain of traffic-sign recognition, allowing it to alter traffic-sign pictograms (i.e., symbols within the signs) a
    
[^7]: 基于f-散度原理的领域自适应：一个改进的框架

    On f-Divergence Principled Domain Adaptation: An Improved Framework

    [https://arxiv.org/abs/2402.01887](https://arxiv.org/abs/2402.01887)

    本文改进了基于f-散度的无监督领域自适应（UDA）框架，引入了f-领域差异度量指标，并通过去除绝对值函数和引入缩放参数，提出了新的目标误差和样本复杂度界限，从而使得我们能够恢复以前的KL结果，将算法和理论之间的差距缩小，并通过定位技术开发了快速率的泛化界限。实验结果证明了基于f-DD的领域学习算法在流行的UDA基准测试中表现出了卓越的性能。

    

    无监督领域自适应（UDA）在解决机器学习中的分布偏移问题中起着至关重要的作用。在本文中，我们通过改进Acuna等人（2021年）提出的UDA的理论基础，对其基于f-散度的差异度进行了改进，并引入了一个新的度量指标，即f-领域差异（f-DD）。通过去除绝对值函数并引入一个缩放参数，f-DD产生了新的目标误差和样本复杂度界限，使我们能够恢复以前基于KL的结果，并弥合了Acuna等人（2021年）中提出的算法和理论之间的差距。利用定位技术，我们还开发了一种快速率的泛化界限。实证结果表明，在流行的UDA基准测试中，基于f-DD的领域学习算法表现出优越性能。

    Unsupervised domain adaptation (UDA) plays a crucial role in addressing distribution shifts in machine learning. In this work, we improve the theoretical foundations of UDA proposed by Acuna et al. (2021) by refining their f-divergence-based discrepancy and additionally introducing a new measure, f-domain discrepancy (f-DD). By removing the absolute value function and incorporating a scaling parameter, f-DD yields novel target error and sample complexity bounds, allowing us to recover previous KL-based results and bridging the gap between algorithms and theory presented in Acuna et al. (2021). Leveraging a localization technique, we also develop a fast-rate generalization bound. Empirical results demonstrate the superior performance of f-DD-based domain learning algorithms over previous works in popular UDA benchmarks.
    
[^8]: 乳腺癌切片中良性上皮细胞、原位病变和浸润性上皮细胞的免疫组织化学引导分割

    Immunohistochemistry guided segmentation of benign epithelial cells, in situ lesions, and invasive epithelial cells in breast cancer slides

    [https://arxiv.org/abs/2311.13261](https://arxiv.org/abs/2311.13261)

    该研究旨在开发一个用于乳腺癌切片中上皮细胞分割的AI模型，通过免疫组织化学引导分割出良性上皮细胞、原位病变和浸润性上皮细胞。

    

    数字病理学使得可以利用人工智能自动分析组织病理切片。自动评估可以提高诊断效率，并帮助找到形态特征与临床结果之间的关联。为了开发这样的预测模型，辨认浸润性上皮细胞，并将其与良性上皮细胞和原位病变分开将是第一步。在本研究中，我们旨在开发一个用于乳腺癌切片中上皮细胞分割的AI模型。我们通过重新染色血红蛋白和嗜酸性染色细胞角蛋白(CK) AE1/AE3 HE切片，以及病理学家的注释生成了上皮基本真值掩模。HE/CK图像对被用于训练卷积神经网络，数据增强被用来使模型更稳健。839名患者的组织微阵列（TMAs）和两名患者的整张切片图像用于训练

    arXiv:2311.13261v2 Announce Type: replace-cross  Abstract: Digital pathology enables automatic analysis of histopathological sections using artificial intelligence (AI). Automatic evaluation could improve diagnostic efficiency and help find associations between morphological features and clinical outcome. For development of such prediction models, identifying invasive epithelial cells, and separating these from benign epithelial cells and in situ lesions would be the first step. In this study, we aimed to develop an AI model for segmentation of epithelial cells in sections from breast cancer. We generated epithelial ground truth masks by restaining hematoxylin and eosin (HE) sections with cytokeratin (CK) AE1/AE3, and by pathologists' annotations. HE/CK image pairs were used to train a convolutional neural network, and data augmentation was used to make the model more robust. Tissue microarrays (TMAs) from 839 patients, and whole slide images from two patients were used for training an
    
[^9]: 稳定扩散技术的鲁棒图像水印

    Robust Image Watermarking using Stable Diffusion. (arXiv:2401.04247v1 [cs.CV])

    [http://arxiv.org/abs/2401.04247](http://arxiv.org/abs/2401.04247)

    本研究提出了一种名为ZoDiac的方法，利用稳定扩散模型在可训练的潜在空间中注入水印，从而使水印能够在受攻击时可靠检测到，对最先进的水印攻击具有很强的鲁棒性，优于现有的水印方法。

    

    图像水印对于追踪图像来源和声明所有权非常重要。随着生成模型（如稳定扩散）的出现，能够创建虚假但逼真的图像，水印成为了尤为重要的任务，例如使生成的图像可靠地辨认出来。然而，正是这种稳定扩散技术可以移除使用现有方法注入的水印。为了解决这个问题，我们提出了一种名为ZoDiac的方法，它使用预训练的稳定扩散模型将水印注入到可训练的潜在空间中，从而在受攻击时仍然可以可靠地在潜在向量中检测到水印。我们在三个基准数据集 MS-COCO、DiffusionDB 和 WikiArt 上评估了 ZoDiac，并发现 ZoDiac 对于最先进的水印攻击具有很强的鲁棒性，水印检测率超过98%，误报率低于6.4%，优于现有的水印方法。我们的研究表明，稳定扩散是一种有前途的方法。

    Watermarking images is critical for tracking image provenance and claiming ownership. With the advent of generative models, such as stable diffusion, able to create fake but realistic images, watermarking has become particularly important, e.g., to make generated images reliably identifiable. Unfortunately, the very same stable diffusion technology can remove watermarks injected using existing methods. To address this problem, we present a ZoDiac, which uses a pre-trained stable diffusion model to inject a watermark into the trainable latent space, resulting in watermarks that can be reliably detected in the latent vector, even when attacked. We evaluate ZoDiac on three benchmarks, MS-COCO, DiffusionDB, and WikiArt, and find that ZoDiac is robust against state-of-the-art watermark attacks, with a watermark detection rate over 98% and a false positive rate below 6.4%, outperforming state-of-the-art watermarking methods. Our research demonstrates that stable diffusion is a promising appr
    
[^10]: 解耦式KL散度损失函数

    Decoupled Kullback-Leibler Divergence Loss. (arXiv:2305.13948v1 [cs.CV])

    [http://arxiv.org/abs/2305.13948](http://arxiv.org/abs/2305.13948)

    本文提出了改进的KL散度损失函数，通过解决解耦式KL散度损失函数的对称性限制和引入全局信息来提升性能，在CIFAR-10/100和ImageNet数据集上展示了其在对抗训练和知识蒸馏任务中的优越表现。

    

    本文更深入地探究了KL散度损失函数，并发现它与解耦式KL散度损失函数等价，后者由加权均方差损失和包含软标签的交叉熵损失组成。通过对解耦式KL散度损失函数的分析，本文确定了两个改进方向。首先，我们解决了在知识蒸馏等场景下解耦式KL散度损失函数的对称性限制问题。这个改进保证了在训练期间wMSE组件始终有效，提供额外的构造性暗示。其次，我们将全局信息引入解耦式KL散度损失函数中，用于类内一致性正则化。通过这两个改进，我们得到了改进的KL散度损失函数，通过在CIFAR-10/100和ImageNet数据集上进行实验来评估其有效性，重点是对抗训练和知识蒸馏任务。所提出的方法表现出了比其他最先进模型更优越的性能，展示了其在各种实际应用中的潜力。

    In this paper, we delve deeper into the Kullback-Leibler (KL) Divergence loss and observe that it is equivalent to the Doupled Kullback-Leibler (DKL) Divergence loss that consists of 1) a weighted Mean Square Error (wMSE) loss and 2) a Cross-Entropy loss incorporating soft labels. From our analysis of the DKL loss, we have identified two areas for improvement. Firstly, we address the limitation of DKL in scenarios like knowledge distillation by breaking its asymmetry property in training optimization. This modification ensures that the wMSE component is always effective during training, providing extra constructive cues. Secondly, we introduce global information into DKL for intra-class consistency regularization. With these two enhancements, we derive the Improved Kullback-Leibler (IKL) Divergence loss and evaluate its effectiveness by conducting experiments on CIFAR-10/100 and ImageNet datasets, focusing on adversarial training and knowledge distillation tasks. The proposed approach 
    
[^11]: 通过本地几何角度理解VAEs的对抗鲁棒性

    Adversarial robustness of VAEs through the lens of local geometry. (arXiv:2208.03923v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2208.03923](http://arxiv.org/abs/2208.03923)

    本文证明了对手攻击VAEs的最佳方法是利用由编码器和解码器网络引起的随机回溯度规张量的方向偏差。

    

    在对变分自编码器（VAEs）进行无监督攻击时，对手会找到一个输入样本中的小扰动，从而显着改变其潜在空间编码，从而损害了一个固定编码器的重构。这种脆弱性已知的原因是潜在后验分布的近似与先验分布之间的不匹配导致的潜在空间扭曲。因此，输入样本中的微小变化可能会将其编码移动到潜在空间中的低/零密度区域，从而产生无限制的生成。本文证明了对手攻击VAEs的最佳方法是利用由编码器和解码器网络引起的随机回溯度规张量的方向偏差。编码器的回溯度规张量测量它从输入到潜在空间的微小潜在体积的变化。因此，它可以被视为分析输入扰动导致潜在空间扭曲效果的镜头。

    In an unsupervised attack on variational autoencoders (VAEs), an adversary finds a small perturbation in an input sample that significantly changes its latent space encoding, thereby compromising the reconstruction for a fixed decoder. A known reason for such vulnerability is the distortions in the latent space resulting from a mismatch between approximated latent posterior and a prior distribution. Consequently, a slight change in an input sample can move its encoding to a low/zero density region in the latent space resulting in an unconstrained generation. This paper demonstrates that an optimal way for an adversary to attack VAEs is to exploit a directional bias of a stochastic pullback metric tensor induced by the encoder and decoder networks. The pullback metric tensor of an encoder measures the change in infinitesimal latent volume from an input to a latent space. Thus, it can be viewed as a lens to analyse the effect of input perturbations leading to latent space distortions. We
    

