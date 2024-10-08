# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generalized Consistency Trajectory Models for Image Manipulation](https://arxiv.org/abs/2403.12510) | 本研究提出了广义一致性轨迹模型（GCTMs），能够在任何噪声分布和数据分布之间实现转换。 |
| [^2] | [SELFI: Autonomous Self-Improvement with Reinforcement Learning for Social Navigation](https://arxiv.org/abs/2403.00991) | SELFI提出了一种在线学习方法，通过将在线无模型强化学习与离线基于模型的学习相结合，实现了机器人行为的快速改进，并在避撞和社交遵从行为方面取得了显著进展。 |
| [^3] | [Spectrum Extraction and Clipping for Implicitly Linear Layers](https://arxiv.org/abs/2402.16017) | 展示自动微分在计算和控制隐式线性算子频谱中的有效性；提供第一个适用于一般卷积层的裁剪方法；研究了批归一化层与卷积层组合的效果；通过比较算法与最先进方法的精度和性能表明更精确和高效。 |
| [^4] | [Large-Scale Actionless Video Pre-Training via Discrete Diffusion for Efficient Policy Learning](https://arxiv.org/abs/2402.14407) | 利用离散扩散结合生成式预训练和少量机器人视频微调，实现从人类视频到机器人策略学习的知识迁移。 |
| [^5] | [Switch EMA: A Free Lunch for Better Flatness and Sharpness](https://arxiv.org/abs/2402.09240) | 本研究提出了一种称为Switch EMA（SEMA）的方法，通过简单的修改指数移动平均（EMA）参数，可以帮助深度神经网络（DNN）达到更好的平坦性和锐度的概括最优解。通过在多个任务和数据集上进行实验证明了SEMA的有效性。 |
| [^6] | [Learning Contrastive Feature Representations for Facial Action Unit Detection](https://arxiv.org/abs/2402.06165) | 这项研究提出了一种对比学习框架，通过监督和自监督信号来增强面部动作单元检测模型的性能。采用正样本抽样和权衡重要性的损失函数来应对噪声AU标签和AU类型分布不平衡的挑战。 |
| [^7] | [NoSENSE: Learned unrolled cardiac MRI reconstruction without explicit sensitivity maps.](http://arxiv.org/abs/2309.15608) | 本文提出了一种无需显式灵敏度图的展开心脏MRI重建方法，使用深度卷积神经网络和算法展开，通过学习图像之间的接收线圈关系来实现加速心脏MRI重建，在实验中取得了较好的性能。 |
| [^8] | [Attention-Driven Multi-Modal Fusion: Enhancing Sign Language Recognition and Translation.](http://arxiv.org/abs/2309.01860) | 本文提出了一种注意力驱动的多模态融合机制，通过将光流信息与RGB图像相结合，丰富了连续手语识别和翻译流程中的特征。该方法在手语识别任务中降低了WER 0.9，在翻译任务中提高了测试集上大多数BLEU分数约0.6。 |
| [^9] | [Classification of Blood Cells Using Deep Learning Models.](http://arxiv.org/abs/2308.06300) | 这项研究使用深度学习模型通过图像分类对人类血细胞进行了分类和识别，为诊断疾病提供了重要的帮助。 |
| [^10] | [Encode-Store-Retrieve: Enhancing Memory Augmentation through Language-Encoded Egocentric Perception.](http://arxiv.org/abs/2308.05822) | 本研究提出了一种记忆增强系统，它利用自然语言编码视频数据并将其存储在向量数据库中，通过利用大型视觉语言模型的强大功能来进行语言编码的过程。 |
| [^11] | [CST-YOLO: A Novel Method for Blood Cell Detection Based on Improved YOLOv7 and CNN-Swin Transformer.](http://arxiv.org/abs/2306.14590) | 本论文提出了一种CST-YOLO算法，基于改进的YOLOv7和CNN-Swin Transformer，引入了几个有用的模型，有效提高了血细胞检测精度，实验结果显示其在三个血细胞数据集上均优于现有最先进算法。 |

# 详细

[^1]: 图像操作的广义一致性轨迹模型

    Generalized Consistency Trajectory Models for Image Manipulation

    [https://arxiv.org/abs/2403.12510](https://arxiv.org/abs/2403.12510)

    本研究提出了广义一致性轨迹模型（GCTMs），能够在任何噪声分布和数据分布之间实现转换。

    

    基于扩散的生成模型在无条件生成以及图像编辑和恢复等应用任务中表现出色。扩散模型的成功在于扩散的迭代性质：扩散将将噪声到数据的复杂映射过程分解为一系列简单的去噪任务。此外，通过在每个去噪步骤中注入引导项，我们能够对生成过程进行精细控制。然而，迭代过程也常常计算密集，通常需要进行数十次甚至数千次函数评估。虽然一致性轨迹模型（CTMs）可以在概率流ODE（PFODE）上任意时间点之间进行遍历，并且通过单次函数评估进行得分推导，但CTMs仅允许从高斯噪声转换为数据。因此，本文旨在通过提出广义CTMs（GCTMs）来发挥CTMs的全部潜力，实现在任何噪声分布和数据分布之间进行转换。

    arXiv:2403.12510v1 Announce Type: cross  Abstract: Diffusion-based generative models excel in unconditional generation, as well as on applied tasks such as image editing and restoration. The success of diffusion models lies in the iterative nature of diffusion: diffusion breaks down the complex process of mapping noise to data into a sequence of simple denoising tasks. Moreover, we are able to exert fine-grained control over the generation process by injecting guidance terms into each denoising step. However, the iterative process is also computationally intensive, often taking from tens up to thousands of function evaluations. Although consistency trajectory models (CTMs) enable traversal between any time points along the probability flow ODE (PFODE) and score inference with a single function evaluation, CTMs only allow translation from Gaussian noise to data. Thus, this work aims to unlock the full potential of CTMs by proposing generalized CTMs (GCTMs), which translate between arbit
    
[^2]: SELFI: 利用强化学习实现自主自我改进以进行社交导航

    SELFI: Autonomous Self-Improvement with Reinforcement Learning for Social Navigation

    [https://arxiv.org/abs/2403.00991](https://arxiv.org/abs/2403.00991)

    SELFI提出了一种在线学习方法，通过将在线无模型强化学习与离线基于模型的学习相结合，实现了机器人行为的快速改进，并在避撞和社交遵从行为方面取得了显著进展。

    

    自主自我改进的机器人通过与环境互动和经验积累来实现将是机器人系统在现实世界中投入使用的关键。本文提出了一种在线学习方法SELFI，利用在线机器人经验来快速高效地微调预训练的控制策略。SELFI将在线无模型强化学习应用于离线基于模型的学习之上，以发挥这两种学习范式的优点。具体来说，SELFI通过将离线预训练的模型学习目标与在线无模型强化学习中学习到的Q值相结合，稳定了在线学习过程。我们在多个现实环境中评估了SELFI，并报告了在避撞方面的改善，以及通过人类用户研究测量的更具社交遵从行为。SELFI使我们能够快速学习有用的机器人行为，减少了预先干预的人员干预。

    arXiv:2403.00991v1 Announce Type: cross  Abstract: Autonomous self-improving robots that interact and improve with experience are key to the real-world deployment of robotic systems. In this paper, we propose an online learning method, SELFI, that leverages online robot experience to rapidly fine-tune pre-trained control policies efficiently. SELFI applies online model-free reinforcement learning on top of offline model-based learning to bring out the best parts of both learning paradigms. Specifically, SELFI stabilizes the online learning process by incorporating the same model-based learning objective from offline pre-training into the Q-values learned with online model-free reinforcement learning. We evaluate SELFI in multiple real-world environments and report improvements in terms of collision avoidance, as well as more socially compliant behavior, measured by a human user study. SELFI enables us to quickly learn useful robotic behaviors with less human interventions such as pre-e
    
[^3]: 隐式线性层的频谱提取和裁剪

    Spectrum Extraction and Clipping for Implicitly Linear Layers

    [https://arxiv.org/abs/2402.16017](https://arxiv.org/abs/2402.16017)

    展示自动微分在计算和控制隐式线性算子频谱中的有效性；提供第一个适用于一般卷积层的裁剪方法；研究了批归一化层与卷积层组合的效果；通过比较算法与最先进方法的精度和性能表明更精确和高效。

    

    我们展示了自动微分在高效准确计算和控制隐式线性算子的频谱上的有效性，这是一类包括所有标准卷积和全连接层的丰富层类型。我们提供了第一个适用于一般卷积层的裁剪方法，并阐明了导致之前工作中正确性问题的表示限制。我们研究了批归一化层与卷积层串联时的效果，并展示了我们的裁剪方法如何应用于它们的组合。通过对我们的算法与最先进方法的精度和性能进行比较，我们使用各种实验表明它们更精确和高效，能够实现更好的泛化和对抗鲁棒性。我们提供了使用我们方法的代码链接https://github.com/Ali-E/FastClip。

    arXiv:2402.16017v1 Announce Type: new  Abstract: We show the effectiveness of automatic differentiation in efficiently and correctly computing and controlling the spectrum of implicitly linear operators, a rich family of layer types including all standard convolutional and dense layers. We provide the first clipping method which is correct for general convolution layers, and illuminate the representational limitation that caused correctness issues in prior work. We study the effect of the batch normalization layers when concatenated with convolutional layers and show how our clipping method can be applied to their composition. By comparing the accuracy and performance of our algorithms to the state-of-the-art methods, using various experiments, we show they are more precise and efficient and lead to better generalization and adversarial robustness. We provide the code for using our methods at https://github.com/Ali-E/FastClip.
    
[^4]: 通过离散扩散进行大规模无动作视频预训练，以实现高效策略学习

    Large-Scale Actionless Video Pre-Training via Discrete Diffusion for Efficient Policy Learning

    [https://arxiv.org/abs/2402.14407](https://arxiv.org/abs/2402.14407)

    利用离散扩散结合生成式预训练和少量机器人视频微调，实现从人类视频到机器人策略学习的知识迁移。

    

    学习一个能够完成多个任务的通用实体代理面临挑战，主要源自缺乏有标记动作的机器人数据集。相比之下，存在大量捕捉复杂任务和与物理世界互动的人类视频。本文介绍了一种新颖框架，利用统一的离散扩散将人类视频上的生成式预训练与少量有标记机器人视频上的策略微调结合起来。我们首先将人类和机器人视频压缩成统一的视频标记。在预训练阶段，我们使用一个带有蒙版替换扩散策略的离散扩散模型来预测潜空间中的未来视频标记。在微调阶段，我们 h

    arXiv:2402.14407v1 Announce Type: new  Abstract: Learning a generalist embodied agent capable of completing multiple tasks poses challenges, primarily stemming from the scarcity of action-labeled robotic datasets. In contrast, a vast amount of human videos exist, capturing intricate tasks and interactions with the physical world. Promising prospects arise for utilizing actionless human videos for pre-training and transferring the knowledge to facilitate robot policy learning through limited robot demonstrations. In this paper, we introduce a novel framework that leverages a unified discrete diffusion to combine generative pre-training on human videos and policy fine-tuning on a small number of action-labeled robot videos. We start by compressing both human and robot videos into unified video tokens. In the pre-training stage, we employ a discrete diffusion model with a mask-and-replace diffusion strategy to predict future video tokens in the latent space. In the fine-tuning stage, we h
    
[^5]: Switch EMA: 更好的平坦性和锐度的免费午餐

    Switch EMA: A Free Lunch for Better Flatness and Sharpness

    [https://arxiv.org/abs/2402.09240](https://arxiv.org/abs/2402.09240)

    本研究提出了一种称为Switch EMA（SEMA）的方法，通过简单的修改指数移动平均（EMA）参数，可以帮助深度神经网络（DNN）达到更好的平坦性和锐度的概括最优解。通过在多个任务和数据集上进行实验证明了SEMA的有效性。

    

    指数移动平均（EMA）是一种广泛使用的权重平均正则化方法，在深度神经网络（DNN）优化中学习平坦的最优解，以获得更好的概括能力而不增加额外的成本。尽管可以获得更好的平坦性，但现有的平均方法可能陷入更差的最终性能或需要额外的测试时间计算。本文通过一个简单的修改，即在每个周期后将EMA参数切换回原始模型，揭示了EMA的充分潜力，被称为Switch EMA（SEMA）。从理论和实证方面都证明了SEMA可以帮助DNNs达到更好的平坦性和锐度之间平衡的概括最优解。为了验证SEMA的效果，我们在视觉和语言数据集上进行了辨别性、生成性和回归任务的比较实验，包括图像分类、自监督学习、目标检测和分割、图像生成等。

    arXiv:2402.09240v1 Announce Type: new Abstract: Exponential Moving Average (EMA) is a widely used weight averaging (WA) regularization to learn flat optima for better generalizations without extra cost in deep neural network (DNN) optimization. Despite achieving better flatness, existing WA methods might fall into worse final performances or require extra test-time computations. This work unveils the full potential of EMA with a single line of modification, i.e., switching the EMA parameters to the original model after each epoch, dubbed as Switch EMA (SEMA). From both theoretical and empirical aspects, we demonstrate that SEMA can help DNNs to reach generalization optima that better trade-off between flatness and sharpness. To verify the effectiveness of SEMA, we conduct comparison experiments with discriminative, generative, and regression tasks on vision and language datasets, including image classification, self-supervised learning, object detection and segmentation, image generati
    
[^6]: 学习对比特征表示来进行面部动作单元检测

    Learning Contrastive Feature Representations for Facial Action Unit Detection

    [https://arxiv.org/abs/2402.06165](https://arxiv.org/abs/2402.06165)

    这项研究提出了一种对比学习框架，通过监督和自监督信号来增强面部动作单元检测模型的性能。采用正样本抽样和权衡重要性的损失函数来应对噪声AU标签和AU类型分布不平衡的挑战。

    

    面部动作单元（AU）检测的主要方法涉及监督的多标签二进制分类问题。现有的方法常常对AU的像素级信息进行编码，从而对模型的复杂性和表达能力提出了很大的要求。此外，由于存在噪声AU标签，这种做法增加了过拟合的风险。在本研究中，我们引入了一个对比学习框架，通过监督和自监督信号增强。目标是在AU检测领域中摆脱传统的像素级学习范式，获得判别特征。为了应对噪声AU标签带来的挑战，我们通过引入自监督信号来增强监督信号。这种增强是通过正样本抽样实现的，包括三种不同类型的正样本对。另外，为了减轻每个AU类型的分布不平衡问题，我们采用了一种权衡重要性的损失函数。

    The predominant approach to facial action unit (AU) detection revolves around a supervised multi-label binary classification problem. Existing methodologies often encode pixel-level information of AUs, thereby imposing substantial demands on model complexity and expressiveness. Moreover, this practice elevates the susceptibility to overfitting due to the presence of noisy AU labels. In the present study, we introduce a contrastive learning framework enhanced by both supervised and self-supervised signals. The objective is to acquire discriminative features, deviating from the conventional pixel-level learning paradigm within the domain of AU detection. To address the challenge posed by noisy AU labels, we augment the supervised signal through the introduction of a self-supervised signal. This augmentation is achieved through positive sample sampling, encompassing three distinct types of positive sample pairs. Furthermore, to mitigate the imbalanced distribution of each AU type, we empl
    
[^7]: NoSENSE：学习无需显式灵敏度图的展开心脏MRI重建方法

    NoSENSE: Learned unrolled cardiac MRI reconstruction without explicit sensitivity maps. (arXiv:2309.15608v1 [eess.IV])

    [http://arxiv.org/abs/2309.15608](http://arxiv.org/abs/2309.15608)

    本文提出了一种无需显式灵敏度图的展开心脏MRI重建方法，使用深度卷积神经网络和算法展开，通过学习图像之间的接收线圈关系来实现加速心脏MRI重建，在实验中取得了较好的性能。

    

    本文提出了一种新颖的学习图像重建方法，适用于基于多个接收线圈的加速心脏MRI。该方法基于深度卷积神经网络（CNN）和算法展开。与许多现有的学习MR图像重建技术不同，需要将灵敏度映射（CSM）估计作为一个独立的网络组件，我们提出的方法避免了显式的CSM估计。相反，它隐含地捕捉并学习利用图像之间的接收线圈关系。我们的方法由一系列新颖的学习图像块和k空间块组成，共享潜在信息，并通过特征调节和接收线圈数据一致性实现对采集参数的适应性。在MICCAI STACOM CMRxRecon挑战赛的影片追踪和映射追踪验证排行榜中，我们的方法分别在PSNR值上达到了34.89和35.56，SSIM值分别为0.920和0.942，在撰写本文时位列不同小组第4位。

    We present a novel learned image reconstruction method for accelerated cardiac MRI with multiple receiver coils based on deep convolutional neural networks (CNNs) and algorithm unrolling. In contrast to many existing learned MR image reconstruction techniques that necessitate coil-sensitivity map (CSM) estimation as a distinct network component, our proposed approach avoids explicit CSM estimation. Instead, it implicitly captures and learns to exploit the inter-coil relationships of the images. Our method consists of a series of novel learned image and k-space blocks with shared latent information and adaptation to the acquisition parameters by feature-wise modulation (FiLM), as well as coil-wise data-consistency (DC) blocks.  Our method achieved PSNR values of 34.89 and 35.56 and SSIM values of 0.920 and 0.942 in the cine track and mapping track validation leaderboard of the MICCAI STACOM CMRxRecon Challenge, respectively, ranking 4th among different teams at the time of writing.  Cod
    
[^8]: 基于注意力驱动的多模态融合：增强手语识别和翻译

    Attention-Driven Multi-Modal Fusion: Enhancing Sign Language Recognition and Translation. (arXiv:2309.01860v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2309.01860](http://arxiv.org/abs/2309.01860)

    本文提出了一种注意力驱动的多模态融合机制，通过将光流信息与RGB图像相结合，丰富了连续手语识别和翻译流程中的特征。该方法在手语识别任务中降低了WER 0.9，在翻译任务中提高了测试集上大多数BLEU分数约0.6。

    

    本文中，我们设计了一种机制，用于将多模态信息与现有的连续手语识别和翻译流程相结合。在我们的过程中，我们将光流信息与RGB图像结合，以丰富具有与运动相关信息的特征。该工作通过使用跨模态编码器研究了这种模态包含的可行性。我们使用的插件非常轻量级，并且不需要以端到端的方式为新模态包括一个单独的特征提取器。我们在手语识别和翻译中应用了这些改变，改善了每个任务的结果。我们在RWTH-PHOENIX-2014数据集上评估了性能，用于手语识别，并在RWTH-PHOENIX-2014T数据集上评估了翻译任务。在识别任务上，我们的方法将WER降低了0.9，在翻译任务上，我们的方法将大部分BLEU分数在测试集上提高了约0.6。

    In this paper, we devise a mechanism for the addition of multi-modal information with an existing pipeline for continuous sign language recognition and translation. In our procedure, we have incorporated optical flow information with RGB images to enrich the features with movement-related information. This work studies the feasibility of such modality inclusion using a cross-modal encoder. The plugin we have used is very lightweight and doesn't need to include a separate feature extractor for the new modality in an end-to-end manner. We have applied the changes in both sign language recognition and translation, improving the result in each case. We have evaluated the performance on the RWTH-PHOENIX-2014 dataset for sign language recognition and the RWTH-PHOENIX-2014T dataset for translation. On the recognition task, our approach reduced the WER by 0.9, and on the translation task, our approach increased most of the BLEU scores by ~0.6 on the test set.
    
[^9]: 使用深度学习模型进行血细胞分类

    Classification of Blood Cells Using Deep Learning Models. (arXiv:2308.06300v1 [eess.IV])

    [http://arxiv.org/abs/2308.06300](http://arxiv.org/abs/2308.06300)

    这项研究使用深度学习模型通过图像分类对人类血细胞进行了分类和识别，为诊断疾病提供了重要的帮助。

    

    人类血液主要包括血浆、红细胞、白细胞和血小板。血细胞为身体细胞提供氧气，滋养它们，保护它们免受感染，增强免疫力并促进凝血。人的健康状况可以从血细胞中反映出来。一个人被诊断出某种疾病的机会很大程度上受其血细胞类型和计数的影响。因此，血细胞分类非常重要，它可以帮助识别疾病，包括癌症、骨髓损伤、良性肿瘤和它们的生长。这种分类可以帮助血液学家区分不同的血细胞片段，以便确定疾病的原因。卷积神经网络是一种深度学习技术，它将人类血细胞（红细胞、白细胞和血小板）的图像分类为它们的亚型。在这项研究中，使用迁移学习将不同的CNN预训练模型应用于血细胞分类。

    Human blood mainly comprises plasma, red blood cells, white blood cells, and platelets. The blood cells provide the body's cells oxygen to nourish them, shield them from infections, boost immunity, and aid in clotting. Human health is reflected in blood cells. The chances that a human being can be diagnosed with a disease are significantly influenced by their blood cell type and count. Therefore, blood cell classification is crucial because it helps identify diseases, including cancer, damaged bone marrow, benign tumors, and their growth. This classification allows hematologists to distinguish between different blood cell fragments so that the cause of diseases can be identified. Convolution neural networks are a deep learning technique that classifies images of human blood cells (RBCs, WBCs, and platelets) into their subtypes. For this study, transfer learning is used to apply different CNN pre-trained models, including VGG16, VGG19, ResNet-50, ResNet-101, ResNet-152, InceptionV3 Mobi
    
[^10]: 编码-存储-检索：通过语言编码的自我中心感知增强记忆

    Encode-Store-Retrieve: Enhancing Memory Augmentation through Language-Encoded Egocentric Perception. (arXiv:2308.05822v1 [cs.CV])

    [http://arxiv.org/abs/2308.05822](http://arxiv.org/abs/2308.05822)

    本研究提出了一种记忆增强系统，它利用自然语言编码视频数据并将其存储在向量数据库中，通过利用大型视觉语言模型的强大功能来进行语言编码的过程。

    

    我们依赖于自己的记忆来编码、存储和检索我们的经历。然而，记忆间隔有时会发生。实现记忆增强的一种有希望的方法是通过使用增强现实头戴式显示设备来捕捉和保留自我中心的视频，这种做法通常被称为生活记录。然而，由于当前技术缺乏高效编码和存储如此大量的视频数据的能力，从庞大的视频存档中检索特定信息需要大量的计算能力，进一步复杂了快速访问所需内容的任务。

    We depend on our own memory to encode, store, and retrieve our experiences. However, memory lapses can occur. One promising avenue for achieving memory augmentation is through the use of augmented reality head-mounted displays to capture and preserve egocentric videos, a practice commonly referred to as life logging. However, a significant challenge arises from the sheer volume of video data generated through life logging, as the current technology lacks the capability to encode and store such large amounts of data efficiently. Further, retrieving specific information from extensive video archives requires substantial computational power, further complicating the task of quickly accessing desired content. To address these challenges, we propose a memory augmentation system that involves leveraging natural language encoding for video data and storing them in a vector database. This approach harnesses the power of large vision language models to perform the language encoding process. Add
    
[^11]: CST-YOLO: 一种基于改进的YOLOv7和CNN-Swin Transformer的血细胞检测新方法

    CST-YOLO: A Novel Method for Blood Cell Detection Based on Improved YOLOv7 and CNN-Swin Transformer. (arXiv:2306.14590v1 [cs.CV])

    [http://arxiv.org/abs/2306.14590](http://arxiv.org/abs/2306.14590)

    本论文提出了一种CST-YOLO算法，基于改进的YOLOv7和CNN-Swin Transformer，引入了几个有用的模型，有效提高了血细胞检测精度，实验结果显示其在三个血细胞数据集上均优于现有最先进算法。

    

    血细胞检测是计算机视觉中典型的小物体检测问题。本文提出了一种CST-YOLO模型，基于YOLOv7结构并使用CNN-Swin Transformer（CST）进行增强，这是一种CNN-Transformer融合的新尝试。同时，我们还引入了三个有用的模型：加权高效层聚合网络（W-ELAN）、多尺度通道分割（MCS）和级联卷积层（CatConv），以提高小物体检测精度。实验结果表明，我们提出的CST-YOLO在三个血细胞数据集上分别达到了92.7、95.6和91.1 mAP@0.5，优于最先进的物体检测器，如YOLOv5和YOLOv7。我们的代码可在https://github.com/mkang315/CST-YOLO上找到。

    Blood cell detection is a typical small-scale object detection problem in computer vision. In this paper, we propose a CST-YOLO model for blood cell detection based on YOLOv7 architecture and enhance it with the CNN-Swin Transformer (CST), which is a new attempt at CNN-Transformer fusion. We also introduce three other useful modules: Weighted Efficient Layer Aggregation Networks (W-ELAN), Multiscale Channel Split (MCS), and Concatenate Convolutional Layers (CatConv) in our CST-YOLO to improve small-scale object detection precision. Experimental results show that the proposed CST-YOLO achieves 92.7, 95.6, and 91.1 mAP@0.5 respectively on three blood cell datasets, outperforming state-of-the-art object detectors, e.g., YOLOv5 and YOLOv7. Our code is available at https://github.com/mkang315/CST-YOLO.
    

