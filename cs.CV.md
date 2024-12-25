# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Phase Transition in Diffusion Models Reveals the Hierarchical Nature of Data](https://arxiv.org/abs/2402.16991) | 扩散模型在研究数据的分层生成模型中展示出了在阈值时间发生相变的特性，这影响了高级特征和低级特征的重建过程。 |
| [^2] | [Learning Mutual Excitation for Hand-to-Hand and Human-to-Human Interaction Recognition](https://arxiv.org/abs/2402.02431) | 本文介绍了一种学习相互激励的图卷积网络（me-GCN），用于手对手和人对人交互识别。通过堆叠相互激励图卷积层（me-GC），该网络能够自适应地建模成对实体之间的互相约束，并提取和合并深度特征。 |
| [^3] | [Conquering the Communication Constraints to Enable Large Pre-Trained Models in Federated Learning](https://arxiv.org/abs/2210.01708) | 研究克服联邦学习中通信约束的方法，以实现强大的预训练模型在FL中的应用，并同时减少通信负担。 |
| [^4] | [ProCNS: Progressive Prototype Calibration and Noise Suppression for Weakly-Supervised Medical Image Segmentation.](http://arxiv.org/abs/2401.14074) | ProCNS是一种用于弱监督医学图像分割的新方法，采用渐进式原型校准和噪声抑制的原则来解决现有方法中的问题。 |
| [^5] | [Bridging the Domain Gap by Clustering-based Image-Text Graph Matching.](http://arxiv.org/abs/2310.02692) | 通过基于聚类的图像-文本图匹配来弥合领域差距，学习领域不变特征以实现在未见过领域上的良好泛化能力，实验结果显示在公共数据集上达到最先进性能。 |
| [^6] | [Optimizing Convolutional Neural Networks for Chronic Obstructive Pulmonary Disease Detection in Clinical Computed Tomography Imaging.](http://arxiv.org/abs/2303.07189) | 本文旨在通过探索手动调整和自动化窗口设置优化，利用卷积神经网络在临床计算机断层扫描图像中检测慢性阻塞性肺疾病。研究结果表明，通过添加自定义层实现的自动化窗口设置优化可以改善检测性能。 |

# 详细

[^1]: 扩散模型中的相变揭示了数据的分层性质

    A Phase Transition in Diffusion Models Reveals the Hierarchical Nature of Data

    [https://arxiv.org/abs/2402.16991](https://arxiv.org/abs/2402.16991)

    扩散模型在研究数据的分层生成模型中展示出了在阈值时间发生相变的特性，这影响了高级特征和低级特征的重建过程。

    

    理解真实数据的结构在推动现代深度学习方法方面至关重要。自然数据，如图像，被认为是由以层次和组合方式组织的特征组成的，神经网络在学习过程中捕捉到这些特征。最近的进展显示，扩散模型能够生成高质量的图像，暗示了它们捕捉到这种潜在结构的能力。我们研究了数据的分层生成模型中的这一现象。我们发现，在时间$t$后作用的反向扩散过程受到某个阈值时间处的相变控制，此时重建高级特征（如图像的类别）的概率突然下降。相反，低级特征（如图像的具体细节）的重建在整个扩散过程中平稳演变。这一结果暗示，在超出转变时间的时刻，类别已变化，但是基

    arXiv:2402.16991v1 Announce Type: cross  Abstract: Understanding the structure of real data is paramount in advancing modern deep-learning methodologies. Natural data such as images are believed to be composed of features organised in a hierarchical and combinatorial manner, which neural networks capture during learning. Recent advancements show that diffusion models can generate high-quality images, hinting at their ability to capture this underlying structure. We study this phenomenon in a hierarchical generative model of data. We find that the backward diffusion process acting after a time $t$ is governed by a phase transition at some threshold time, where the probability of reconstructing high-level features, like the class of an image, suddenly drops. Instead, the reconstruction of low-level features, such as specific details of an image, evolves smoothly across the whole diffusion process. This result implies that at times beyond the transition, the class has changed but the gene
    
[^2]: 学习相互激励以实现手对手和人对人交互识别

    Learning Mutual Excitation for Hand-to-Hand and Human-to-Human Interaction Recognition

    [https://arxiv.org/abs/2402.02431](https://arxiv.org/abs/2402.02431)

    本文介绍了一种学习相互激励的图卷积网络（me-GCN），用于手对手和人对人交互识别。通过堆叠相互激励图卷积层（me-GC），该网络能够自适应地建模成对实体之间的互相约束，并提取和合并深度特征。

    

    识别交互动作，包括手对手交互和人对人交互，在视频分析和人机交互领域具有广泛的应用。考虑到图卷积在建模骨骼数据的拓扑感知特征方面的成功，最近的方法通常将图卷积应用于独立实体，并在交互动作识别时使用后期融合，这几乎无法建模成对实体之间的互相语义关系。为此，我们通过堆叠相互激励图卷积（me-GC）层，提出了一种相互激励图卷积网络（me-GCN）。具体来说，me-GC使用相互拓扑激励模块首先从单个实体中提取邻接矩阵，然后自适应地对它们之间的相互约束进行建模。此外，me-GC进一步使用相互特征激励模块从成对实体中提取和合并深度特征。

    Recognizing interactive actions, including hand-to-hand interaction and human-to-human interaction, has attracted increasing attention for various applications in the field of video analysis and human-robot interaction. Considering the success of graph convolution in modeling topology-aware features from skeleton data, recent methods commonly operate graph convolution on separate entities and use late fusion for interactive action recognition, which can barely model the mutual semantic relationships between pairwise entities. To this end, we propose a mutual excitation graph convolutional network (me-GCN) by stacking mutual excitation graph convolution (me-GC) layers. Specifically, me-GC uses a mutual topology excitation module to firstly extract adjacency matrices from individual entities and then adaptively model the mutual constraints between them. Moreover, me-GC extends the above idea and further uses a mutual feature excitation module to extract and merge deep features from pairw
    
[^3]: 克服通信约束，实现联邦学习中大型预训练模型的应用

    Conquering the Communication Constraints to Enable Large Pre-Trained Models in Federated Learning

    [https://arxiv.org/abs/2210.01708](https://arxiv.org/abs/2210.01708)

    研究克服联邦学习中通信约束的方法，以实现强大的预训练模型在FL中的应用，并同时减少通信负担。

    

    联邦学习（FL）已经成为一种旨在在本地设备上协力训练模型而不需要对原始数据进行中心化访问的有前景的范式。在典型的FL范式（例如FedAvg）中，每一轮模型权重都会被发送到参与客户端并回传到服务器。最近，在联邦学习优化和收敛改进方面展示了使用小型预训练模型是有效的。然而，最近的最先进预训练模型变得更加强大，但也拥有更多参数。在传统的FL中，共享巨大的模型权重可以迅速给系统带来巨大的通信负担，尤其是如果采用更加强大的模型。我们能否找到一个解决方案，在FL中启用这些强大且现成的预训练模型以实现出色性能的同时减少通信负担？为此，我们研究了使用参数高效的方法

    arXiv:2210.01708v3 Announce Type: replace  Abstract: Federated learning (FL) has emerged as a promising paradigm for enabling the collaborative training of models without centralized access to the raw data on local devices. In the typical FL paradigm (e.g., FedAvg), model weights are sent to and from the server each round to participating clients. Recently, the use of small pre-trained models has been shown effective in federated learning optimization and improving convergence. However, recent state-of-the-art pre-trained models are getting more capable but also have more parameters. In conventional FL, sharing the enormous model weights can quickly put a massive communication burden on the system, especially if more capable models are employed. Can we find a solution to enable those strong and readily-available pre-trained models in FL to achieve excellent performance while simultaneously reducing the communication burden? To this end, we investigate the use of parameter-efficient fin
    
[^4]: ProCNS: 用于弱监督医学图像分割的渐进式原型校准和噪声抑制

    ProCNS: Progressive Prototype Calibration and Noise Suppression for Weakly-Supervised Medical Image Segmentation. (arXiv:2401.14074v1 [cs.CV])

    [http://arxiv.org/abs/2401.14074](http://arxiv.org/abs/2401.14074)

    ProCNS是一种用于弱监督医学图像分割的新方法，采用渐进式原型校准和噪声抑制的原则来解决现有方法中的问题。

    

    弱监督分割（WSS）作为缓解注释成本和模型性能之间冲突的解决方案而出现，采用稀疏的注释格式（例如点、涂鸦、块等）。典型的方法试图利用解剖和拓扑先验将稀疏注释直接扩展为伪标签。然而，由于对医学图像中模糊边缘的关注不足和对稀疏监督的不充分探索，现有方法往往会在噪声区域生成错误且过于自信的伪建议，导致模型误差累积和性能下降。在这项工作中，我们提出了一种新颖的WSS方法，名为ProCNS，它包含两个协同模块，设计原则是渐进式原型校准和噪声抑制。具体而言，我们设计了一种基于原型的区域空间相似性（PRSA）损失函数，最大化空间和语义元素之间的成对相似度，为我们感兴趣的模型提供了

    Weakly-supervised segmentation (WSS) has emerged as a solution to mitigate the conflict between annotation cost and model performance by adopting sparse annotation formats (e.g., point, scribble, block, etc.). Typical approaches attempt to exploit anatomy and topology priors to directly expand sparse annotations into pseudo-labels. However, due to a lack of attention to the ambiguous edges in medical images and insufficient exploration of sparse supervision, existing approaches tend to generate erroneous and overconfident pseudo proposals in noisy regions, leading to cumulative model error and performance degradation. In this work, we propose a novel WSS approach, named ProCNS, encompassing two synergistic modules devised with the principles of progressive prototype calibration and noise suppression. Specifically, we design a Prototype-based Regional Spatial Affinity (PRSA) loss to maximize the pair-wise affinities between spatial and semantic elements, providing our model of interest 
    
[^5]: 基于聚类的图像-文本图匹配来弥合领域差距

    Bridging the Domain Gap by Clustering-based Image-Text Graph Matching. (arXiv:2310.02692v1 [cs.CV])

    [http://arxiv.org/abs/2310.02692](http://arxiv.org/abs/2310.02692)

    通过基于聚类的图像-文本图匹配来弥合领域差距，学习领域不变特征以实现在未见过领域上的良好泛化能力，实验结果显示在公共数据集上达到最先进性能。

    

    学习领域不变表示对于训练可以很好地推广到未见过目标任务领域的模型非常重要。文本描述本身包含概念的语义结构，这样的辅助语义线索可以用作领域概括问题的有效枢纽嵌入。我们使用多模态图像和文本融合的图表示来获得在局部图像和文本描述符之间考虑内在语义结构的领域不变枢纽嵌入。具体来说，我们通过(i)用图表示图像和文本描述，以及(ii)将基于图像节点特征的聚类和匹配应用到文本图中，来学习领域不变特征。我们使用大规模公共数据集（如CUB-DG和DomainBed）进行实验，并在这些数据集上达到与或优于现有最先进模型的性能。我们的代码将在出版后公开提供。

    Learning domain-invariant representations is important to train a model that can generalize well to unseen target task domains. Text descriptions inherently contain semantic structures of concepts and such auxiliary semantic cues can be used as effective pivot embedding for domain generalization problems. Here, we use multimodal graph representations, fusing images and text, to get domain-invariant pivot embeddings by considering the inherent semantic structure between local images and text descriptors. Specifically, we aim to learn domain-invariant features by (i) representing the image and text descriptions with graphs, and by (ii) clustering and matching the graph-based image node features into textual graphs simultaneously. We experiment with large-scale public datasets, such as CUB-DG and DomainBed, and our model achieves matched or better state-of-the-art performance on these datasets. Our code will be publicly available upon publication.
    
[^6]: 在临床计算机断层扫描成像中优化卷积神经网络用于慢性阻塞性肺疾病检测

    Optimizing Convolutional Neural Networks for Chronic Obstructive Pulmonary Disease Detection in Clinical Computed Tomography Imaging. (arXiv:2303.07189v2 [eess.IV] UPDATED)

    [http://arxiv.org/abs/2303.07189](http://arxiv.org/abs/2303.07189)

    本文旨在通过探索手动调整和自动化窗口设置优化，利用卷积神经网络在临床计算机断层扫描图像中检测慢性阻塞性肺疾病。研究结果表明，通过添加自定义层实现的自动化窗口设置优化可以改善检测性能。

    

    目的：通过探索手动调整和自动化窗口设置优化，利用卷积神经网络（CNN）在肺部计算机断层扫描（CT）图像中检测慢性阻塞性肺疾病（COPD）的存在，来优化二进制COPD的检测。方法：回顾性选择了78名受试者（43名COPD患者；35名健康对照组）的7,194个CT图像（3,597个COPD；3,597个健康对照组）（2018年10月至2019年12月）。对每个图像，将强度值手动裁剪到肺气肿窗口设置和基准的“全范围”窗口设置。类平衡的训练、验证和测试集包含了3,392、1,114和2,688个图像。通过比较不同的CNN架构来优化网络主干。此外，还通过向模型添加自定义层来实现自动化的窗口设置优化。根据受试者工作特征曲线（ROC）下面积（AUC）的图像水平，计算出P值来评估性能。

    Purpose: To optimize the binary detection of Chronic Obstructive Pulmonary Disease (COPD) based on emphysema presence in the lung with convolutional neural networks (CNN) by exploring manually adjusted versus automated window-setting optimization (WSO) on computed tomography (CT) images.  Methods: 7,194 CT images (3,597 with COPD; 3,597 healthy controls) from 78 subjects (43 with COPD; 35 healthy controls) were selected retrospectively (10.2018-12.2019) and preprocessed. For each image, intensity values were manually clipped to the emphysema window setting and a baseline 'full-range' window setting. Class-balanced train, validation, and test sets contained 3,392, 1,114, and 2,688 images. The network backbone was optimized by comparing various CNN architectures. Furthermore, automated WSO was implemented by adding a customized layer to the model. The image-level area under the Receiver Operating Characteristics curve (AUC) [lower, upper limit 95% confidence] and P-values calculated from
    

