# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards Adversarially Robust Dataset Distillation by Curvature Regularization](https://arxiv.org/abs/2403.10045) | 本文探讨了如何通过曲率正则化方法在精炼数据集中嵌入对抗鲁棒性，以保持模型高准确性并获得更好的对抗鲁棒性。 |
| [^2] | [Synthesizing Moving People with 3D Control.](http://arxiv.org/abs/2401.10889) | 本文提出了一种基于扩散模型的框架，用于从单张图像中生成具有逼真移动的人物动画，并成功处理了人体不可见部分的合成问题。 |
| [^3] | [Learning ECG signal features without backpropagation.](http://arxiv.org/abs/2307.01930) | 该论文提出了一种用于生成时间序列数据表示的新方法，依靠理论物理的思想以数据驱动的方式构建紧凑的表示。该方法能够捕捉数据的基本结构和任务特定信息，同时保持直观、可解释和可验证性，并可以在广义设置中应用。 |
| [^4] | [Layer-level activation mechanism.](http://arxiv.org/abs/2306.04940) | 去噪声更好，表现更好的分层级别激活机制 |
| [^5] | [RoCOCO: Robust Benchmark MS-COCO to Stress-test Robustness of Image-Text Matching Models.](http://arxiv.org/abs/2304.10727) | 本文提出了一个新的评估基准来测试ITM模型的鲁棒性，通过将一些“愚弄”的图片和标题添加到检索池中，在MS COCO数据集上为各种最先进的模型进行鲁棒性测试，揭示了它们的不足之处。 |
| [^6] | [EDO-Net: Learning Elastic Properties of Deformable Objects from Graph Dynamics.](http://arxiv.org/abs/2209.08996) | EDO-Net是一个学习可变形物体弹性属性的图动力学模型，通过利用可提取的潜在表示，可以推广到未知的物理属性，实现对类似布料的对象未来状态的预测和转移学习。 |

# 详细

[^1]: 通过曲率正则化实现对抗鲁棒性数据集精炼

    Towards Adversarially Robust Dataset Distillation by Curvature Regularization

    [https://arxiv.org/abs/2403.10045](https://arxiv.org/abs/2403.10045)

    本文探讨了如何通过曲率正则化方法在精炼数据集中嵌入对抗鲁棒性，以保持模型高准确性并获得更好的对抗鲁棒性。

    

    数据集精炼（DD）允许将数据集精炼为原始大小的分数，同时保留丰富的分布信息，使得在精炼数据集上训练的模型可以在节省显著计算负载的同时达到可比的准确性。最近在这一领域的研究集中在提高在精炼数据集上训练的模型的准确性。在本文中，我们旨在探索DD的一种新视角。我们研究如何在精炼数据集中嵌入对抗鲁棒性，以使在这些数据集上训练的模型保持高精度的同时获得更好的对抗鲁棒性。我们提出了一种通过将曲率正则化纳入到精炼过程中来实现这一目标的新方法，而这种方法的计算开销比标准的对抗训练要少得多。大量的实证实验表明，我们的方法不仅在准确性上优于标准对抗训练，同时在对抗性能方面也取得了显著改进。

    arXiv:2403.10045v1 Announce Type: new  Abstract: Dataset distillation (DD) allows datasets to be distilled to fractions of their original size while preserving the rich distributional information so that models trained on the distilled datasets can achieve a comparable accuracy while saving significant computational loads. Recent research in this area has been focusing on improving the accuracy of models trained on distilled datasets. In this paper, we aim to explore a new perspective of DD. We study how to embed adversarial robustness in distilled datasets, so that models trained on these datasets maintain the high accuracy and meanwhile acquire better adversarial robustness. We propose a new method that achieves this goal by incorporating curvature regularization into the distillation process with much less computational overhead than standard adversarial training. Extensive empirical experiments suggest that our method not only outperforms standard adversarial training on both accur
    
[^2]: 使用3D控制合成移动人物

    Synthesizing Moving People with 3D Control. (arXiv:2401.10889v1 [cs.CV])

    [http://arxiv.org/abs/2401.10889](http://arxiv.org/abs/2401.10889)

    本文提出了一种基于扩散模型的框架，用于从单张图像中生成具有逼真移动的人物动画，并成功处理了人体不可见部分的合成问题。

    

    本文提出了一种基于扩散模型的框架，用于从单张图像中为给定的目标3D运动序列生成人物动画。我们的方法包含两个核心组成部分：a) 学习关于人体和服装不可见部分的先验知识，b) 以适当的服装和纹理渲染新的人体姿势。对于第一部分，我们学习了一种填充扩散模型，以给定单张图像生成人物的不可见部分。我们在纹理映射空间上训练这个模型，使其对姿势和视角不变，从而提高了样本效率。其次，我们开发了一个基于扩散的渲染流水线，由3D人体姿势控制。这可以产生逼真的人物新姿势的渲染图像，包括服装、头发和未知区域的合理补充。这种分解的方法使我们的方法能够生成一系列图像，既符合3D姿势中的目标运动，也符合视觉上与输入图像的相似性。

    In this paper, we present a diffusion model-based framework for animating people from a single image for a given target 3D motion sequence. Our approach has two core components: a) learning priors about invisible parts of the human body and clothing, and b) rendering novel body poses with proper clothing and texture. For the first part, we learn an in-filling diffusion model to hallucinate unseen parts of a person given a single image. We train this model on texture map space, which makes it more sample-efficient since it is invariant to pose and viewpoint. Second, we develop a diffusion-based rendering pipeline, which is controlled by 3D human poses. This produces realistic renderings of novel poses of the person, including clothing, hair, and plausible in-filling of unseen regions. This disentangled approach allows our method to generate a sequence of images that are faithful to the target motion in the 3D pose and, to the input image in terms of visual similarity. In addition to tha
    
[^3]: 学习ECG信号特征的非反向传播方法

    Learning ECG signal features without backpropagation. (arXiv:2307.01930v1 [cs.LG])

    [http://arxiv.org/abs/2307.01930](http://arxiv.org/abs/2307.01930)

    该论文提出了一种用于生成时间序列数据表示的新方法，依靠理论物理的思想以数据驱动的方式构建紧凑的表示。该方法能够捕捉数据的基本结构和任务特定信息，同时保持直观、可解释和可验证性，并可以在广义设置中应用。

    

    表示学习已经成为机器学习领域的一个关键研究领域，它旨在发现用于提高分类和预测等下游任务的原始数据的有效特征的有效方法。在本文中，我们提出了一种用于生成时间序列类型数据表示的新方法。这种方法依靠理论物理的思想以数据驱动的方式构建紧凑的表示，并可以捕捉到数据的基本结构和任务特定信息，同时保持直观、可解释和可验证性。这个新方法旨在识别能够有效捕捉属于特定类别的样本之间共享特征的线性规律。通过随后利用这些规律在前向方式下生成一个与分类器无关的表示，它们可以在广义设置中应用。我们展示了我们方法的有效性。

    Representation learning has become a crucial area of research in machine learning, as it aims to discover efficient ways of representing raw data with useful features to increase the effectiveness, scope and applicability of downstream tasks such as classification and prediction. In this paper, we propose a novel method to generate representations for time series-type data. This method relies on ideas from theoretical physics to construct a compact representation in a data-driven way, and it can capture both the underlying structure of the data and task-specific information while still remaining intuitive, interpretable and verifiable. This novel methodology aims to identify linear laws that can effectively capture a shared characteristic among samples belonging to a specific class. By subsequently utilizing these laws to generate a classifier-agnostic representation in a forward manner, they become applicable in a generalized setting. We demonstrate the effectiveness of our approach o
    
[^4]: 分层级别激活机制

    Layer-level activation mechanism. (arXiv:2306.04940v1 [cs.LG])

    [http://arxiv.org/abs/2306.04940](http://arxiv.org/abs/2306.04940)

    去噪声更好，表现更好的分层级别激活机制

    

    本文提出了一种新颖的激活机制，旨在建立分层级别激活功能（LayerAct）。这些功能旨在通过减少输入偏移所导致的激活输出的分层级波动来降低传统元素级激活功能的噪音鲁棒性。此外，LayerAct功能实现了类似于零的平均激活输出，而不限制激活输出空间。我们进行了分析和实验，证明LayerAct功能在噪声鲁棒性方面优于元素级激活功能，并且经验证明这些功能的平均激活结果类似于零。在三个基准图像分类任务的实验结果表明，在处理嘈杂的图像数据集时，LayerAct功能比元素级激活功能表现更好，而在大多数情况下，清洁数据集的表现也是优越的。

    In this work, we propose a novel activation mechanism aimed at establishing layer-level activation (LayerAct) functions. These functions are designed to be more noise-robust compared to traditional element-level activation functions by reducing the layer-level fluctuation of the activation outputs due to shift in inputs. Moreover, the LayerAct functions achieve a zero-like mean activation output without restricting the activation output space. We present an analysis and experiments demonstrating that LayerAct functions exhibit superior noise-robustness compared to element-level activation functions, and empirically show that these functions have a zero-like mean activation. Experimental results on three benchmark image classification tasks show that LayerAct functions excel in handling noisy image datasets, outperforming element-level activation functions, while the performance on clean datasets is also superior in most cases.
    
[^5]: RoCOCO：稳健的基准MS-COCO评估图文匹配模型的鲁棒性

    RoCOCO: Robust Benchmark MS-COCO to Stress-test Robustness of Image-Text Matching Models. (arXiv:2304.10727v1 [cs.CV])

    [http://arxiv.org/abs/2304.10727](http://arxiv.org/abs/2304.10727)

    本文提出了一个新的评估基准来测试ITM模型的鲁棒性，通过将一些“愚弄”的图片和标题添加到检索池中，在MS COCO数据集上为各种最先进的模型进行鲁棒性测试，揭示了它们的不足之处。

    

    近年来，大规模的视觉语言预训练模型和视觉语义嵌入方法显著提高了MS COCO 5K测试集上图文匹配（ITM）的准确性。然而，当将这些最先进的模型用于实际应用时，它们的鲁棒性仍不清楚。本文提出了一个新的评估基准来测试ITM模型的鲁棒性。为此，我们将各种“愚弄”的图片和标题添加到检索池中。具体而言，我们通过插入不相关的图像来更改图像，并通过替换名词来更改标题，从而改变句子的含义。我们发现，仅仅将这些新创建的图像和标题添加到测试集中就可以降低各种最先进模型的性能（例如，在BLIP中从81.9％降至64.5％，在VSE∞中从66.1％降至37.5％）。我们希望我们的发现能为提高视觉语言模型的鲁棒性和设计更多样化的压力测试提供启示。

    Recently, large-scale vision-language pre-training models and visual semantic embedding methods have significantly improved image-text matching (ITM) accuracy on MS COCO 5K test set. However, it is unclear how robust these state-of-the-art (SOTA) models are when using them in the wild. In this paper, we propose a novel evaluation benchmark to stress-test the robustness of ITM models. To this end, we add various fooling images and captions to a retrieval pool. Specifically, we change images by inserting unrelated images, and change captions by substituting a noun, which can change the meaning of a sentence. We discover that just adding these newly created images and captions to the test set can degrade performances (i.e., Recall@1) of a wide range of SOTA models (e.g., 81.9% $\rightarrow$ 64.5% in BLIP, 66.1% $\rightarrow$ 37.5% in VSE$\infty$). We expect that our findings can provide insights for improving the robustness of the vision-language models and devising more diverse stress-te
    
[^6]: EDO-Net: 从图动力学中学习可变形物体的弹性属性

    EDO-Net: Learning Elastic Properties of Deformable Objects from Graph Dynamics. (arXiv:2209.08996v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2209.08996](http://arxiv.org/abs/2209.08996)

    EDO-Net是一个学习可变形物体弹性属性的图动力学模型，通过利用可提取的潜在表示，可以推广到未知的物理属性，实现对类似布料的对象未来状态的预测和转移学习。

    

    我们研究了学习可变形物体的图动力学的问题，该问题可以推广到未知的物理属性。我们的关键洞察力是利用可提取的类似布料的可变形物体的弹性物理属性的潜在表示，例如从拉伸交互中提取。在本文中，我们提出了EDO-Net（弹性可变形对象-Net），这是一个在具有不同弹性属性的大量样本上进行训练的图动力学模型，不依赖于属性的真实标签。EDO-Net共同学习了一个自适应模块和一个前向动力学模块。前者负责提取对象的物理特性的潜在表示，而后者利用潜在表示来预测以图形表示的类似布料的对象的未来状态。我们在仿真和真实世界中评估了EDO-Net的能力：1）推广到未知的物理属性，2）转移学习所学到的表示

    We study the problem of learning graph dynamics of deformable objects that generalizes to unknown physical properties. Our key insight is to leverage a latent representation of elastic physical properties of cloth-like deformable objects that can be extracted, for example, from a pulling interaction. In this paper we propose EDO-Net (Elastic Deformable Object - Net), a model of graph dynamics trained on a large variety of samples with different elastic properties that does not rely on ground-truth labels of the properties. EDO-Net jointly learns an adaptation module, and a forward-dynamics module. The former is responsible for extracting a latent representation of the physical properties of the object, while the latter leverages the latent representation to predict future states of cloth-like objects represented as graphs. We evaluate EDO-Net both in simulation and real world, assessing its capabilities of: 1) generalizing to unknown physical properties, 2) transferring the learned rep
    

