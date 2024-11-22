# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AI-generated faces free from racial and gender stereotypes](https://rss.arxiv.org/abs/2402.01002) | 这项研究发现并解决了AI生成的面孔中存在的种族和性别刻板印象问题，提出了分类器用于预测面部属性的方法，并提出了有效的去偏见解决方案。 |
| [^2] | [Design2Code: How Far Are We From Automating Front-End Engineering?](https://arxiv.org/abs/2403.03163) | 生成式人工智能在多模态理解和代码生成方面取得了突破，提出了Design2Code任务并进行了全面基准测试，展示了多模态LLMs直接将视觉设计转换为代码实现的能力。 |
| [^3] | [Uncertainty-Guided Alignment for Unsupervised Domain Adaptation in Regression.](http://arxiv.org/abs/2401.13721) | 该论文提出了一种利用不确定性引导的无监督领域自适应回归方法，通过将不确定性作为置信度估计和嵌入空间的正则项来实现对齐。 |
| [^4] | [Criticality-Guided Efficient Pruning in Spiking Neural Networks Inspired by Critical Brain Hypothesis.](http://arxiv.org/abs/2311.16141) | 本研究受到神经科学中的关键大脑假设的启发，提出了一种基于神经元关键性的高效SNN修剪方法，以加强特征提取和加速修剪过程，并取得了比当前最先进方法更好的性能。 |
| [^5] | [Multi-task Deep Convolutional Network to Predict Sea Ice Concentration and Drift in the Arctic Ocean.](http://arxiv.org/abs/2311.00167) | 提出了一种名为HIS-Unet的多任务深度卷积网络架构，通过加权注意力模块实现海冰浓度和漂移的预测。与其他方法相比，HIS-Unet在海冰预测中取得了显著的改进。 |
| [^6] | [A Fusion of Variational Distribution Priors and Saliency Map Replay for Continual 3D Reconstruction.](http://arxiv.org/abs/2308.08812) | 本研究提出了一种基于连续学习的三维重建方法，通过使用变分先验和显著性图重新播放，实现对之前已见类别的合理重建。实验结果表明，与已建立的方法相比具有竞争力。 |

# 详细

[^1]: AI生成的面孔摆脱了种族和性别刻板印象

    AI-generated faces free from racial and gender stereotypes

    [https://rss.arxiv.org/abs/2402.01002](https://rss.arxiv.org/abs/2402.01002)

    这项研究发现并解决了AI生成的面孔中存在的种族和性别刻板印象问题，提出了分类器用于预测面部属性的方法，并提出了有效的去偏见解决方案。

    

    诸如Stable Diffusion之类的文本到图像生成AI模型每天都被全球数百万人使用。然而，许多人对这些模型如何放大种族和性别刻板印象提出了关切。为了研究这一现象，我们开发了一个分类器来预测任意给定面部图像的种族、性别和年龄组，并展示其达到了最先进的性能。利用这个分类器，我们对Stable Diffusion在六种种族、两种性别、五个年龄组、32个职业和八个属性上的偏见进行了量化。然后，我们提出了超越最先进替代方案的新型去偏见解决方案。此外，我们还检查了Stable Diffusion在描绘同一种族的个体时相似程度。分析结果显示出高度的刻板印象，例如，将大多数中东男性描绘为皮肤黝黑、留着胡子、戴着传统头饰。我们提出了另一种增加面部多样性的新型解决方案来解决这些限制。

    Text-to-image generative AI models such as Stable Diffusion are used daily by millions worldwide. However, many have raised concerns regarding how these models amplify racial and gender stereotypes. To study this phenomenon, we develop a classifier to predict the race, gender, and age group of any given face image, and show that it achieves state-of-the-art performance. Using this classifier, we quantify biases in Stable Diffusion across six races, two genders, five age groups, 32 professions, and eight attributes. We then propose novel debiasing solutions that outperform state-of-the-art alternatives. Additionally, we examine the degree to which Stable Diffusion depicts individuals of the same race as being similar to one another. This analysis reveals a high degree of stereotyping, e.g., depicting most middle eastern males as being dark-skinned, bearded, and wearing a traditional headdress. We address these limitations by proposing yet another novel solution that increases facial div
    
[^2]: Design2Code：我们离自动化前端工程有多远？

    Design2Code: How Far Are We From Automating Front-End Engineering?

    [https://arxiv.org/abs/2403.03163](https://arxiv.org/abs/2403.03163)

    生成式人工智能在多模态理解和代码生成方面取得了突破，提出了Design2Code任务并进行了全面基准测试，展示了多模态LLMs直接将视觉设计转换为代码实现的能力。

    

    近年来，生成式人工智能在多模态理解和代码生成方面取得了突飞猛进的进展，实现了前所未有的能力。这可以实现一种新的前端开发范式，其中多模态LLMs可能直接将视觉设计转换为代码实现。本文将这一过程形式化为Design2Code任务，并进行全面基准测试。我们手动策划了一个包含484个多样化真实网页的基准测试用例，并开发了一套自动评估指标，以评估当前多模态LLMs能否生成直接渲染为给定参考网页的代码实现，以输入为屏幕截图。我们还结合了全面的人工评估。我们开发了一套多模态提示方法，并展示了它们在GPT-4V和Gemini Pro Vision上的有效性。我们进一步对一个开源的Design2Code-18B模型进行了微调。

    arXiv:2403.03163v1 Announce Type: new  Abstract: Generative AI has made rapid advancements in recent years, achieving unprecedented capabilities in multimodal understanding and code generation. This can enable a new paradigm of front-end development, in which multimodal LLMs might directly convert visual designs into code implementations. In this work, we formalize this as a Design2Code task and conduct comprehensive benchmarking. Specifically, we manually curate a benchmark of 484 diverse real-world webpages as test cases and develop a set of automatic evaluation metrics to assess how well current multimodal LLMs can generate the code implementations that directly render into the given reference webpages, given the screenshots as input. We also complement automatic metrics with comprehensive human evaluations. We develop a suite of multimodal prompting methods and show their effectiveness on GPT-4V and Gemini Pro Vision. We further finetune an open-source Design2Code-18B model that su
    
[^3]: 无监督领域自适应回归中的不确定性引导对齐

    Uncertainty-Guided Alignment for Unsupervised Domain Adaptation in Regression. (arXiv:2401.13721v1 [cs.CV])

    [http://arxiv.org/abs/2401.13721](http://arxiv.org/abs/2401.13721)

    该论文提出了一种利用不确定性引导的无监督领域自适应回归方法，通过将不确定性作为置信度估计和嵌入空间的正则项来实现对齐。

    

    无监督领域自适应回归（UDAR）旨在将来自有标签源领域的模型调整到无标签目标领域，以完成回归任务。最近在UDAR领域取得的成功主要集中在子空间对齐上，涉及整个特征空间中所选择子空间的对齐。这与用于分类的特征对齐方法形成对比，后者旨在对齐整个特征空间，在分类任务中已被证明是有效的，但在回归任务中效果较差。具体而言，分类任务旨在在整个嵌入空间的维度上识别独立的簇，而回归任务对数据表示的结构性要求较低，需要额外的指导以实现有效的对齐。在本文中，我们提出了一种通过不确定性引导的有效UDAR方法。我们的方法具有双重作用：提供了对预测结果的置信度衡量，并作为嵌入空间的正则化。具体而言，我们利用深度证据模型来提供对预测的置信度估计，并将其作为嵌入空间的正则项进行优化。

    Unsupervised Domain Adaptation for Regression (UDAR) aims to adapt a model from a labeled source domain to an unlabeled target domain for regression tasks. Recent successful works in UDAR mostly focus on subspace alignment, involving the alignment of a selected subspace within the entire feature space. This contrasts with the feature alignment methods used for classification, which aim at aligning the entire feature space and have proven effective but are less so in regression settings. Specifically, while classification aims to identify separate clusters across the entire embedding dimension, regression induces less structure in the data representation, necessitating additional guidance for efficient alignment. In this paper, we propose an effective method for UDAR by incorporating guidance from uncertainty. Our approach serves a dual purpose: providing a measure of confidence in predictions and acting as a regularization of the embedding space. Specifically, we leverage the Deep Evid
    
[^4]: SNNs中基于关键性的高效修剪方法，受到关键性大脑假设的启发

    Criticality-Guided Efficient Pruning in Spiking Neural Networks Inspired by Critical Brain Hypothesis. (arXiv:2311.16141v2 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2311.16141](http://arxiv.org/abs/2311.16141)

    本研究受到神经科学中的关键大脑假设的启发，提出了一种基于神经元关键性的高效SNN修剪方法，以加强特征提取和加速修剪过程，并取得了比当前最先进方法更好的性能。

    

    由于其节能和无乘法特性，SNNs已经引起了相当大的关注。深度SNNs规模的不断增长给模型部署带来了挑战。网络修剪通过压缩网络规模来减少模型部署的硬件资源需求。然而，现有的SNN修剪方法由于修剪迭代增加了SNNs的训练难度，导致修剪成本高昂且性能损失严重。本文受到神经科学中的关键大脑假设的启发，提出了一种基于神经元关键性的用于SNN修剪的再生机制，以增强特征提取并加速修剪过程。首先，我们提出了一种SNN中用于关键性的低成本度量方式。然后，在修剪后对所修剪结构进行重新排序，并再生那些具有较高关键性的结构，以获取关键网络。我们的方法表现优于当前的最先进方法。

    Spiking Neural Networks (SNNs) have gained considerable attention due to the energy-efficient and multiplication-free characteristics. The continuous growth in scale of deep SNNs poses challenges for model deployment. Network pruning reduces hardware resource requirements of model deployment by compressing the network scale. However, existing SNN pruning methods cause high pruning costs and performance loss because the pruning iterations amplify the training difficulty of SNNs. In this paper, inspired by the critical brain hypothesis in neuroscience, we propose a regeneration mechanism based on the neuron criticality for SNN pruning to enhance feature extraction and accelerate the pruning process. Firstly, we propose a low-cost metric for the criticality in SNNs. Then, we re-rank the pruned structures after pruning and regenerate those with higher criticality to obtain the critical network. Our method achieves higher performance than the current state-of-the-art (SOTA) method with up t
    
[^5]: 多任务深度卷积网络预测北冰洋海冰浓度和漂移

    Multi-task Deep Convolutional Network to Predict Sea Ice Concentration and Drift in the Arctic Ocean. (arXiv:2311.00167v1 [cs.LG])

    [http://arxiv.org/abs/2311.00167](http://arxiv.org/abs/2311.00167)

    提出了一种名为HIS-Unet的多任务深度卷积网络架构，通过加权注意力模块实现海冰浓度和漂移的预测。与其他方法相比，HIS-Unet在海冰预测中取得了显著的改进。

    

    在北冰洋地区，预测海冰浓度(SIC)和海冰漂移(SID)具有重要意义，因为最近的气候变暖已经改变了这个环境。由于物理海冰模型需要高计算成本和复杂的参数化，深度学习技术可以有效替代物理模型，并提高海冰预测的性能。本研究提出了一种新颖的多任务全卷积网络架构，名为Hierarchical Information-Sharing U-Net (HIS-Unet)，用于预测每日的SIC和SID。我们通过加权注意力模块(WAMs)允许SIC和SID层共享信息，并互相辅助预测。结果表明，相比于其他统计方法、海冰物理模型和没有信息共享单元的神经网络，我们的HIS-Unet在SIC和SID预测方面表现更优。在预测北冰洋海冰浓度和漂移方面，HIS-Unet的改进都是显著的。

    Forecasting sea ice concentration (SIC) and sea ice drift (SID) in the Arctic Ocean is of great significance as the Arctic environment has been changed by the recent warming climate. Given that physical sea ice models require high computational costs with complex parameterization, deep learning techniques can effectively replace the physical model and improve the performance of sea ice prediction. This study proposes a novel multi-task fully conventional network architecture named hierarchical information-sharing U-net (HIS-Unet) to predict daily SIC and SID. Instead of learning SIC and SID separately at each branch, we allow the SIC and SID layers to share their information and assist each other's prediction through the weighting attention modules (WAMs). Consequently, our HIS-Unet outperforms other statistical approaches, sea ice physical models, and neural networks without such information-sharing units. The improvement of HIS-Unet is obvious both for SIC and SID prediction when and
    
[^6]: 变分分布先验与显著性图重新播放的融合，用于连续三维重建

    A Fusion of Variational Distribution Priors and Saliency Map Replay for Continual 3D Reconstruction. (arXiv:2308.08812v1 [cs.CV])

    [http://arxiv.org/abs/2308.08812](http://arxiv.org/abs/2308.08812)

    本研究提出了一种基于连续学习的三维重建方法，通过使用变分先验和显著性图重新播放，实现对之前已见类别的合理重建。实验结果表明，与已建立的方法相比具有竞争力。

    

    单图像三维重建是研究如何根据单视角图像预测三维物体形状的一个挑战。这个任务需要大量数据获取来预测形状的可见和遮挡部分。此外，基于学习的方法面临创建针对所有可能类别的全面训练数据集的困难。为此，我们提出了一种基于连续学习的三维重建方法，我们的目标是设计一个使用变分先验的模型，即使在训练新类别后仍可以合理重建以前见过的类别。变分先验代表抽象形状并避免遗忘，而显著性图以较少的内存使用保留对象属性。这对于存储大量训练数据的资源限制至关重要。此外，我们引入了基于显著性图的经验重放，以捕捉全局和独特的对象特征。详细的实验显示与已建立方法相比具有竞争力的结果。

    Single-image 3D reconstruction is a research challenge focused on predicting 3D object shapes from single-view images. This task requires significant data acquisition to predict both visible and occluded portions of the shape. Furthermore, learning-based methods face the difficulty of creating a comprehensive training dataset for all possible classes. To this end, we propose a continual learning-based 3D reconstruction method where our goal is to design a model using Variational Priors that can still reconstruct the previously seen classes reasonably even after training on new classes. Variational Priors represent abstract shapes and combat forgetting, whereas saliency maps preserve object attributes with less memory usage. This is vital due to resource constraints in storing extensive training data. Additionally, we introduce saliency map-based experience replay to capture global and distinct object features. Thorough experiments show competitive results compared to established method
    

