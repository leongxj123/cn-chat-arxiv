# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DeNetDM: Debiasing by Network Depth Modulation](https://arxiv.org/abs/2403.19863) | DeNetDM 是一种基于网络深度调制的新型去偏见方法，通过使用来自专家乘积的训练范式，在创建深浅架构的偏见和去偏见分支后，将知识提炼产生目标去偏见模型，相比当前去偏见技术取得更优异的效果。 |
| [^2] | [Deep Bayesian Future Fusion for Self-Supervised, High-Resolution, Off-Road Mapping](https://arxiv.org/abs/2403.11876) | 该论文提出了一种深度贝叶斯未来融合的方法，通过自监督的方式实现高分辨率越野地图的制作，为长程预测提供更好的支持。 |
| [^3] | [Implicit Image-to-Image Schrodinger Bridge for CT Super-Resolution and Denoising](https://arxiv.org/abs/2403.06069) | I3SB方法通过引入非马尔可夫过程，结合损坏的图像改善纹理恢复，在CT超分辨率和去噪任务中表现优异。 |
| [^4] | [Cross-Domain Few-Shot Object Detection via Enhanced Open-Set Object Detector](https://arxiv.org/abs/2402.03094) | 本文提出了一种跨领域少样本目标检测器，通过增强的开集目标检测方法来解决跨领域数据差异带来的性能下降问题。 |
| [^5] | [Fast ODE-based Sampling for Diffusion Models in Around 5 Steps](https://arxiv.org/abs/2312.00094) | 提出了一种基于几何观察的Approximate MEan-Direction Solver（AMED-Solver），能够通过直接学习均方向来消除截断误差，从而实现快速扩散抽样。 |
| [^6] | [RoCOCO: Robust Benchmark MS-COCO to Stress-test Robustness of Image-Text Matching Models.](http://arxiv.org/abs/2304.10727) | 本文提出了一个新的评估基准来测试ITM模型的鲁棒性，通过将一些“愚弄”的图片和标题添加到检索池中，在MS COCO数据集上为各种最先进的模型进行鲁棒性测试，揭示了它们的不足之处。 |
| [^7] | [MaskedKD: Efficient Distillation of Vision Transformers with Masked Images.](http://arxiv.org/abs/2302.10494) | MaskedKD提出了一种通过遮蔽图像块来显著降低Vision Transformer (ViT)蒸馏成本的方法，而不影响学生模型的预测准确性。 |

# 详细

[^1]: DeNetDM: 通过网络深度调制来消除偏见

    DeNetDM: Debiasing by Network Depth Modulation

    [https://arxiv.org/abs/2403.19863](https://arxiv.org/abs/2403.19863)

    DeNetDM 是一种基于网络深度调制的新型去偏见方法，通过使用来自专家乘积的训练范式，在创建深浅架构的偏见和去偏见分支后，将知识提炼产生目标去偏见模型，相比当前去偏见技术取得更优异的效果。

    

    当神经网络在偏见数据集上训练时，它们往往会无意间学习到虚假的相关性，从而导致在实现强大的泛化性和鲁棒性方面面临挑战。目前解决这种偏见的方法通常包括利用偏见注释、根据伪偏见标签进行加权重、或通过增强技术增加偏见冲突数据点的多样性。我们引入了DeNetDM，这是一种基于观察结果的新型去偏见方法，浅层神经网络优先学习核心属性，而更深层次的神经网络在获取不同信息时强调偏见。我们利用从专家乘积中推导出的训练范式，创建了深浅架构的偏见和去偏见分支，然后用知识提炼产生目标的去偏见模型。大量实验证明，我们的方法优于当前的去偏见技术，实现了一个...

    arXiv:2403.19863v1 Announce Type: new  Abstract: When neural networks are trained on biased datasets, they tend to inadvertently learn spurious correlations, leading to challenges in achieving strong generalization and robustness. Current approaches to address such biases typically involve utilizing bias annotations, reweighting based on pseudo-bias labels, or enhancing diversity within bias-conflicting data points through augmentation techniques. We introduce DeNetDM, a novel debiasing method based on the observation that shallow neural networks prioritize learning core attributes, while deeper ones emphasize biases when tasked with acquiring distinct information. Using a training paradigm derived from Product of Experts, we create both biased and debiased branches with deep and shallow architectures and then distill knowledge to produce the target debiased model. Extensive experiments and analyses demonstrate that our approach outperforms current debiasing techniques, achieving a not
    
[^2]: 深度贝叶斯未来融合用于自监督、高分辨率、越野地图制作

    Deep Bayesian Future Fusion for Self-Supervised, High-Resolution, Off-Road Mapping

    [https://arxiv.org/abs/2403.11876](https://arxiv.org/abs/2403.11876)

    该论文提出了一种深度贝叶斯未来融合的方法，通过自监督的方式实现高分辨率越野地图的制作，为长程预测提供更好的支持。

    

    资源受限的越野车辆的传感器分辨率有限，这给可靠的越野自主性带来了巨大挑战。为了克服这一局限性，我们提出了一个基于融合未来信息（即未来融合）进行自监督的通用框架。最近的方法利用未来信息以及手工制作的启发式方法来直接监督目标下游任务（例如可穿越性估计）。然而，在本文中，我们选择了一个更为通用的发展方向 - 通过未来融合以自监督的方式时间高效地完成最高分辨率（即每像素2厘米）BEV地图，可用于任何下游任务以获得更好的长程预测。为此，首先，我们创建了一个高分辨率未来融合数据集，其中包含（RGB / 高度）原始稀疏噪音输入和基于地图的密集标签的成对数据。接下来，为了适应传感器的噪声和稀疏性

    arXiv:2403.11876v1 Announce Type: cross  Abstract: The limited sensing resolution of resource-constrained off-road vehicles poses significant challenges towards reliable off-road autonomy. To overcome this limitation, we propose a general framework based on fusing the future information (i.e. future fusion) for self-supervision. Recent approaches exploit this future information alongside the hand-crafted heuristics to directly supervise the targeted downstream tasks (e.g. traversability estimation). However, in this paper, we opt for a more general line of development - time-efficient completion of the highest resolution (i.e. 2cm per pixel) BEV map in a self-supervised manner via future fusion, which can be used for any downstream tasks for better longer range prediction. To this end, first, we create a high-resolution future-fusion dataset containing pairs of (RGB / height) raw sparse and noisy inputs and map-based dense labels. Next, to accommodate the noise and sparsity of the sens
    
[^3]: 隐式图像对图像Schrodinger桥用于CT超分辨率和去噪

    Implicit Image-to-Image Schrodinger Bridge for CT Super-Resolution and Denoising

    [https://arxiv.org/abs/2403.06069](https://arxiv.org/abs/2403.06069)

    I3SB方法通过引入非马尔可夫过程，结合损坏的图像改善纹理恢复，在CT超分辨率和去噪任务中表现优异。

    

    有条件扩散模型因其在图像恢复任务中的有效性而得到认可，然而，其从高斯噪声开始的迭代去噪过程往往导致推断速度慢。作为一种有希望的替代方案，图像对图像Schrödinger桥（I2SB）从损坏的图像开始初始化生成过程，并集成了有条件扩散模型的训练技术。在本研究中，我们通过引入隐式图像对图像Schrödinger桥（I3SB）扩展了I2SB方法，通过在每一生成步骤中纳入损坏的图像，将其生成过程转换为非马尔可夫过程。这种增强使得I3SB能够在少量生成步骤中生成具有更好纹理恢复的图像。所提出的方法在CT超分辨率和去噪任务上得到验证，并超越了包括有条件去噪扩散概率模型在内的现有方法。

    arXiv:2403.06069v1 Announce Type: cross  Abstract: Conditional diffusion models have gained recognition for their effectiveness in image restoration tasks, yet their iterative denoising process, starting from Gaussian noise, often leads to slow inference speeds. As a promising alternative, the Image-to-Image Schr\"odinger Bridge (I2SB) initializes the generative process from corrupted images and integrates training techniques from conditional diffusion models. In this study, we extended the I2SB method by introducing the Implicit Image-to-Image Schrodinger Bridge (I3SB), transitioning its generative process to a non-Markovian process by incorporating corrupted images in each generative step. This enhancement empowers I3SB to generate images with better texture restoration using a small number of generative steps. The proposed method was validated on CT super-resolution and denoising tasks and outperformed existing methods, including the conditional denoising diffusion probabilistic mod
    
[^4]: 跨领域少样本目标检测通过增强的开集目标检测器

    Cross-Domain Few-Shot Object Detection via Enhanced Open-Set Object Detector

    [https://arxiv.org/abs/2402.03094](https://arxiv.org/abs/2402.03094)

    本文提出了一种跨领域少样本目标检测器，通过增强的开集目标检测方法来解决跨领域数据差异带来的性能下降问题。

    

    本文解决了跨领域少样本目标检测（CD-FSOD）的挑战，旨在开发一个准确的目标检测器，用最少的标记样本来检测新领域的目标。虽然基于转换器的开集检测器（例如DE-ViT）在开放词汇目标检测和传统的少样本目标检测方面表现出色，能够检测到训练过程中没有见过的类别，我们自然会提出两个关键问题：1）这种开集检测方法能否容易地推广到CD-FSOD？2）如果不能，如何在面对显著的领域差异时增强开集方法的结果？为了回答第一个问题，我们引入了几个衡量领域差异的指标，并建立了一个具有多样领域度量值的新的CD-FSOD基准。在这个基准上评估了一些最先进的开集目标检测方法，在域外数据集中观察到明显的性能下降。这表明采用这些方法在CD-FSOD上失败了。

    This paper addresses the challenge of cross-domain few-shot object detection (CD-FSOD), aiming to develop an accurate object detector for novel domains with minimal labeled examples. While transformer-based open-set detectors e.g., DE-ViT~\cite{zhang2023detect} have excelled in both open-vocabulary object detection and traditional few-shot object detection, detecting categories beyond those seen during training, we thus naturally raise two key questions: 1) can such open-set detection methods easily generalize to CD-FSOD? 2) If no, how to enhance the results of open-set methods when faced with significant domain gaps? To address the first question, we introduce several metrics to quantify domain variances and establish a new CD-FSOD benchmark with diverse domain metric values. Some State-Of-The-Art (SOTA) open-set object detection methods are evaluated on this benchmark, with evident performance degradation observed across out-of-domain datasets. This indicates the failure of adopting 
    
[^5]: 在大约5个步骤中，用于扩散模型的快速基于ODE的抽样

    Fast ODE-based Sampling for Diffusion Models in Around 5 Steps

    [https://arxiv.org/abs/2312.00094](https://arxiv.org/abs/2312.00094)

    提出了一种基于几何观察的Approximate MEan-Direction Solver（AMED-Solver），能够通过直接学习均方向来消除截断误差，从而实现快速扩散抽样。

    

    从扩散模型中进行抽样可以被视为解决相应的常微分方程（ODE），旨在以尽可能少的函数评估次数（NFE）获得准确解。最近，出现了利用高阶ODE求解器的各种快速抽样器，并且比最初的一阶求解器表现更好。然而，这些数值方法固有地导致某些近似误差，极大地降低了具有极小NFE（例如，约为5）的样本质量。相反，基于几何观察，每个抽样轨迹几乎位于嵌入在环境空间中的二维子空间中，我们提出了用于快速扩散抽样的AME近似均方向求解器（AMED-Solver），通过直接学习均方向来消除截断误差。此外，我们的方法可以轻松作为插件使用，以进一步改进现有的基于ODE的方法。

    arXiv:2312.00094v2 Announce Type: replace-cross  Abstract: Sampling from diffusion models can be treated as solving the corresponding ordinary differential equations (ODEs), with the aim of obtaining an accurate solution with as few number of function evaluations (NFE) as possible. Recently, various fast samplers utilizing higher-order ODE solvers have emerged and achieved better performance than the initial first-order one. However, these numerical methods inherently result in certain approximation errors, which significantly degrades sample quality with extremely small NFE (e.g., around 5). In contrast, based on the geometric observation that each sampling trajectory almost lies in a two-dimensional subspace embedded in the ambient space, we propose Approximate MEan-Direction Solver (AMED-Solver) that eliminates truncation errors by directly learning the mean direction for fast diffusion sampling. Besides, our method can be easily used as a plugin to further improve existing ODE-base
    
[^6]: RoCOCO：稳健的基准MS-COCO评估图文匹配模型的鲁棒性

    RoCOCO: Robust Benchmark MS-COCO to Stress-test Robustness of Image-Text Matching Models. (arXiv:2304.10727v1 [cs.CV])

    [http://arxiv.org/abs/2304.10727](http://arxiv.org/abs/2304.10727)

    本文提出了一个新的评估基准来测试ITM模型的鲁棒性，通过将一些“愚弄”的图片和标题添加到检索池中，在MS COCO数据集上为各种最先进的模型进行鲁棒性测试，揭示了它们的不足之处。

    

    近年来，大规模的视觉语言预训练模型和视觉语义嵌入方法显著提高了MS COCO 5K测试集上图文匹配（ITM）的准确性。然而，当将这些最先进的模型用于实际应用时，它们的鲁棒性仍不清楚。本文提出了一个新的评估基准来测试ITM模型的鲁棒性。为此，我们将各种“愚弄”的图片和标题添加到检索池中。具体而言，我们通过插入不相关的图像来更改图像，并通过替换名词来更改标题，从而改变句子的含义。我们发现，仅仅将这些新创建的图像和标题添加到测试集中就可以降低各种最先进模型的性能（例如，在BLIP中从81.9％降至64.5％，在VSE∞中从66.1％降至37.5％）。我们希望我们的发现能为提高视觉语言模型的鲁棒性和设计更多样化的压力测试提供启示。

    Recently, large-scale vision-language pre-training models and visual semantic embedding methods have significantly improved image-text matching (ITM) accuracy on MS COCO 5K test set. However, it is unclear how robust these state-of-the-art (SOTA) models are when using them in the wild. In this paper, we propose a novel evaluation benchmark to stress-test the robustness of ITM models. To this end, we add various fooling images and captions to a retrieval pool. Specifically, we change images by inserting unrelated images, and change captions by substituting a noun, which can change the meaning of a sentence. We discover that just adding these newly created images and captions to the test set can degrade performances (i.e., Recall@1) of a wide range of SOTA models (e.g., 81.9% $\rightarrow$ 64.5% in BLIP, 66.1% $\rightarrow$ 37.5% in VSE$\infty$). We expect that our findings can provide insights for improving the robustness of the vision-language models and devising more diverse stress-te
    
[^7]: MaskedKD：使用遮蔽图像的高效Vision Transformer蒸馏

    MaskedKD: Efficient Distillation of Vision Transformers with Masked Images. (arXiv:2302.10494v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.10494](http://arxiv.org/abs/2302.10494)

    MaskedKD提出了一种通过遮蔽图像块来显著降低Vision Transformer (ViT)蒸馏成本的方法，而不影响学生模型的预测准确性。

    

    知识蒸馏对于训练轻量级模型是一种有效的方法，但它会在训练成本中引入大量的计算开销，因为该方法需要在训练样本上获取教师监督。当使用大规模的Vision Transformer（ViTs）等教师模型时，这种附加成本——蒸馏成本——最为明显。我们提出了MaskedKD，这是一种简单但有效的策略，可以显着降低蒸馏ViTs的成本，同时不损失学生模型的预测准确性。具体来说，MaskedKD通过遮蔽一部分输入到教师模型的图像块令教师模型的推理成本减少，因此可以跳过处理这些块所需的计算。所选的遮罩位置旨在防止屏蔽学生模型用于预测的图像的核心特征。该遮罩选择机制基于学生模型的某些注意力分数操作。

    Knowledge distillation is an effective method for training lightweight models, but it introduces a significant amount of computational overhead to the training cost, as the method requires acquiring teacher supervisions on training samples. This additional cost -- called distillation cost -- is most pronounced when we employ large-scale teacher models such as vision transformers (ViTs). We present MaskedKD, a simple yet effective strategy that can significantly reduce the cost of distilling ViTs without sacrificing the prediction accuracy of the student model. Specifically, MaskedKD diminishes the cost of running teacher at inference by masking a fraction of image patch tokens fed to the teacher, and therefore skipping the computations required to process those patches. The mask locations are selected to prevent masking away the core features of an image that the student model uses for prediction. This mask selection mechanism operates based on some attention score of the student model
    

