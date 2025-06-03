# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Diffusion Denoising as a Certified Defense against Clean-label Poisoning](https://arxiv.org/abs/2403.11981) | 提出了一种扩散去噪作为针对干净标签中毒的认证防御，能将攻击成功率降低到0-16%，同时几乎不影响测试准确性，为未来开发更强干净标签攻击和利用该防御措施作为强有力基础提供了重要启示。 |
| [^2] | [Unlocking the Potential of Multimodal Unified Discrete Representation through Training-Free Codebook Optimization and Hierarchical Alignment](https://arxiv.org/abs/2403.05168) | 通过无需训练的码本优化和分层对齐，本研究提出了一种方法扩展了多模态统一表示的细粒度，并实现了更好的跨模态泛化。 |
| [^3] | [Generating by Understanding: Neural Visual Generation with Logical Symbol Groundings.](http://arxiv.org/abs/2310.17451) | 这篇论文提出了一种神经符号学习方法，AbdGen，用于将知识推理系统与神经视觉生成模型集成。它解决了符号赋值和规则学习的问题，通过量化诱导方法实现可靠高效的符号赋值，通过对比元诱导方法实现精确的规则学习。 |
| [^4] | [Fact-Checking of AI-Generated Reports.](http://arxiv.org/abs/2307.14634) | 本文提出了一种使用相关联的图像对AI生成报告进行事实核查的新方法，以区分报告中的真假句子。这对加快临床工作流程，提高准确性并降低总体成本具有重要意义。 |
| [^5] | [Translation Consistent Semi-supervised Segmentation for 3D Medical Images.](http://arxiv.org/abs/2203.14523) | 本论文提出了一种名为TraCoCo的半监督学习方法，通过改变输入数据视图的不同空间上下文来扰动训练，从而使模型能够从可视化对象中学习分割模式，实现了三维医学图像的翻译一致半监督分割。 |

# 详细

[^1]: 扩散去噪作为一种针对干净标签中毒的认证防御

    Diffusion Denoising as a Certified Defense against Clean-label Poisoning

    [https://arxiv.org/abs/2403.11981](https://arxiv.org/abs/2403.11981)

    提出了一种扩散去噪作为针对干净标签中毒的认证防御，能将攻击成功率降低到0-16%，同时几乎不影响测试准确性，为未来开发更强干净标签攻击和利用该防御措施作为强有力基础提供了重要启示。

    

    我们提出了一种针对干净标签中毒攻击的认证防御方法。这些攻击通过向训练数据中注入少量的毒害样本（例如1%），其中包含$p$-范数受限的对抗性扰动，从而诱导对测试输入的目标误分类。受到$去噪平滑$实现的对抗鲁棒性的启发，我们展示了如何使用一个现成的扩散模型对篡改的训练数据进行消毒。我们广泛测试了我们的防御措施对七种干净标签中毒攻击的防护效果，并且将它们的攻击成功率降低到0-16%，同时测试准确性几乎没有下降。我们将我们的防御与现有的针对干净标签中毒的对策进行比较，显示出我们的防御效果最佳，并提供最佳的模型效用。我们的结果凸显了未来需要开展更强大的干净标签攻击并使用我们的认证但实用的防御作为稳固基础的必要性。

    arXiv:2403.11981v1 Announce Type: cross  Abstract: We present a certified defense to clean-label poisoning attacks. These attacks work by injecting a small number of poisoning samples (e.g., 1%) that contain $p$-norm bounded adversarial perturbations into the training data to induce a targeted misclassification of a test-time input. Inspired by the adversarial robustness achieved by $denoised$ $smoothing$, we show how an off-the-shelf diffusion model can sanitize the tampered training data. We extensively test our defense against seven clean-label poisoning attacks and reduce their attack success to 0-16% with only a negligible drop in the test time accuracy. We compare our defense with existing countermeasures against clean-label poisoning, showing that the defense reduces the attack success the most and offers the best model utility. Our results highlight the need for future work on developing stronger clean-label attacks and using our certified yet practical defense as a strong base
    
[^2]: 通过无需训练的码本优化和分层对齐解锁多模态统一离散表示的潜力

    Unlocking the Potential of Multimodal Unified Discrete Representation through Training-Free Codebook Optimization and Hierarchical Alignment

    [https://arxiv.org/abs/2403.05168](https://arxiv.org/abs/2403.05168)

    通过无需训练的码本优化和分层对齐，本研究提出了一种方法扩展了多模态统一表示的细粒度，并实现了更好的跨模态泛化。

    

    最近在表示学习方面的进展表明多模态对齐的重要性。利用统一码本的双交叉模态信息解缠（DCID）模型在实现细粒度表示和跨模态泛化方面取得了令人期待的结果。然而，它仍受到对所有通道的均等对待以及忽视次要事件信息的阻碍，导致来自无关通道的干扰并在细粒度任务中表现有限。因此，在这项工作中，我们提出了一种无需训练的码本优化（TOC）方法，通过在统一空间中选择重要通道来增强模型性能。此外，我们引入了分层双交叉模态信息解缠（H-DCID）方法将信息分离和对齐扩展到两个级别，捕捉更多跨模态细节。实验结果表明显著的改进。

    arXiv:2403.05168v1 Announce Type: cross  Abstract: Recent advances in representation learning have demonstrated the significance of multimodal alignment. The Dual Cross-modal Information Disentanglement (DCID) model, utilizing a unified codebook, shows promising results in achieving fine-grained representation and cross-modal generalization. However, it is still hindered by equal treatment of all channels and neglect of minor event information, resulting in interference from irrelevant channels and limited performance in fine-grained tasks. Thus, in this work, We propose a Training-free Optimization of Codebook (TOC) method to enhance model performance by selecting important channels in the unified space without retraining. Additionally, we introduce the Hierarchical Dual Cross-modal Information Disentanglement (H-DCID) approach to extend information separation and alignment to two levels, capturing more cross-modal details. The experiment results demonstrate significant improvements a
    
[^3]: 通过理解生成：具有逻辑符号基础的神经视觉生成

    Generating by Understanding: Neural Visual Generation with Logical Symbol Groundings. (arXiv:2310.17451v1 [cs.AI])

    [http://arxiv.org/abs/2310.17451](http://arxiv.org/abs/2310.17451)

    这篇论文提出了一种神经符号学习方法，AbdGen，用于将知识推理系统与神经视觉生成模型集成。它解决了符号赋值和规则学习的问题，通过量化诱导方法实现可靠高效的符号赋值，通过对比元诱导方法实现精确的规则学习。

    

    尽管近年来神经视觉生成模型取得了很大的成功，但将其与强大的符号知识推理系统集成仍然是一个具有挑战性的任务。主要挑战有两个方面：一个是符号赋值，即将神经视觉生成器的潜在因素与知识推理系统中的有意义的符号进行绑定。另一个是规则学习，即学习新的规则，这些规则控制数据的生成过程，以增强知识推理系统。为了解决这些符号基础问题，我们提出了一种神经符号学习方法，Abductive Visual Generation (AbdGen)，用于基于诱导学习框架将逻辑编程系统与神经视觉生成模型集成起来。为了实现可靠高效的符号赋值，引入了量化诱导方法，通过语义编码本中的最近邻查找生成诱导提案。为了实现精确的规则学习，引入了对比元诱导方法。

    Despite the great success of neural visual generative models in recent years, integrating them with strong symbolic knowledge reasoning systems remains a challenging task. The main challenges are two-fold: one is symbol assignment, i.e. bonding latent factors of neural visual generators with meaningful symbols from knowledge reasoning systems. Another is rule learning, i.e. learning new rules, which govern the generative process of the data, to augment the knowledge reasoning systems. To deal with these symbol grounding problems, we propose a neural-symbolic learning approach, Abductive Visual Generation (AbdGen), for integrating logic programming systems with neural visual generative models based on the abductive learning framework. To achieve reliable and efficient symbol assignment, the quantized abduction method is introduced for generating abduction proposals by the nearest-neighbor lookups within semantic codebooks. To achieve precise rule learning, the contrastive meta-abduction
    
[^4]: AI生成报告的事实核查

    Fact-Checking of AI-Generated Reports. (arXiv:2307.14634v1 [cs.AI])

    [http://arxiv.org/abs/2307.14634](http://arxiv.org/abs/2307.14634)

    本文提出了一种使用相关联的图像对AI生成报告进行事实核查的新方法，以区分报告中的真假句子。这对加快临床工作流程，提高准确性并降低总体成本具有重要意义。

    

    随着生成人工智能（AI）的进步，现在可以生成逼真的自动报告来对放射学图像进行初步阅读。这可以加快临床工作流程，提高准确性并降低总体成本。然而，众所周知，这种模型往往会产生幻觉，导致生成报告中出现错误的发现。在本文中，我们提出了一种使用相关联的图像对AI生成报告进行事实核查的新方法。具体而言，通过学习图像与描述真实或潜在虚假发现的句子之间的关联，开发的核查者区分报告中的真假句子。为了训练这样的核查者，我们首先通过扰动原始与图像相关的放射学报告中的发现，创建了一个新的伪造报告数据集。然后将来自这些报告的真假句子的文本编码与图像编码配对，学习映射到真/假标签的关系。

    With advances in generative artificial intelligence (AI), it is now possible to produce realistic-looking automated reports for preliminary reads of radiology images. This can expedite clinical workflows, improve accuracy and reduce overall costs. However, it is also well-known that such models often hallucinate, leading to false findings in the generated reports. In this paper, we propose a new method of fact-checking of AI-generated reports using their associated images. Specifically, the developed examiner differentiates real and fake sentences in reports by learning the association between an image and sentences describing real or potentially fake findings. To train such an examiner, we first created a new dataset of fake reports by perturbing the findings in the original ground truth radiology reports associated with images. Text encodings of real and fake sentences drawn from these reports are then paired with image encodings to learn the mapping to real/fake labels. The utility 
    
[^5]: 三维医学图像的翻译一致半监督分割

    Translation Consistent Semi-supervised Segmentation for 3D Medical Images. (arXiv:2203.14523v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2203.14523](http://arxiv.org/abs/2203.14523)

    本论文提出了一种名为TraCoCo的半监督学习方法，通过改变输入数据视图的不同空间上下文来扰动训练，从而使模型能够从可视化对象中学习分割模式，实现了三维医学图像的翻译一致半监督分割。

    

    三维医学图像分割的方法已经成功，但其依赖于涵盖大量体素的注释数据，这是一个需要解决的劣势，因为获取这种注释的成本很高。半监督学习（SSL）通过使用大量未标记的数据集和少量标记的数据集来训练模型来解决这个问题。最成功的SSL方法基于一致性学习，该方法通过最小化从未标记数据的扰动视图获得的模型响应之间的距离来实现。这些扰动通常会保持视图之间的空间输入上下文相当一致，这可能会使模型从空间输入上下文中学习分割模式，而不是从分割对象本身中学习。在本文中，我们介绍了翻译一致协同训练（TraCoCo），这是一种一致性学习SSL方法，它通过改变不同的空间输入上下文来扰动输入数据视图，使模型能够从可视化对象中学习分割模式。

    3D medical image segmentation methods have been successful, but their dependence on large amounts of voxel-level annotated data is a disadvantage that needs to be addressed given the high cost to obtain such annotation. Semi-supervised learning (SSL) solve this issue by training models with a large unlabelled and a small labelled dataset. The most successful SSL approaches are based on consistency learning that minimises the distance between model responses obtained from perturbed views of the unlabelled data. These perturbations usually keep the spatial input context between views fairly consistent, which may cause the model to learn segmentation patterns from the spatial input contexts instead of the segmented objects. In this paper, we introduce the Translation Consistent Co-training (TraCoCo) which is a consistency learning SSL method that perturbs the input data views by varying their spatial input context, allowing the model to learn segmentation patterns from visual objects. Fur
    

