# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Hyper-Diffusion: Estimating Epistemic and Aleatoric Uncertainty with a Single Model](https://arxiv.org/abs/2402.03478) | 本研究引入了一种新的集合方法，超扩散，可以使用单一模型准确估计认识和偶然不确定性。 |
| [^2] | [GD doesn't make the cut: Three ways that non-differentiability affects neural network training](https://arxiv.org/abs/2401.08426) | 本文研究了非可微性对神经网络训练的影响，包括收敛性差异、$L_1$正则化问题的矛盾性质以及稳定边界现象的不适用性。 |
| [^3] | [Ventricular Segmentation: A Brief Comparison of U-Net Derivatives.](http://arxiv.org/abs/2401.09980) | 本文探讨了深度学习技术在心脏图像分割中的应用，实施了多个U-Net衍生模型以实现对心脏特定部位的全面解剖和功能分析。通过图像、图表和定量指标验证了模型的效果，并讨论了面临的挑战和未来改进策略。 |
| [^4] | [Robust multimodal models have outlier features and encode more concepts.](http://arxiv.org/abs/2310.13040) | 健壮的多模态模型展示了异常特征和更多概念的编码方式。 |
| [^5] | [Language Aligned Visual Representations Predict Human Behavior in Naturalistic Learning Tasks.](http://arxiv.org/abs/2306.09377) | 语言对齐的视觉表示方式比纯视觉表示方式更有效地预测人类在自然学习任务中的行为。 |

# 详细

[^1]: 超扩散：使用单一模型估计认识和偶然不确定性

    Hyper-Diffusion: Estimating Epistemic and Aleatoric Uncertainty with a Single Model

    [https://arxiv.org/abs/2402.03478](https://arxiv.org/abs/2402.03478)

    本研究引入了一种新的集合方法，超扩散，可以使用单一模型准确估计认识和偶然不确定性。

    

    在将机器学习应用于高风险应用领域（如医学影像和天气预报）时，准确估计和区分认识不确定性（可以通过更多的训练数据降低的不确定性）和偶然不确定性（与当前任务固有的不确定性）至关重要。条件扩散模型具有准确有效地从数据集的后验分布中采样的突破性能力，现在使得不确定性估计从概念上变得简单明了：只需要训练和从一个大型扩散模型集合中采样即可。然而，随着模型架构的复杂性增加，训练这样一个集合变得难以计算。在本文中，我们介绍了一种新的集合方法，超扩散，它可以使用单一模型准确估计认识和偶然不确定性。

    Estimating and disentangling epistemic uncertainty (uncertainty that can be reduced with more training data) and aleatoric uncertainty (uncertainty that is inherent to the task at hand) is critically important when applying machine learning (ML) to high-stakes applications such as medical imaging and weather forecasting. Conditional diffusion models' breakthrough ability to accurately and efficiently sample from the posterior distribution of a dataset now makes uncertainty estimation conceptually straightforward: One need only train and sample from a large ensemble of diffusion models. Unfortunately, training such an ensemble becomes computationally intractable as the complexity of the model architecture grows.   In this work we introduce a new approach to ensembling, hyper-diffusion, which allows one to accurately estimate epistemic and aleatoric uncertainty with a single model. Unlike existing Monte Carlo dropout based single-model ensembling methods, hyper-diffusion offers the same 
    
[^2]: GD无法胜任：非可微性对神经网络训练的三种影响方式

    GD doesn't make the cut: Three ways that non-differentiability affects neural network training

    [https://arxiv.org/abs/2401.08426](https://arxiv.org/abs/2401.08426)

    本文研究了非可微性对神经网络训练的影响，包括收敛性差异、$L_1$正则化问题的矛盾性质以及稳定边界现象的不适用性。

    

    本文研究了应用于非可微函数（NGDMs）和应用于可微函数的传统梯度下降（GDs）之间的区别。首先，我们证明了NGDMs的收敛性质与GDs存在显著差异，挑战了基于$L$-光滑性的广泛神经网络收敛文献对非光滑神经网络的适用性。接下来，我们展示了NGDM解决$L_1$正则化问题的矛盾性质，表明增加正则化惩罚会导致NGDMs中最优解的$L_1$范数增加。因此，我们证明了广泛采用的基于$L_1$惩罚的网络修剪技术并未产生预期结果。最后，我们探索了稳定边界现象（Edge of Stability），指出即使对于Lipschitz连续凸可微函数，它也不适用于非凸非可微的神经网络。

    This paper investigates the distinctions between gradient methods applied to non-differentiable functions (NGDMs) and classical gradient descents (GDs) designed for differentiable functions. First, we demonstrate significant differences in the convergence properties of NGDMs compared to GDs, challenging the applicability of the extensive neural network convergence literature based on $L-smoothness$ to non-smooth neural networks. Next, we demonstrate the paradoxical nature of NGDM solutions for $L_{1}$-regularized problems, showing that increasing the regularization penalty leads to an increase in the $L_{1}$ norm of optimal solutions in NGDMs. Consequently, we show that widely adopted $L_{1}$ penalization-based techniques for network pruning do not yield expected results. Finally, we explore the Edge of Stability phenomenon, indicating its inapplicability even to Lipschitz continuous convex differentiable functions, leaving its relevance to non-convex non-differentiable neural networks
    
[^3]: 心室分割：U-Net衍生模型的简要比较

    Ventricular Segmentation: A Brief Comparison of U-Net Derivatives. (arXiv:2401.09980v1 [eess.IV])

    [http://arxiv.org/abs/2401.09980](http://arxiv.org/abs/2401.09980)

    本文探讨了深度学习技术在心脏图像分割中的应用，实施了多个U-Net衍生模型以实现对心脏特定部位的全面解剖和功能分析。通过图像、图表和定量指标验证了模型的效果，并讨论了面临的挑战和未来改进策略。

    

    医学影像是指用于观察人体及其内部的技术和方法，以诊断、监测甚至治疗医学疾病。本文旨在探讨深度学习技术在心脏短轴磁共振成像图像的语义分割中的应用，旨在提高与心脏相关的医学疾病的诊断、监测和治疗。重点是实施各种U-Net的衍生体系结构，以有效地分离心脏的特定部分，进行全面的解剖和功能分析。通过图像、图表和定量指标的组合展示了模型及其预测的效果。此外，本文还讨论了遇到的挑战，并概述了未来改进的策略。本摘要简要概述了利用深度学习进行心脏图像分割的工作，强调了模型的有效性。

    Medical imaging refers to the technologies and methods utilized to view the human body and its inside, in order to diagnose, monitor, or even treat medical disorders. This paper aims to explore the application of deep learning techniques in the semantic segmentation of Cardiac short-axis MRI (Magnetic Resonance Imaging) images, aiming to enhance the diagnosis, monitoring, and treatment of medical disorders related to the heart. The focus centers on implementing various architectures that are derivatives of U-Net, to effectively isolate specific parts of the heart for comprehensive anatomical and functional analysis. Through a combination of images, graphs, and quantitative metrics, the efficacy of the models and their predictions are showcased. Additionally, this paper addresses encountered challenges and outline strategies for future improvements. This abstract provides a concise overview of the efforts in utilizing deep learning for cardiac image segmentation, emphasizing both the ac
    
[^4]: 健壮的多模态模型具有异常特征并编码更多概念

    Robust multimodal models have outlier features and encode more concepts. (arXiv:2310.13040v1 [cs.LG])

    [http://arxiv.org/abs/2310.13040](http://arxiv.org/abs/2310.13040)

    健壮的多模态模型展示了异常特征和更多概念的编码方式。

    

    什么区分健壮模型与非健壮模型？随着大规模多模态模型（如CLIP）的出现，这个问题引起了人们的关注。这些模型在自然分布转变方面表现出了前所未有的健壮性。尽管已经证明了健壮性的差异可以追溯到训练数据上的差异，但迄今为止还不清楚这对于模型学习到了什么意味着。在这项工作中，我们通过探测12个具有不同骨干（ResNets和ViTs）和预训练集（OpenAI，LAION-400M，LAION-2B，YFCC15M，CC12M和DataComp）的健壮多模态模型的表示空间来填补这一空白。我们发现这些模型的表示空间中存在两个健壮性的特征：（1）健壮模型具有由其激活特征表征的异常特征，其中一些特征值比平均值高几个数量级。这些异常特征在模型的表示空间中引入了特权方向。我们证明了...

    What distinguishes robust models from non-robust ones? This question has gained traction with the appearance of large-scale multimodal models, such as CLIP. These models have demonstrated unprecedented robustness with respect to natural distribution shifts. While it has been shown that such differences in robustness can be traced back to differences in training data, so far it is not known what that translates to in terms of what the model has learned. In this work, we bridge this gap by probing the representation spaces of 12 robust multimodal models with various backbones (ResNets and ViTs) and pretraining sets (OpenAI, LAION-400M, LAION-2B, YFCC15M, CC12M and DataComp). We find two signatures of robustness in the representation spaces of these models: (1) Robust models exhibit outlier features characterized by their activations, with some being several orders of magnitude above average. These outlier features induce privileged directions in the model's representation space. We demon
    
[^5]: 对齐语言的视觉表示预测人类在自然学习任务中的行为

    Language Aligned Visual Representations Predict Human Behavior in Naturalistic Learning Tasks. (arXiv:2306.09377v1 [cs.LG])

    [http://arxiv.org/abs/2306.09377](http://arxiv.org/abs/2306.09377)

    语言对齐的视觉表示方式比纯视觉表示方式更有效地预测人类在自然学习任务中的行为。

    

    人类具备识别和概括自然物体相关特征的能力，在各种情境中有所帮助。为了研究这种现象并确定最有效的表示方式以预测人类行为，我们进行了两个涉及类别学习和奖励学习的实验。我们的实验使用逼真的图像作为刺激物，并要求参与者基于所有试验的新型刺激物作出准确的决策，因此需要泛化。在两个任务中，底层规则是使用人类相似性判断提取的刺激维度生成的简单线性函数。值得注意的是，参与者在几次试验内就成功地确定了相关的刺激特征，证明了有效的泛化。我们进行了广泛的模型比较，评估了各种深度学习模型的表示对人类选择的逐次预测准确性。有趣的是，自然语言处理任务（如语言建模和机器翻译）训练的模型表示优于视觉任务训练的模型表示，表明对齐语言的视觉表示可能更有效地预测人类在自然学习任务中的行为。

    Humans possess the ability to identify and generalize relevant features of natural objects, which aids them in various situations. To investigate this phenomenon and determine the most effective representations for predicting human behavior, we conducted two experiments involving category learning and reward learning. Our experiments used realistic images as stimuli, and participants were tasked with making accurate decisions based on novel stimuli for all trials, thereby necessitating generalization. In both tasks, the underlying rules were generated as simple linear functions using stimulus dimensions extracted from human similarity judgments. Notably, participants successfully identified the relevant stimulus features within a few trials, demonstrating effective generalization. We performed an extensive model comparison, evaluating the trial-by-trial predictive accuracy of diverse deep learning models' representations of human choices. Intriguingly, representations from models train
    

