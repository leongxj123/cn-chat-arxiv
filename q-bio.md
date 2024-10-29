# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Antigen-Specific Antibody Design via Direct Energy-based Preference Optimization](https://arxiv.org/abs/2403.16576) | 通过直接基于能量偏好优化的方法，解决了抗原特异性抗体设计中的蛋白质序列-结构共设计问题，以生成具有理性结构和良好结合亲和力的抗体设计。 |
| [^2] | [Not all tickets are equal and we know it: Guiding pruning with domain-specific knowledge](https://arxiv.org/abs/2403.04805) | 使用领域特定结构信息来引导修剪的方法 DASH 在学习动态基因调控网络模型时表现出色，提供了更有意义的生物学见解 |
| [^3] | [Brain Networks and Intelligence: A Graph Neural Network Based Approach to Resting State fMRI Data](https://arxiv.org/abs/2311.03520) | 本文提出了一种新颖的BrainRGIN建模架构，使用图神经网络来预测智力，扩展了现有的图卷积网络并结合了聚类嵌入、图同构网络、TopK池化和基于注意力的读出函数。 |
| [^4] | [Diffusion Variational Inference: Diffusion Models as Expressive Variational Posteriors.](http://arxiv.org/abs/2401.02739) | 本文提出了去噪扩散变分推断（DDVI）算法，该算法使用扩散模型作为表达性变分后验，并通过反转加噪过程在潜空间中进行扩散。该方法易于实现，兼容黑盒变分推断，并在深度潜变量模型中的任务中表现优异。 |

# 详细

[^1]: 通过直接基于能量偏好优化的抗原特异性抗体设计

    Antigen-Specific Antibody Design via Direct Energy-based Preference Optimization

    [https://arxiv.org/abs/2403.16576](https://arxiv.org/abs/2403.16576)

    通过直接基于能量偏好优化的方法，解决了抗原特异性抗体设计中的蛋白质序列-结构共设计问题，以生成具有理性结构和良好结合亲和力的抗体设计。

    

    抗体设计是一个至关重要的任务，对各种领域都有重要影响，如治疗和生物学，由于其错综复杂的性质，面临着相当大的挑战。在本文中，我们将抗原特异性抗体设计作为一个蛋白质序列-结构共设计问题，考虑了理性和功能性。利用一个预先训练的条件扩散模型，该模型联合建模抗体中互补决定区（CDR）的序列和结构，并结合了等变神经网络，我们提出了直接基于能量偏好优化的方法，以引导生成既具有合理结构又具有明显结合亲和力的抗体。我们的方法涉及使用残基级分解能量偏好对预先训练的扩散模型进行微调。此外，我们采用梯度手术来解决各种类型能量之间的冲突，例如吸引和斥

    arXiv:2403.16576v1 Announce Type: cross  Abstract: Antibody design, a crucial task with significant implications across various disciplines such as therapeutics and biology, presents considerable challenges due to its intricate nature. In this paper, we tackle antigen-specific antibody design as a protein sequence-structure co-design problem, considering both rationality and functionality. Leveraging a pre-trained conditional diffusion model that jointly models sequences and structures of complementarity-determining regions (CDR) in antibodies with equivariant neural networks, we propose direct energy-based preference optimization to guide the generation of antibodies with both rational structures and considerable binding affinities to given antigens. Our method involves fine-tuning the pre-trained diffusion model using a residue-level decomposed energy preference. Additionally, we employ gradient surgery to address conflicts between various types of energy, such as attraction and repu
    
[^2]: 不是所有的票据都是平等的，而我们知道：用领域特定知识来引导修剪

    Not all tickets are equal and we know it: Guiding pruning with domain-specific knowledge

    [https://arxiv.org/abs/2403.04805](https://arxiv.org/abs/2403.04805)

    使用领域特定结构信息来引导修剪的方法 DASH 在学习动态基因调控网络模型时表现出色，提供了更有意义的生物学见解

    

    神经结构学习对于科学发现和可解释性至关重要。然而，当代侧重于计算资源效率的修剪算法在选择符合领域专业知识的有意义模型方面面临算法障碍。为了减轻这一挑战，我们提出了DASH，利用可用的领域特定结构信息来引导修剪。在学习动态基因调控网络模型的背景下，我们展示了DASH与现有一般知识相结合，提供了与生物学一致的数据特定见解。对于这一任务，我们展示了在具有地面真实信息的合成数据和两个真实世界应用中，DASH的有效性，其优于竞争方法很大，并提供了更有意义的生物学见解。我们的工作表明，领域特定的结构信息具有提高模型衍生科学洞见的潜力。

    arXiv:2403.04805v1 Announce Type: new  Abstract: Neural structure learning is of paramount importance for scientific discovery and interpretability. Yet, contemporary pruning algorithms that focus on computational resource efficiency face algorithmic barriers to select a meaningful model that aligns with domain expertise. To mitigate this challenge, we propose DASH, which guides pruning by available domain-specific structural information. In the context of learning dynamic gene regulatory network models, we show that DASH combined with existing general knowledge on interaction partners provides data-specific insights aligned with biology. For this task, we show on synthetic data with ground truth information and two real world applications the effectiveness of DASH, which outperforms competing methods by a large margin and provides more meaningful biological insights. Our work shows that domain specific structural information bears the potential to improve model-derived scientific insi
    
[^3]: 大脑网络与智力：基于图神经网络的静息态fMRI数据方法

    Brain Networks and Intelligence: A Graph Neural Network Based Approach to Resting State fMRI Data

    [https://arxiv.org/abs/2311.03520](https://arxiv.org/abs/2311.03520)

    本文提出了一种新颖的BrainRGIN建模架构，使用图神经网络来预测智力，扩展了现有的图卷积网络并结合了聚类嵌入、图同构网络、TopK池化和基于注意力的读出函数。

    

    静息态功能磁共振成像（rsfMRI）是一种研究大脑功能和认知过程关系的强大工具，因为它可以捕获大脑的功能组织，而无需依赖于特定任务或刺激。本文提出了一种称为BrainRGIN的新颖建模架构，利用rsfMRI推导的静态功能网络连接矩阵，基于图神经网络预测智力（流体、晶体和总体智力）。我们的方法扩展了现有的图卷积网络，将聚类嵌入和图同构网络纳入到图卷积层中，以反映大脑子网络组织的性质和高效网络表达，再辅以TopK池化和基于注意力的读出函数。我们在一个大型数据集上评估了我们提出的架构。

    arXiv:2311.03520v2 Announce Type: replace-cross  Abstract: Resting-state functional magnetic resonance imaging (rsfMRI) is a powerful tool for investigating the relationship between brain function and cognitive processes as it allows for the functional organization of the brain to be captured without relying on a specific task or stimuli. In this paper, we present a novel modeling architecture called BrainRGIN for predicting intelligence (fluid, crystallized, and total intelligence) using graph neural networks on rsfMRI derived static functional network connectivity matrices. Extending from the existing graph convolution networks, our approach incorporates a clustering-based embedding and graph isomorphism network in the graph convolutional layer to reflect the nature of the brain sub-network organization and efficient network expression, in combination with TopK pooling and attention-based readout functions. We evaluated our proposed architecture on a large dataset, specifically the A
    
[^4]: 扩散变分推断：扩散模型作为表达性变分后验

    Diffusion Variational Inference: Diffusion Models as Expressive Variational Posteriors. (arXiv:2401.02739v1 [cs.LG])

    [http://arxiv.org/abs/2401.02739](http://arxiv.org/abs/2401.02739)

    本文提出了去噪扩散变分推断（DDVI）算法，该算法使用扩散模型作为表达性变分后验，并通过反转加噪过程在潜空间中进行扩散。该方法易于实现，兼容黑盒变分推断，并在深度潜变量模型中的任务中表现优异。

    

    我们提出了去噪扩散变分推断（DDVI），一种用扩散模型作为表达性变分后验的潜变量模型的近似推断算法。我们的方法通过辅助潜变量增加了变分后验，从而得到一个表达性的模型类，通过反转用户指定的加噪过程在潜空间中进行扩散。我们通过优化一个受到觉醒-睡眠算法启发的边际似然新下界来拟合这些模型。我们的方法易于实现（它适配了正则化的ELBO扩展），与黑盒变分推断兼容，并且表现优于基于归一化流或对抗网络的替代近似后验类别。将我们的方法应用于深度潜变量模型时，我们的方法得到了去噪扩散变分自动编码器（DD-VAE）算法。我们将该算法应用于生物学中的一个激励任务 -- 从人类基因组中推断潜在血统 -- 超过了强基线模型。

    We propose denoising diffusion variational inference (DDVI), an approximate inference algorithm for latent variable models which relies on diffusion models as expressive variational posteriors. Our method augments variational posteriors with auxiliary latents, which yields an expressive class of models that perform diffusion in latent space by reversing a user-specified noising process. We fit these models by optimizing a novel lower bound on the marginal likelihood inspired by the wake-sleep algorithm. Our method is easy to implement (it fits a regularized extension of the ELBO), is compatible with black-box variational inference, and outperforms alternative classes of approximate posteriors based on normalizing flows or adversarial networks. When applied to deep latent variable models, our method yields the denoising diffusion VAE (DD-VAE) algorithm. We use this algorithm on a motivating task in biology -- inferring latent ancestry from human genomes -- outperforming strong baselines
    

