# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Extract Mechanisms from Heterogeneous Effects: Identification Strategy for Mediation Analysis](https://arxiv.org/abs/2403.04131) | 该论文开发了一种新的识别策略，通过将总处理效应进行分解，将复杂的中介问题转化为简单的线性回归问题，实现了因果效应和中介效应的同时估计，建立了新的因果中介和因果调节之间的联系。 |
| [^2] | [Zeroth-Order Sampling Methods for Non-Log-Concave Distributions: Alleviating Metastability by Denoising Diffusion](https://arxiv.org/abs/2402.17886) | 本文提出了一种基于去噪扩散过程的零阶扩散蒙特卡洛算法，克服了非对数凹分布采样中的亚稳定性问题，并证明其采样精度具有倒多项式依赖。 |
| [^3] | [Unlocking the Power of Multi-institutional Data: Integrating and Harmonizing Genomic Data Across Institutions](https://arxiv.org/abs/2402.00077) | 该研究介绍了一种名为Bridge的模型，致力于解决利用多机构测序数据时面临的挑战，包括基因组板块的变化、测序技术的差异以及数据的高维度和稀疏性等。 |

# 详细

[^1]: 从异质效应中提取机制：中介分析的识别策略

    Extract Mechanisms from Heterogeneous Effects: Identification Strategy for Mediation Analysis

    [https://arxiv.org/abs/2403.04131](https://arxiv.org/abs/2403.04131)

    该论文开发了一种新的识别策略，通过将总处理效应进行分解，将复杂的中介问题转化为简单的线性回归问题，实现了因果效应和中介效应的同时估计，建立了新的因果中介和因果调节之间的联系。

    

    理解因果机制对于解释和概括经验现象至关重要。因果中介分析提供了量化中介效应的统计技术。然而，现有方法通常需要强大的识别假设或复杂的研究设计。我们开发了一种新的识别策略，简化了这些假设，实现了因果效应和中介效应的同时估计。该策略基于总处理效应的新型分解，将具有挑战性的中介问题转化为简单的线性回归问题。新方法建立了因果中介和因果调节之间的新联系。我们讨论了几种研究设计和估计器，以增加我们的识别策略在各种实证研究中的可用性。我们通过在实验中估计因果中介效应来演示我们方法的应用。

    arXiv:2403.04131v1 Announce Type: cross  Abstract: Understanding causal mechanisms is essential for explaining and generalizing empirical phenomena. Causal mediation analysis offers statistical techniques to quantify mediation effects. However, existing methods typically require strong identification assumptions or sophisticated research designs. We develop a new identification strategy that simplifies these assumptions, enabling the simultaneous estimation of causal and mediation effects. The strategy is based on a novel decomposition of total treatment effects, which transforms the challenging mediation problem into a simple linear regression problem. The new method establishes a new link between causal mediation and causal moderation. We discuss several research designs and estimators to increase the usability of our identification strategy for a variety of empirical studies. We demonstrate the application of our method by estimating the causal mediation effect in experiments concer
    
[^2]: 用于非对数凹分布的零阶采样方法：通过去噪扩散缓解亚稳定性

    Zeroth-Order Sampling Methods for Non-Log-Concave Distributions: Alleviating Metastability by Denoising Diffusion

    [https://arxiv.org/abs/2402.17886](https://arxiv.org/abs/2402.17886)

    本文提出了一种基于去噪扩散过程的零阶扩散蒙特卡洛算法，克服了非对数凹分布采样中的亚稳定性问题，并证明其采样精度具有倒多项式依赖。

    

    这篇论文考虑了基于其非对数凹分布未归一化密度查询的采样问题。首先描述了一个基于模拟去噪扩散过程的框架，即扩散蒙特卡洛（DMC），其得分函数通过通用蒙特卡洛估计器逼近。DMC是一个基于神谕的元算法，其中神谕是假设可以访问生成蒙特卡洛分数估计器的样本的访问。然后，我们提供了一个基于拒绝采样的这个神谕的实现，这将DMC转化为一个真正的算法，称为零阶扩散蒙特卡洛（ZOD-MC）。我们通过首先构建一个通用框架，即DMC的性能保证，而不假设目标分布为对数凹或满足任何等周不等式，提供了收敛分析。然后我们证明ZOD-MC对所需采样精度具有倒多项式依赖，尽管仍然受到...

    arXiv:2402.17886v1 Announce Type: cross  Abstract: This paper considers the problem of sampling from non-logconcave distribution, based on queries of its unnormalized density. It first describes a framework, Diffusion Monte Carlo (DMC), based on the simulation of a denoising diffusion process with its score function approximated by a generic Monte Carlo estimator. DMC is an oracle-based meta-algorithm, where its oracle is the assumed access to samples that generate a Monte Carlo score estimator. Then we provide an implementation of this oracle, based on rejection sampling, and this turns DMC into a true algorithm, termed Zeroth-Order Diffusion Monte Carlo (ZOD-MC). We provide convergence analyses by first constructing a general framework, i.e. a performance guarantee for DMC, without assuming the target distribution to be log-concave or satisfying any isoperimetric inequality. Then we prove that ZOD-MC admits an inverse polynomial dependence on the desired sampling accuracy, albeit sti
    
[^3]: 多机构数据的释放力量：整合和协调跨机构的基因组数据

    Unlocking the Power of Multi-institutional Data: Integrating and Harmonizing Genomic Data Across Institutions

    [https://arxiv.org/abs/2402.00077](https://arxiv.org/abs/2402.00077)

    该研究介绍了一种名为Bridge的模型，致力于解决利用多机构测序数据时面临的挑战，包括基因组板块的变化、测序技术的差异以及数据的高维度和稀疏性等。

    

    癌症是由基因突变驱动的复杂疾病，肿瘤测序已成为癌症患者临床护理的重要手段。出现的多机构测序数据为学习真实世界的证据以增强精准肿瘤医学提供了强大的资源。由美国癌症研究协会领导的GENIE BPC建立了一个独特的数据库，将基因组数据与多个癌症中心的临床信息联系起来。然而，利用这种多机构测序数据面临着重大挑战。基因组板块的变化导致在使用常见基因集进行分析时信息丢失。此外，不同的测序技术和机构之间的患者异质性增加了复杂性。高维数据、稀疏基因突变模式以及个体基因水平上的弱信号进一步增加了问题的复杂性。在这些现实世界的挑战的推动下，我们引入了Bridge模型。

    Cancer is a complex disease driven by genomic alterations, and tumor sequencing is becoming a mainstay of clinical care for cancer patients. The emergence of multi-institution sequencing data presents a powerful resource for learning real-world evidence to enhance precision oncology. GENIE BPC, led by the American Association for Cancer Research, establishes a unique database linking genomic data with clinical information for patients treated at multiple cancer centers. However, leveraging such multi-institutional sequencing data presents significant challenges. Variations in gene panels result in loss of information when the analysis is conducted on common gene sets. Additionally, differences in sequencing techniques and patient heterogeneity across institutions add complexity. High data dimensionality, sparse gene mutation patterns, and weak signals at the individual gene level further complicate matters. Motivated by these real-world challenges, we introduce the Bridge model. It use
    

