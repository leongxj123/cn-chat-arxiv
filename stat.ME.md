# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DR-VIDAL -- Doubly Robust Variational Information-theoretic Deep Adversarial Learning for Counterfactual Prediction and Treatment Effect Estimation on Real World Data.](http://arxiv.org/abs/2303.04201) | DR-VIDAL是一个新型的生成框架，可用于处理真实世界数据中的干预措施对结果的因果效应估计，并具有处理混淆偏差和模型不良的能力。 |

# 详细

[^1]: DR-VIDAL--双重稳健变分信息论深度对抗学习用于真实世界数据的反事实预测和治疗效果估计

    DR-VIDAL -- Doubly Robust Variational Information-theoretic Deep Adversarial Learning for Counterfactual Prediction and Treatment Effect Estimation on Real World Data. (arXiv:2303.04201v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.04201](http://arxiv.org/abs/2303.04201)

    DR-VIDAL是一个新型的生成框架，可用于处理真实世界数据中的干预措施对结果的因果效应估计，并具有处理混淆偏差和模型不良的能力。

    

    从真实世界的观察性（非随机化）数据中确定干预措施对结果的因果效应，例如使用电子健康记录的治疗重用，由于潜在偏差而具有挑战性。因果深度学习已经改进了传统技术，用于估计个性化治疗效果（ITE）。我们提出了双重稳健变分信息论深度对抗学习（DR-VIDAL），这是一个结合了治疗和结果两个联合模型的新型生成框架，确保无偏的ITE估计，即使其中一个模型设定不正确。DR-VIDAL整合了： （i）变分自编码器（VAE）根据因果假设将混淆变量分解为潜在变量; （ii）基于信息论的生成对抗网络（Info-GAN）用于生成反事实情况; （iii）一个双重稳健块，其中包括治疗倾向于预测结果。在合成和真实数据集（Infant Health和Development Program，Transforming Clinical Practice Initiative [TCPI]）中进行实验，我们证明了DR-VIDAL在估计ITE方面优于现有的最先进方法，因为它具有处理混淆偏差和模型不正确的能力。

    Determining causal effects of interventions onto outcomes from real-world, observational (non-randomized) data, e.g., treatment repurposing using electronic health records, is challenging due to underlying bias. Causal deep learning has improved over traditional techniques for estimating individualized treatment effects (ITE). We present the Doubly Robust Variational Information-theoretic Deep Adversarial Learning (DR-VIDAL), a novel generative framework that combines two joint models of treatment and outcome, ensuring an unbiased ITE estimation even when one of the two is misspecified. DR-VIDAL integrates: (i) a variational autoencoder (VAE) to factorize confounders into latent variables according to causal assumptions; (ii) an information-theoretic generative adversarial network (Info-GAN) to generate counterfactuals; (iii) a doubly robust block incorporating treatment propensities for outcome predictions. On synthetic and real-world datasets (Infant Health and Development Program, T
    

