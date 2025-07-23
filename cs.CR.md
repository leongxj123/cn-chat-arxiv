# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Quantifying and Mitigating Privacy Risks for Tabular Generative Models](https://arxiv.org/abs/2403.07842) | 该论文研究了量化和减轻表格生成模型的隐私风险，通过对五种最先进的表格合成器进行实证分析，提出了差分隐私表格潜在扩散模型。 |

# 详细

[^1]: 量化和减轻表格生成模型的隐私风险

    Quantifying and Mitigating Privacy Risks for Tabular Generative Models

    [https://arxiv.org/abs/2403.07842](https://arxiv.org/abs/2403.07842)

    该论文研究了量化和减轻表格生成模型的隐私风险，通过对五种最先进的表格合成器进行实证分析，提出了差分隐私表格潜在扩散模型。

    

    针对合成数据生成模型出现作为保护隐私的数据共享解决方案的情况，该合成数据集应该类似于原始数据，而不会透露可识别的私人信息。表格合成器的核心技术根植于图像生成模型，范围从生成对抗网络（GAN）到最近的扩散模型。最近的先前工作揭示和量化了表格数据上的效用-隐私权衡，揭示了合成数据的隐私风险。我们首先进行了详尽的实证分析，突出了五种最先进的表格合成器针对八种隐私攻击的效用-隐私权衡，特别关注成员推断攻击。在观察到表格扩散中高数据质量但也高隐私风险的情况下，我们提出了DP-TLDM，差分隐私表格潜在扩散模型，由自动编码器网络组成。

    arXiv:2403.07842v1 Announce Type: new  Abstract: Synthetic data from generative models emerges as the privacy-preserving data-sharing solution. Such a synthetic data set shall resemble the original data without revealing identifiable private information. The backbone technology of tabular synthesizers is rooted in image generative models, ranging from Generative Adversarial Networks (GANs) to recent diffusion models. Recent prior work sheds light on the utility-privacy tradeoff on tabular data, revealing and quantifying privacy risks on synthetic data. We first conduct an exhaustive empirical analysis, highlighting the utility-privacy tradeoff of five state-of-the-art tabular synthesizers, against eight privacy attacks, with a special focus on membership inference attacks. Motivated by the observation of high data quality but also high privacy risk in tabular diffusion, we propose DP-TLDM, Differentially Private Tabular Latent Diffusion Model, which is composed of an autoencoder networ
    

