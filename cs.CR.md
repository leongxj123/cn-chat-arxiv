# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bounding the Excess Risk for Linear Models Trained on Marginal-Preserving, Differentially-Private, Synthetic Data](https://arxiv.org/abs/2402.04375) | 本文提出了在保持边缘一致的差分隐私合成数据上训练线性模型的过量风险的新界限，为连续和Lipschitz损失函数提供了上界和下界。 |
| [^2] | [Samplable Anonymous Aggregation for Private Federated Data Analysis.](http://arxiv.org/abs/2307.15017) | 本论文在解决每个设备持有私有数据情况下的私有统计和私有联邦学习设计中，提出了一个简单的原语，以实现高效的算法，并在不需要强信任假设的情况下保护隐私。 |
| [^3] | [Differentially Private Latent Diffusion Models.](http://arxiv.org/abs/2305.15759) | 本文提出使用差分隐私训练潜在扩散模型(LDMs)，通过预训练自编码器将高维像素空间转变为低维潜在空间实现更高效快速的DMs训练，并且通过只微调注意力模块减少了可训练参数的数量。 |

# 详细

[^1]: 在保持边缘一致的差分隐私合成数据上训练线性模型的过量风险界限

    Bounding the Excess Risk for Linear Models Trained on Marginal-Preserving, Differentially-Private, Synthetic Data

    [https://arxiv.org/abs/2402.04375](https://arxiv.org/abs/2402.04375)

    本文提出了在保持边缘一致的差分隐私合成数据上训练线性模型的过量风险的新界限，为连续和Lipschitz损失函数提供了上界和下界。

    

    机器学习的广泛应用引发了人们对于模型可能揭示训练数据中个体的私密信息的担忧。为了防止敏感数据的泄露，我们考虑使用差分隐私的合成训练数据而不是真实训练数据来训练机器学习模型。合成数据的一个关键优点是能够保持原始分布的低阶边缘特征。我们的主要贡献是针对在这种合成数据上训练的线性模型，针对连续和Lipschitz损失函数提出了新的过量经验风险的上界和下界。我们在理论结果之外进行了大量实验。

    The growing use of machine learning (ML) has raised concerns that an ML model may reveal private information about an individual who has contributed to the training dataset. To prevent leakage of sensitive data, we consider using differentially-private (DP), synthetic training data instead of real training data to train an ML model. A key desirable property of synthetic data is its ability to preserve the low-order marginals of the original distribution. Our main contribution comprises novel upper and lower bounds on the excess empirical risk of linear models trained on such synthetic data, for continuous and Lipschitz loss functions. We perform extensive experimentation alongside our theoretical results.
    
[^2]: 私有联邦数据分析的可采样匿名聚合

    Samplable Anonymous Aggregation for Private Federated Data Analysis. (arXiv:2307.15017v1 [cs.CR])

    [http://arxiv.org/abs/2307.15017](http://arxiv.org/abs/2307.15017)

    本论文在解决每个设备持有私有数据情况下的私有统计和私有联邦学习设计中，提出了一个简单的原语，以实现高效的算法，并在不需要强信任假设的情况下保护隐私。

    

    在每个设备持有私有数据的情况下，我们重新审视设计可扩展的私有统计协议和私有联邦学习的问题。我们的第一个贡献是提出了一个简单的原语，可以有效地实现几种常用算法，并且可以在不需要强信任假设的情况下进行隐私账务，接近于集中设置中的隐私保护。其次，我们提出了一个实现该原语的系统架构，并对该系统进行了安全性分析。

    We revisit the problem of designing scalable protocols for private statistics and private federated learning when each device holds its private data. Our first contribution is to propose a simple primitive that allows for efficient implementation of several commonly used algorithms, and allows for privacy accounting that is close to that in the central setting without requiring the strong trust assumptions it entails. Second, we propose a system architecture that implements this primitive and perform a security analysis of the proposed system.
    
[^3]: 差分隐私潜在扩散模型

    Differentially Private Latent Diffusion Models. (arXiv:2305.15759v1 [stat.ML])

    [http://arxiv.org/abs/2305.15759](http://arxiv.org/abs/2305.15759)

    本文提出使用差分隐私训练潜在扩散模型(LDMs)，通过预训练自编码器将高维像素空间转变为低维潜在空间实现更高效快速的DMs训练，并且通过只微调注意力模块减少了可训练参数的数量。

    

    扩散模型(DMs)被广泛用于生成高质量图像数据集。然而，由于它们直接在高维像素空间中运行，DMs的优化计算成本高，需要长时间的训练。这导致由于差分隐私的可组合性属性，大量噪音注入到差分隐私学习过程中。为了解决这个挑战，我们提出使用差分隐私训练潜在扩散模型(LDMs)。LDMs使用强大的预训练自编码器将高维像素空间减少到更低维的潜在空间，使训练DMs更加高效和快速。与[Ghalebikesabi等人，2023]预先用公共数据预训练DMs，然后再用隐私数据进行微调不同，我们仅微调LDMs中不同层的注意力模块以获得隐私敏感数据，相对于整个DM微调，可减少大约96%的可训练参数数量。

    Diffusion models (DMs) are widely used for generating high-quality image datasets. However, since they operate directly in the high-dimensional pixel space, optimization of DMs is computationally expensive, requiring long training times. This contributes to large amounts of noise being injected into the differentially private learning process, due to the composability property of differential privacy. To address this challenge, we propose training Latent Diffusion Models (LDMs) with differential privacy. LDMs use powerful pre-trained autoencoders to reduce the high-dimensional pixel space to a much lower-dimensional latent space, making training DMs more efficient and fast. Unlike [Ghalebikesabi et al., 2023] that pre-trains DMs with public data then fine-tunes them with private data, we fine-tune only the attention modules of LDMs at varying layers with privacy-sensitive data, reducing the number of trainable parameters by approximately 96% compared to fine-tuning the entire DM. We te
    

