# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Stylized Facts of High-Frequency Bitcoin Time Series](https://arxiv.org/abs/2402.11930) | 本文研究了2019年至2022年间的高频比特币日内数据集，发现比特币价格回报表现出了两个不同时期的异常扩散过程，具有重尾特征和时间依赖性。 |
| [^2] | [A Bayesian Non-parametric Approach to Generative Models: Integrating Variational Autoencoder and Generative Adversarial Networks using Wasserstein and Maximum Mean Discrepancy.](http://arxiv.org/abs/2308.14048) | 本研究提出了一种融合生成对抗网络和变分自编码器的贝叶斯非参数方法，通过在损失函数中使用Wasserstein和最大均值差异度量，实现了对潜在空间的有效学习，并能够生成多样且高质量的样本。 |

# 详细

[^1]: 高频比特币时间序列的风格化事实

    Stylized Facts of High-Frequency Bitcoin Time Series

    [https://arxiv.org/abs/2402.11930](https://arxiv.org/abs/2402.11930)

    本文研究了2019年至2022年间的高频比特币日内数据集，发现比特币价格回报表现出了两个不同时期的异常扩散过程，具有重尾特征和时间依赖性。

    

    本文分析了2019年至2022年间的高频比特币日内数据集。在此期间，比特币市场指数表现出两个明显时期，其特征是波动性的突然变化。两个时期的比特币价格回报可用异常扩散过程来描述，从短时间间隔下的次扩散过渡到长时间间隔上的弱超扩散。本文研究的与这种异常行为相关的特征包括重尾，可以用$q$-高斯分布和相关性来描述。当对绝对回报的自相关进行取样时，我们观察到一种幂律关系，表明两个时期中最初存在时间依赖性。回报的整体自相关迅速衰减并表现出周期性。我们将自相关拟合为幂律和余弦函数，以捕捉衰减和波动，并发现了这两种函数的特征

    arXiv:2402.11930v1 Announce Type: new  Abstract: This paper analyses the high-frequency intraday Bitcoin dataset from 2019 to 2022. During this time frame, the Bitcoin market index exhibited two distinct periods characterized by abrupt changes in volatility. The Bitcoin price returns for both periods can be described by an anomalous diffusion process, transitioning from subdiffusion for short intervals to weak superdiffusion over longer time intervals. The characteristic features related to this anomalous behavior studied in the present paper include heavy tails, which can be described using a $q$-Gaussian distribution and correlations. When we sample the autocorrelation of absolute returns, we observe a power-law relationship, indicating time dependency in both periods initially. The ensemble autocorrelation of returns decays rapidly and exhibits periodicity. We fitted the autocorrelation with a power law and a cosine function to capture both the decay and the fluctuation and found th
    
[^2]: 一种贝叶斯非参数方法用于生成模型：使用Wasserstein和最大均值差异度量集成变分自编码器和生成对抗网络

    A Bayesian Non-parametric Approach to Generative Models: Integrating Variational Autoencoder and Generative Adversarial Networks using Wasserstein and Maximum Mean Discrepancy. (arXiv:2308.14048v1 [stat.ML])

    [http://arxiv.org/abs/2308.14048](http://arxiv.org/abs/2308.14048)

    本研究提出了一种融合生成对抗网络和变分自编码器的贝叶斯非参数方法，通过在损失函数中使用Wasserstein和最大均值差异度量，实现了对潜在空间的有效学习，并能够生成多样且高质量的样本。

    

    生成模型已成为一种产生与真实图像难以区分的高质量图像的有前途的技术。生成对抗网络（GAN）和变分自编码器（VAE）是最为重要且被广泛研究的两种生成模型。GAN在生成逼真图像方面表现出色，而VAE则能够生成多样的图像。然而，GAN忽视了大部分可能的输出空间，这导致不能完全体现目标分布的多样性，而VAE则常常生成模糊图像。为了充分发挥两种模型的优点并减轻它们的弱点，我们采用了贝叶斯非参数方法将GAN和VAE相结合。我们的方法在损失函数中同时使用了Wasserstein和最大均值差异度量，以有效学习潜在空间并生成多样且高质量的样本。

    Generative models have emerged as a promising technique for producing high-quality images that are indistinguishable from real images. Generative adversarial networks (GANs) and variational autoencoders (VAEs) are two of the most prominent and widely studied generative models. GANs have demonstrated excellent performance in generating sharp realistic images and VAEs have shown strong abilities to generate diverse images. However, GANs suffer from ignoring a large portion of the possible output space which does not represent the full diversity of the target distribution, and VAEs tend to produce blurry images. To fully capitalize on the strengths of both models while mitigating their weaknesses, we employ a Bayesian non-parametric (BNP) approach to merge GANs and VAEs. Our procedure incorporates both Wasserstein and maximum mean discrepancy (MMD) measures in the loss function to enable effective learning of the latent space and generate diverse and high-quality samples. By fusing the di
    

