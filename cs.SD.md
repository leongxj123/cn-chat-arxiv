# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Speech Enhancement and Dereverberation with Diffusion-based Generative Models.](http://arxiv.org/abs/2208.05830) | 本文基于扩散生成模型，通过混合噪声语音和高斯噪声进行反向过程，仅使用30步就实现高质量的干净语音估计。调整网络架构，可以显著提高语音增强性能，达到了最新方法的竞争水平。 |

# 详细

[^1]: 基于扩散生成模型的语音增强和去混响

    Speech Enhancement and Dereverberation with Diffusion-based Generative Models. (arXiv:2208.05830v2 [eess.AS] UPDATED)

    [http://arxiv.org/abs/2208.05830](http://arxiv.org/abs/2208.05830)

    本文基于扩散生成模型，通过混合噪声语音和高斯噪声进行反向过程，仅使用30步就实现高质量的干净语音估计。调整网络架构，可以显著提高语音增强性能，达到了最新方法的竞争水平。

    

    在这项工作中，我们基于我们之前的出版物，并使用基于扩散的生成模型进行语音增强。我们详细介绍了基于随机微分方程的扩散过程，并进行了广泛的理论探讨其含义。与通常的条件生成任务相反，我们不是从纯高斯噪声开始反向过程，而是从噪声语音和高斯噪声的混合物开始。这与我们的正向过程相匹配，该过程通过包括漂移项将干净语音转变成噪声语音。我们表明，这个过程使得使用仅30个扩散步骤来生成高质量的干净语音估计成为可能。通过调整网络架构，我们能够显著提高语音增强性能，这表明网络而不是形式主义是我们原始方法的主要限制。在广泛的跨数据集评估中，我们展示了改进后的方法可以与最新的方法竞争。

    In this work, we build upon our previous publication and use diffusion-based generative models for speech enhancement. We present a detailed overview of the diffusion process that is based on a stochastic differential equation and delve into an extensive theoretical examination of its implications. Opposed to usual conditional generation tasks, we do not start the reverse process from pure Gaussian noise but from a mixture of noisy speech and Gaussian noise. This matches our forward process which moves from clean speech to noisy speech by including a drift term. We show that this procedure enables using only 30 diffusion steps to generate high-quality clean speech estimates. By adapting the network architecture, we are able to significantly improve the speech enhancement performance, indicating that the network, rather than the formalism, was the main limitation of our original approach. In an extensive cross-dataset evaluation, we show that the improved method can compete with recent 
    

