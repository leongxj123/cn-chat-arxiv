# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [EMDM: Efficient Motion Diffusion Model for Fast and High-Quality Motion Generation](https://arxiv.org/abs/2312.02256) | 提出了高效动态扩散模型（EMDM），能够在更少的采样步骤中实现快速且高质量的动作生成 |

# 详细

[^1]: 高效动态扩散模型（EMDM）用于快速且高质量的动作生成

    EMDM: Efficient Motion Diffusion Model for Fast and High-Quality Motion Generation

    [https://arxiv.org/abs/2312.02256](https://arxiv.org/abs/2312.02256)

    提出了高效动态扩散模型（EMDM），能够在更少的采样步骤中实现快速且高质量的动作生成

    

    我们引入了高效的动态扩散模型（EMDM），用于快速且高质量的人类动作生成。当前最先进的生成式扩散模型取得了令人印象深刻的结果，但往往在追求快速生成的同时牺牲了质量。为了解决这些问题，我们提出了EMDM，它通过在扩散模型中的多次采样步骤中捕捉复杂分布，实现了更少的采样步骤和生成过程的显着加速。

    arXiv:2312.02256v2 Announce Type: replace-cross  Abstract: We introduce Efficient Motion Diffusion Model (EMDM) for fast and high-quality human motion generation. Current state-of-the-art generative diffusion models have produced impressive results but struggle to achieve fast generation without sacrificing quality. On the one hand, previous works, like motion latent diffusion, conduct diffusion within a latent space for efficiency, but learning such a latent space can be a non-trivial effort. On the other hand, accelerating generation by naively increasing the sampling step size, e.g., DDIM, often leads to quality degradation as it fails to approximate the complex denoising distribution. To address these issues, we propose EMDM, which captures the complex distribution during multiple sampling steps in the diffusion model, allowing for much fewer sampling steps and significant acceleration in generation. This is achieved by a conditional denoising diffusion GAN to capture multimodal da
    

