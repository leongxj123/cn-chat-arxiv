# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Practical and Asymptotically Exact Conditional Sampling in Diffusion Models.](http://arxiv.org/abs/2306.17775) | 本论文提出了一种名为TDS的扭转式扩散采样器，它是一种针对扩散模型的顺序蒙特卡洛算法。该方法通过使用扭转技术结合启发式近似，能够在不需要特定训练的情况下在广泛的条件分布上提供精确的样本。 |

# 详细

[^1]: 扩散模型中的实用和渐进精确条件采样

    Practical and Asymptotically Exact Conditional Sampling in Diffusion Models. (arXiv:2306.17775v1 [stat.ML])

    [http://arxiv.org/abs/2306.17775](http://arxiv.org/abs/2306.17775)

    本论文提出了一种名为TDS的扭转式扩散采样器，它是一种针对扩散模型的顺序蒙特卡洛算法。该方法通过使用扭转技术结合启发式近似，能够在不需要特定训练的情况下在广泛的条件分布上提供精确的样本。

    

    扩散模型在分子设计和文本到图像生成等条件生成任务中取得了成功。然而，这些成就主要依赖于任务特定的条件训练或容易出错的启发式近似。理想情况下，条件生成方法应该能够在不需要特定训练的情况下为广泛的条件分布提供精确的样本。为此，我们引入了扭转式扩散采样器(TDS)。TDS是一种针对扩散模型的顺序蒙特卡洛(SMC)算法。其主要思想是使用扭转，一种具有良好计算效率的SMC技术，来结合启发式近似而不影响渐进精确性。我们首先在模拟实验和MNIST图像修复以及类条件生成任务中发现，TDS提供了计算统计权衡，使用更多粒子得到更准确的近似结果，但同时需要更多计算资源。

    Diffusion models have been successful on a range of conditional generation tasks including molecular design and text-to-image generation. However, these achievements have primarily depended on task-specific conditional training or error-prone heuristic approximations. Ideally, a conditional generation method should provide exact samples for a broad range of conditional distributions without requiring task-specific training. To this end, we introduce the Twisted Diffusion Sampler, or TDS. TDS is a sequential Monte Carlo (SMC) algorithm that targets the conditional distributions of diffusion models. The main idea is to use twisting, an SMC technique that enjoys good computational efficiency, to incorporate heuristic approximations without compromising asymptotic exactness. We first find in simulation and on MNIST image inpainting and class-conditional generation tasks that TDS provides a computational statistical trade-off, yielding more accurate approximations with many particles but wi
    

