# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SportsNGEN: Sustained Generation of Multi-player Sports Gameplay](https://arxiv.org/abs/2403.12977) | SportsNGEN是一种基于Transformer解码器的模型，经过训练能持续生成逼真的多人体育游戏，包括模拟整个网球比赛和为教练和广播员提供洞察力的能力。 |
| [^2] | [Multistep Consistency Models](https://arxiv.org/abs/2403.06807) | 本文提出了多步一致性模型，通过在一致性模型和扩散模型之间插值，实现了采样速度和采样质量的平衡。 |
| [^3] | [Rethinking cluster-conditioned diffusion models](https://arxiv.org/abs/2403.00570) | 通过结合最新的图片聚类和扩散模型技术，本文提出了一种在考虑最佳聚类粒度的情况下实现最先进FID并具有较强训练样本效率的聚类条件扩散模型，并提出了一种新颖方法来减少视觉组搜索空间。 |
| [^4] | [Spiking NeRF: Making Bio-inspired Neural Networks See through the Real World.](http://arxiv.org/abs/2309.10987) | 本文介绍了SpikingNeRF，它通过将辐射光线与脉冲神经网络的时间维度对齐，以节省能量并减少计算量。 |
| [^5] | [S-HR-VQVAE: Sequential Hierarchical Residual Learning Vector Quantized Variational Autoencoder for Video Prediction.](http://arxiv.org/abs/2307.06701) | S-HR-VQVAE是一种序列分层残差学习向量量化变分自编码器，通过结合分层残差向量量化变分自编码器（HR-VQVAE）和时空PixelCNN（ST-PixelCNN）的能力，解决了视频预测中的主要挑战，并在KTH人体动作和Moving-MNIST任务上取得了较好的实验结果。 |

# 详细

[^1]: SportsNGEN: 持续生成多人体育游戏

    SportsNGEN: Sustained Generation of Multi-player Sports Gameplay

    [https://arxiv.org/abs/2403.12977](https://arxiv.org/abs/2403.12977)

    SportsNGEN是一种基于Transformer解码器的模型，经过训练能持续生成逼真的多人体育游戏，包括模拟整个网球比赛和为教练和广播员提供洞察力的能力。

    

    我们提出了一种基于Transformer解码器的模型SportsNGEN，该模型经过训练使用运动员和球追踪序列，能够生成逼真且持续的游戏场景。我们在大量专业网球追踪数据上训练和评估SportsNGEN，并展示通过将生成的模拟与射击分类器和逻辑相结合来开始和结束球赛，系统能够模拟整个网球比赛。此外，SportsNGEN的通用版本可以通过在包含该球员的比赛数据上微调来定制特定球员。我们展示了我们的模型经过良好校准，可以通过评估反事实或假设选项为教练和广播员提供洞察力。最后，我们展示了质量结果表明相同的方法适用于足球。

    arXiv:2403.12977v1 Announce Type: cross  Abstract: We present a transformer decoder based model, SportsNGEN, that is trained on sports player and ball tracking sequences that is capable of generating realistic and sustained gameplay. We train and evaluate SportsNGEN on a large database of professional tennis tracking data and demonstrate that by combining the generated simulations with a shot classifier and logic to start and end rallies, the system is capable of simulating an entire tennis match. In addition, a generic version of SportsNGEN can be customized to a specific player by fine-tuning on match data that includes that player. We show that our model is well calibrated and can be used to derive insights for coaches and broadcasters by evaluating counterfactual or what if options. Finally, we show qualitative results indicating the same approach works for football.
    
[^2]: 多步一致性模型

    Multistep Consistency Models

    [https://arxiv.org/abs/2403.06807](https://arxiv.org/abs/2403.06807)

    本文提出了多步一致性模型，通过在一致性模型和扩散模型之间插值，实现了采样速度和采样质量的平衡。

    

    扩散模型相对容易训练，但生成样本需要许多步骤。一致性模型更难训练，但可以在一个步骤中生成样本。本文提出了多步一致性模型：通过一致性模型和TRACT的统一，可以在一致性模型和扩散模型之间进行插值：在采样速度和采样质量之间取得平衡。具体来说，1步一致性模型是传统的一致性模型，而我们展示了$\infty$步一致性模型是扩散模型。多步一致性模型在实践中表现良好。将样本预算从单步增加到2-8步，我们可以更轻松地训练模型，生成更高质量的样本，同时保留大部分采样速度优势。在Imagenet 64上8步达到1.4的FID，在Imagenet128上8步达到2.1的FID。

    arXiv:2403.06807v1 Announce Type: new  Abstract: Diffusion models are relatively easy to train but require many steps to generate samples. Consistency models are far more difficult to train, but generate samples in a single step.   In this paper we propose Multistep Consistency Models: A unification between Consistency Models (Song et al., 2023) and TRACT (Berthelot et al., 2023) that can interpolate between a consistency model and a diffusion model: a trade-off between sampling speed and sampling quality. Specifically, a 1-step consistency model is a conventional consistency model whereas we show that a $\infty$-step consistency model is a diffusion model.   Multistep Consistency Models work really well in practice. By increasing the sample budget from a single step to 2-8 steps, we can train models more easily that generate higher quality samples, while retaining much of the sampling speed benefits. Notable results are 1.4 FID on Imagenet 64 in 8 step and 2.1 FID on Imagenet128 in 8 
    
[^3]: 重新思考基于聚类条件的扩散模型

    Rethinking cluster-conditioned diffusion models

    [https://arxiv.org/abs/2403.00570](https://arxiv.org/abs/2403.00570)

    通过结合最新的图片聚类和扩散模型技术，本文提出了一种在考虑最佳聚类粒度的情况下实现最先进FID并具有较强训练样本效率的聚类条件扩散模型，并提出了一种新颖方法来减少视觉组搜索空间。

    

    我们针对使用聚类分配的图片级条件扩散模型进行了全面的实验研究。我们阐明了关于图片聚类的个别组件如何影响三个数据集上的图片合成。通过结合图片聚类和扩散模型的最新进展，我们展示了，在考虑到图片合成（视觉组）的最佳簇粒度的情况下，通过聚类条件可以实现最先进的FID（即在CIFAR10和CIFAR100上分别为1.67和2.17），同时实现了较强的训练样本效率。最后，我们提出了一种新颖的方法，通过仅使用基于特征的聚类来推导减少视觉组搜索空间的上限簇边界。与现有方法不同，我们发现聚类与基于聚类的图片生成之间没有显著联系。代码和聚类分配将会发布。

    arXiv:2403.00570v1 Announce Type: cross  Abstract: We present a comprehensive experimental study on image-level conditioning for diffusion models using cluster assignments. We elucidate how individual components regarding image clustering impact image synthesis across three datasets. By combining recent advancements from image clustering and diffusion models, we show that, given the optimal cluster granularity with respect to image synthesis (visual groups), cluster-conditioning can achieve state-of-the-art FID (i.e. 1.67, 2.17 on CIFAR10 and CIFAR100 respectively), while attaining a strong training sample efficiency. Finally, we propose a novel method to derive an upper cluster bound that reduces the search space of the visual groups using solely feature-based clustering. Unlike existing approaches, we find no significant connection between clustering and cluster-conditional image generation. The code and cluster assignments will be released.
    
[^4]: Spiking NeRF：使生物启发的神经网络穿透现实世界

    Spiking NeRF: Making Bio-inspired Neural Networks See through the Real World. (arXiv:2309.10987v1 [cs.NE])

    [http://arxiv.org/abs/2309.10987](http://arxiv.org/abs/2309.10987)

    本文介绍了SpikingNeRF，它通过将辐射光线与脉冲神经网络的时间维度对齐，以节省能量并减少计算量。

    

    脉冲神经网络（SNN）在许多任务中取得了成功，利用其具有潜在生物学可行性的能量效率和潜力。与此同时，神经辐射场（NeRF）以大量能量消耗渲染高质量的3D场景，但很少有研究深入探索以生物启发的方法进行节能解决方案。本文提出了脉冲NeRF（SpikingNeRF），将辐射光线与SNN的时间维度对齐，以自然地适应SNN对辐射场的重建。因此，计算以基于脉冲、无乘法的方式进行，从而减少能量消耗。在SpikingNeRF中，光线上的每个采样点匹配到特定的时间步，并以混合方式表示，其中体素网格也得到维护。基于体素网格，确定采样点是否在训练和推断过程中被屏蔽以进行更好的处理。然而，这个操作也会产生不可逆性。

    Spiking neuron networks (SNNs) have been thriving on numerous tasks to leverage their promising energy efficiency and exploit their potentialities as biologically plausible intelligence. Meanwhile, the Neural Radiance Fields (NeRF) render high-quality 3D scenes with massive energy consumption, and few works delve into the energy-saving solution with a bio-inspired approach. In this paper, we propose spiking NeRF (SpikingNeRF), which aligns the radiance ray with the temporal dimension of SNN, to naturally accommodate the SNN to the reconstruction of Radiance Fields. Thus, the computation turns into a spike-based, multiplication-free manner, reducing the energy consumption. In SpikingNeRF, each sampled point on the ray is matched onto a particular time step, and represented in a hybrid manner where the voxel grids are maintained as well. Based on the voxel grids, sampled points are determined whether to be masked for better training and inference. However, this operation also incurs irre
    
[^5]: S-HR-VQVAE: 序列分层残差学习向量量化变分自编码器用于视频预测

    S-HR-VQVAE: Sequential Hierarchical Residual Learning Vector Quantized Variational Autoencoder for Video Prediction. (arXiv:2307.06701v1 [cs.CV])

    [http://arxiv.org/abs/2307.06701](http://arxiv.org/abs/2307.06701)

    S-HR-VQVAE是一种序列分层残差学习向量量化变分自编码器，通过结合分层残差向量量化变分自编码器（HR-VQVAE）和时空PixelCNN（ST-PixelCNN）的能力，解决了视频预测中的主要挑战，并在KTH人体动作和Moving-MNIST任务上取得了较好的实验结果。

    

    我们提出了一种新的模型，将我们最近提出的分层残差向量量化变分自编码器（HR-VQVAE）与一种新颖的时空PixelCNN（ST-PixelCNN）相结合，用来解决视频预测任务。我们将这种方法称为序列分层残差学习向量量化变分自编码器（S-HR-VQVAE）。通过利用HR-VQVAE在对静止图像进行建模时的内在能力和紧凑表示，以及ST-PixelCNN处理时空信息的能力， S-HR-VQVAE能够更好地应对视频预测中的主要挑战，包括学习时空信息、处理高维数据、消除模糊预测和隐式建模物理特性。对KTH人体动作和Moving-MNIST任务的大量实验证明，我们的模型在定量和定性评估方面与顶级视频预测技术相比具有优势。

    We address the video prediction task by putting forth a novel model that combines (i) our recently proposed hierarchical residual vector quantized variational autoencoder (HR-VQVAE), and (ii) a novel spatiotemporal PixelCNN (ST-PixelCNN). We refer to this approach as a sequential hierarchical residual learning vector quantized variational autoencoder (S-HR-VQVAE). By leveraging the intrinsic capabilities of HR-VQVAE at modeling still images with a parsimonious representation, combined with the ST-PixelCNN's ability at handling spatiotemporal information, S-HR-VQVAE can better deal with chief challenges in video prediction. These include learning spatiotemporal information, handling high dimensional data, combating blurry prediction, and implicit modeling of physical characteristics. Extensive experimental results on the KTH Human Action and Moving-MNIST tasks demonstrate that our model compares favorably against top video prediction techniques both in quantitative and qualitative evalu
    

