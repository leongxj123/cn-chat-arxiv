# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fast, Scale-Adaptive, and Uncertainty-Aware Downscaling of Earth System Model Fields with Generative Foundation Models](https://arxiv.org/abs/2403.02774) | 通过学习一致性模型，在不需要重新训练的情况下高效、准确地降尺度任意地球系统模型模拟，并产生概率性降尺度场。 |
| [^2] | [Parameter-Efficient Tuning of Large Convolutional Models](https://arxiv.org/abs/2403.00269) | 通过引入滤波器子空间和滤波器原子的概念，本研究提出了一种在微调大型卷积模型时仅调整少量参数来提取任务特定表示的方法。 |
| [^3] | [p$^3$VAE: a physics-integrated generative model. Application to the semantic segmentation of optical remote sensing images.](http://arxiv.org/abs/2210.10418) | 本文介绍了p$^3$VAE生成模型，它将一个完美的物理模型集成到模型中，并应用于高分辨率高光谱遥感图像的语义分割。模型具有更好的外推能力和可解释性，同时具有高度解缕能力。 |

# 详细

[^1]: 快速、自适应尺度和具有不确定性意识的地球系统模型场降尺度与生成基础模型

    Fast, Scale-Adaptive, and Uncertainty-Aware Downscaling of Earth System Model Fields with Generative Foundation Models

    [https://arxiv.org/abs/2403.02774](https://arxiv.org/abs/2403.02774)

    通过学习一致性模型，在不需要重新训练的情况下高效、准确地降尺度任意地球系统模型模拟，并产生概率性降尺度场。

    

    精确和高分辨率的地球系统模型(ESM)模拟对于评估人为气候变化对生态和社会经济影响至关重要，但计算成本过高。最近的机器学习方法在ESM模拟的降尺度中表现出色，优于最先进的统计方法。然而，现有方法对每个ESM都需要计算昂贵的重新训练，并且在训练期间未见过的气候预测效果差。我们通过学习一个一致性模型(CM)，以零样本方式高效准确地降尺度任意ESM模拟来解决这些缺点。我们的基础模型方法以只受观测参考数据限制的分辨率产生概率性降尺度场。我们展示了CM在维持高可控性的同时以较低的计算成本优于最先进的扩散模型。

    arXiv:2403.02774v1 Announce Type: cross  Abstract: Accurate and high-resolution Earth system model (ESM) simulations are essential to assess the ecological and socio-economic impacts of anthropogenic climate change, but are computationally too expensive. Recent machine learning approaches have shown promising results in downscaling ESM simulations, outperforming state-of-the-art statistical approaches. However, existing methods require computationally costly retraining for each ESM and extrapolate poorly to climates unseen during training. We address these shortcomings by learning a consistency model (CM) that efficiently and accurately downscales arbitrary ESM simulations without retraining in a zero-shot manner. Our foundation model approach yields probabilistic downscaled fields at resolution only limited by the observational reference data. We show that the CM outperforms state-of-the-art diffusion models at a fraction of computational cost while maintaining high controllability on
    
[^2]: 大型卷积模型的参数高效调整

    Parameter-Efficient Tuning of Large Convolutional Models

    [https://arxiv.org/abs/2403.00269](https://arxiv.org/abs/2403.00269)

    通过引入滤波器子空间和滤波器原子的概念，本研究提出了一种在微调大型卷积模型时仅调整少量参数来提取任务特定表示的方法。

    

    为了解决微调大型预训练模型所需的高计算和参数复杂性，研究人员开发了参数高效的方法，仅更新下游任务的部分参数。然而，这些工作通常忽视了卷积核的独特属性，而卷积核仍然是许多大型模型的基本元素，比如Stable Diffusion。在本研究中，我们首先通过在每个网络层内分解卷积核到一小组滤波器子空间元素，即滤波器原子，引入了滤波器子空间。然后，我们通过仅调整滤波器原子（通常为几百个参数）对这些模型进行微调，以提取任务特定的表示。为了潜在地扩展调整的参数空间，我们进一步展示了一种简单的方法，通过递归地将每个筛选原子分解到另一组筛选原子来生成一个过完备的滤波器子空间。

    arXiv:2403.00269v1 Announce Type: cross  Abstract: To address the high computational and parameter complexity associated with fine-tuning large pre-trained models, researchers have developed parameter-efficient methods, where only partial parameters are updated for downstream tasks. However, these works often overlook the distinct properties of convolutional kernels, which still remain essential elements in many large models, such as Stable Diffusion. In this study, we first introduce filter subspace by decomposing convolutional kernels within each network layer over a small set of filter subspace elements, referred to as filter atoms. We then fine-tune these models to extract task-specific representation by only adapting the filter atoms, a few hundred parameters typically. To potentially expand the parameter space for tuning, we further show a simple approach to generate an overcomplete filter subspace by recursively decomposing each filter atom over another set of filter atoms. The 
    
[^3]: p$^3$VAE：一个物理集成的生成模型，应用于光学遥感图像的语义分割

    p$^3$VAE: a physics-integrated generative model. Application to the semantic segmentation of optical remote sensing images. (arXiv:2210.10418v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2210.10418](http://arxiv.org/abs/2210.10418)

    本文介绍了p$^3$VAE生成模型，它将一个完美的物理模型集成到模型中，并应用于高分辨率高光谱遥感图像的语义分割。模型具有更好的外推能力和可解释性，同时具有高度解缕能力。

    

    将机器学习模型与物理模型相结合是学习强大数据表示的最新研究方向。本文介绍了p$^3$VAE，这是一个生成模型，它集成了一个完美的物理模型，部分解释了数据中真实的变化因素。为了充分利用我们的混合设计，我们提出了一种半监督优化过程和一种推断方案，同时伴随着有意义的不确定性估计。我们将p$^3$VAE应用于高分辨率高光谱遥感图像的语义分割。我们在一个模拟数据集上的实验表明，与传统的机器学习模型相比，我们的混合模型具有更好的外推能力和可解释性。特别是，我们展示了p$^3$VAE自然具有高度解缕能力。我们的代码和数据已在https://github.com/Romain3Ch216/p3VAE上公开发布。

    The combination of machine learning models with physical models is a recent research path to learn robust data representations. In this paper, we introduce p$^3$VAE, a generative model that integrates a perfect physical model which partially explains the true underlying factors of variation in the data. To fully leverage our hybrid design, we propose a semi-supervised optimization procedure and an inference scheme that comes along meaningful uncertainty estimates. We apply p$^3$VAE to the semantic segmentation of high-resolution hyperspectral remote sensing images. Our experiments on a simulated data set demonstrated the benefits of our hybrid model against conventional machine learning models in terms of extrapolation capabilities and interpretability. In particular, we show that p$^3$VAE naturally has high disentanglement capabilities. Our code and data have been made publicly available at https://github.com/Romain3Ch216/p3VAE.
    

