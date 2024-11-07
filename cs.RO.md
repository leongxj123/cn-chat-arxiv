# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DexDiffuser: Generating Dexterous Grasps with Diffusion Models](https://arxiv.org/abs/2402.02989) | DexDiffuser是一种使用扩散模型生成灵巧抓取姿势的新方法，通过对物体点云的生成、评估和优化，实现了较高的抓取成功率。 |

# 详细

[^1]: DexDiffuser: 使用扩散模型生成灵巧抓取姿势

    DexDiffuser: Generating Dexterous Grasps with Diffusion Models

    [https://arxiv.org/abs/2402.02989](https://arxiv.org/abs/2402.02989)

    DexDiffuser是一种使用扩散模型生成灵巧抓取姿势的新方法，通过对物体点云的生成、评估和优化，实现了较高的抓取成功率。

    

    我们引入了DexDiffuser，一种新颖的灵巧抓取方法，能够在部分物体点云上生成、评估和优化抓取姿势。DexDiffuser包括条件扩散型抓取采样器DexSampler和灵巧抓取评估器DexEvaluator。DexSampler通过对随机抓取进行迭代去噪，生成与物体点云条件相关的高质量抓取姿势。我们还引入了两种抓取优化策略：基于评估器的扩散(Evaluator-Guided Diffusion，EGD)和基于评估器的采样优化(Evaluator-based Sampling Refinement，ESR)。我们在虚拟环境和真实世界的实验中，使用Allegro Hand进行测试，结果表明DexDiffuser相比最先进的多指抓取生成方法FFHNet，平均抓取成功率提高了21.71-22.20%。

    We introduce DexDiffuser, a novel dexterous grasping method that generates, evaluates, and refines grasps on partial object point clouds. DexDiffuser includes the conditional diffusion-based grasp sampler DexSampler and the dexterous grasp evaluator DexEvaluator. DexSampler generates high-quality grasps conditioned on object point clouds by iterative denoising of randomly sampled grasps. We also introduce two grasp refinement strategies: Evaluator-Guided Diffusion (EGD) and Evaluator-based Sampling Refinement (ESR). Our simulation and real-world experiments on the Allegro Hand consistently demonstrate that DexDiffuser outperforms the state-of-the-art multi-finger grasp generation method FFHNet with an, on average, 21.71--22.20\% higher grasp success rate.
    

