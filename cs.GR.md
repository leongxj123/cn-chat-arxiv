# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Controllable Motion Diffusion Model.](http://arxiv.org/abs/2306.00416) | 该论文提出了可控运动扩散模型（COMODO）框架，通过自回归运动扩散模型（A-MDM）生成高保真度、长时间内的运动序列，以实现在响应于时变控制信号的情况下进行实时运动合成。 |

# 详细

[^1]: 可控运动扩散模型

    Controllable Motion Diffusion Model. (arXiv:2306.00416v1 [cs.CV])

    [http://arxiv.org/abs/2306.00416](http://arxiv.org/abs/2306.00416)

    该论文提出了可控运动扩散模型（COMODO）框架，通过自回归运动扩散模型（A-MDM）生成高保真度、长时间内的运动序列，以实现在响应于时变控制信号的情况下进行实时运动合成。

    

    在计算机动画中，为虚拟角色生成逼真且可控的运动是一项具有挑战性的任务。最近的研究从图像生成的扩散模型的成功中汲取灵感，展示了解决这个问题的潜力。然而，这些研究大多限于离线应用，目标是生成同时生成所有步骤的序列级生成。为了能够在响应于时变控制信号的情况下使用扩散模型实现实时运动合成，我们提出了可控运动扩散模型（COMODO）框架。我们的框架以自回归运动扩散模型（A-MDM）为基础，逐步生成运动序列。通过简单地使用标准DDPM算法而无需任何额外复杂性，我们的框架能够产生在不同类型的运动控制下长时间内的高保真度运动序列。

    Generating realistic and controllable motions for virtual characters is a challenging task in computer animation, and its implications extend to games, simulations, and virtual reality. Recent studies have drawn inspiration from the success of diffusion models in image generation, demonstrating the potential for addressing this task. However, the majority of these studies have been limited to offline applications that target at sequence-level generation that generates all steps simultaneously. To enable real-time motion synthesis with diffusion models in response to time-varying control signals, we propose the framework of the Controllable Motion Diffusion Model (COMODO). Our framework begins with an auto-regressive motion diffusion model (A-MDM), which generates motion sequences step by step. In this way, simply using the standard DDPM algorithm without any additional complexity, our framework is able to generate high-fidelity motion sequences over extended periods with different type
    

