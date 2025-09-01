# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TorchCP: A Library for Conformal Prediction based on PyTorch](https://arxiv.org/abs/2402.12683) | TorchCP是一个基于PyTorch的Python工具包，为深度学习模型上的合拟常规预测研究提供了实现后验和训练方法的多种工具，包括分类和回归任务。En_Tdlr: TorchCP is a Python toolbox built on PyTorch for conformal prediction research on deep learning models, providing various implementations for posthoc and training methods for classification and regression tasks, including multi-dimension output. |
| [^2] | [Large Intestine 3D Shape Refinement Using Point Diffusion Models for Digital Phantom Generation.](http://arxiv.org/abs/2309.08289) | 本研究利用几何深度学习和去噪扩散概率模型优化大肠的分割结果，并结合先进的表面重构模型，实现对大肠3D形状的精化恢复。 |

# 详细

[^1]: TorchCP：基于PyTorch的一种适用于合拟常规预测的库

    TorchCP: A Library for Conformal Prediction based on PyTorch

    [https://arxiv.org/abs/2402.12683](https://arxiv.org/abs/2402.12683)

    TorchCP是一个基于PyTorch的Python工具包，为深度学习模型上的合拟常规预测研究提供了实现后验和训练方法的多种工具，包括分类和回归任务。En_Tdlr: TorchCP is a Python toolbox built on PyTorch for conformal prediction research on deep learning models, providing various implementations for posthoc and training methods for classification and regression tasks, including multi-dimension output.

    

    TorchCP是一个用于深度学习模型上的合拟常规预测研究的Python工具包。它包含了用于后验和训练方法的各种实现，用于分类和回归任务（包括多维输出）。TorchCP建立在PyTorch之上，并利用矩阵计算的优势，提供简洁高效的推理实现。该代码采用LGPL许可证，并在$\href{https://github.com/ml-stat-Sustech/TorchCP}{\text{this https URL}}$开源。

    arXiv:2402.12683v1 Announce Type: new  Abstract: TorchCP is a Python toolbox for conformal prediction research on deep learning models. It contains various implementations for posthoc and training methods for classification and regression tasks (including multi-dimension output). TorchCP is built on PyTorch (Paszke et al., 2019) and leverages the advantages of matrix computation to provide concise and efficient inference implementations. The code is licensed under the LGPL license and is open-sourced at $\href{https://github.com/ml-stat-Sustech/TorchCP}{\text{this https URL}}$.
    
[^2]: 利用点扩散模型对大肠的3D形状进行精化以生成数字幻影

    Large Intestine 3D Shape Refinement Using Point Diffusion Models for Digital Phantom Generation. (arXiv:2309.08289v1 [cs.CV])

    [http://arxiv.org/abs/2309.08289](http://arxiv.org/abs/2309.08289)

    本研究利用几何深度学习和去噪扩散概率模型优化大肠的分割结果，并结合先进的表面重构模型，实现对大肠3D形状的精化恢复。

    

    准确建模人体器官在构建虚拟成像试验的计算仿真中起着至关重要的作用。然而，从计算机断层扫描中生成解剖学上可信的器官表面重建仍然对人体结构中的许多器官来说是个挑战。在处理大肠时，这个挑战尤为明显。在这项研究中，我们利用几何深度学习和去噪扩散概率模型的最新进展来优化大肠分割结果。首先，我们将器官表示为从3D分割掩模表面采样得到的点云。随后，我们使用分层变分自编码器获得器官形状的全局和局部潜在表示。我们在分层潜在空间中训练两个条件去噪扩散模型来进行形状精化。为了进一步提高我们的方法，我们还结合了一种先进的表面重构模型，从而实现形状的更好恢复。

    Accurate 3D modeling of human organs plays a crucial role in building computational phantoms for virtual imaging trials. However, generating anatomically plausible reconstructions of organ surfaces from computed tomography scans remains challenging for many structures in the human body. This challenge is particularly evident when dealing with the large intestine. In this study, we leverage recent advancements in geometric deep learning and denoising diffusion probabilistic models to refine the segmentation results of the large intestine. We begin by representing the organ as point clouds sampled from the surface of the 3D segmentation mask. Subsequently, we employ a hierarchical variational autoencoder to obtain global and local latent representations of the organ's shape. We train two conditional denoising diffusion models in the hierarchical latent space to perform shape refinement. To further enhance our method, we incorporate a state-of-the-art surface reconstruction model, allowin
    

