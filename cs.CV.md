# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-Scale Texture Loss for CT denoising with GANs](https://arxiv.org/abs/2403.16640) | 该研究提出了一种利用灰度共生矩阵的多尺度纹理损失函数，以帮助生成对抗网络更好地捕捉复杂的图像关系。 |
| [^2] | [Beyond Specialization: Assessing the Capabilities of MLLMs in Age and Gender Estimation](https://arxiv.org/abs/2403.02302) | 本研究评估了多模态大型语言模型（MLLMs）在年龄和性别估计中的能力，对不同模型进行了比较，揭示了它们在特定任务上的优势和劣势。 |
| [^3] | [Transformer-based model for monocular visual odometry: a video understanding approach.](http://arxiv.org/abs/2305.06121) | 本文提出了一种基于Transformer模型的TSformer-VO方法，将单目视觉里程计作为一项视频理解任务并通过时空自注意机制从视频片段中提取特征，以实现端到端的运动估计，达到了最新成果。 |

# 详细

[^1]: 使用GAN进行CT去噪的多尺度纹理损失

    Multi-Scale Texture Loss for CT denoising with GANs

    [https://arxiv.org/abs/2403.16640](https://arxiv.org/abs/2403.16640)

    该研究提出了一种利用灰度共生矩阵的多尺度纹理损失函数，以帮助生成对抗网络更好地捕捉复杂的图像关系。

    

    生成对抗网络（GANs）已被证明在医学影像中的去噪应用中是一个强大的框架。然而，基于GAN的去噪算法仍然存在捕捉图像内复杂关系的局限性。为了在训练过程中掌握高度复杂和非线性的纹理关系，本文提出了一种利用灰度共生矩阵（GLCM）固有的多尺度性质的损失函数。尽管深度学习的最新进展在分类和检测任务中表现出优越性能，我们假设将其信息内容整合到GANs的训练中会是有价值的。因此，我们提出了适用于基于梯度优化的GLCM的可微分实现。

    arXiv:2403.16640v1 Announce Type: cross  Abstract: Generative Adversarial Networks (GANs) have proved as a powerful framework for denoising applications in medical imaging. However, GAN-based denoising algorithms still suffer from limitations in capturing complex relationships within the images. In this regard, the loss function plays a crucial role in guiding the image generation process, encompassing how much a synthetic image differs from a real image. To grasp highly complex and non-linear textural relationships in the training process, this work presents a loss function that leverages the intrinsic multi-scale nature of the Gray-Level-Co-occurrence Matrix (GLCM). Although the recent advances in deep learning have demonstrated superior performance in classification and detection tasks, we hypothesize that its information content can be valuable when integrated into GANs' training. To this end, we propose a differentiable implementation of the GLCM suited for gradient-based optimiza
    
[^2]: 超越专业化：评估MLLMs在年龄和性别估计中的能力

    Beyond Specialization: Assessing the Capabilities of MLLMs in Age and Gender Estimation

    [https://arxiv.org/abs/2403.02302](https://arxiv.org/abs/2403.02302)

    本研究评估了多模态大型语言模型（MLLMs）在年龄和性别估计中的能力，对不同模型进行了比较，揭示了它们在特定任务上的优势和劣势。

    

    最近，多模态大型语言模型（MLLMs）变得异常流行。像ChatGPT-4V和Gemini这样功能强大的商用模型，以及像LLaVA这样的开源模型，本质上都是通用模型，应用于解决各种各样的任务，包括计算机视觉中的任务。这些神经网络具有如此强大的通用知识和推理能力，以至于它们已被证明能够处理甚至未经专门训练的任务。我们将迄今为止最强大的MLLMs的能力进行了比较：ShareGPT4V、ChatGPT、LLaVA-Next 进行了专门任务的年龄和性别估计，与我们的最新专业化模型MiVOLO进行了比较。我们还更新了MiVOLO，并在本文中提供了详细信息和新的指标。这种比较产生了一些有趣的结果和关于参与模型的优点和缺点的见解。此外，我们尝试了各种微调方法

    arXiv:2403.02302v1 Announce Type: cross  Abstract: Multimodal Large Language Models (MLLMs) have recently gained immense popularity. Powerful commercial models like ChatGPT-4V and Gemini, as well as open-source ones such as LLaVA, are essentially general-purpose models and are applied to solve a wide variety of tasks, including those in computer vision. These neural networks possess such strong general knowledge and reasoning abilities that they have proven capable of working even on tasks for which they were not specifically trained. We compared the capabilities of the most powerful MLLMs to date: ShareGPT4V, ChatGPT, LLaVA-Next in a specialized task of age and gender estimation with our state-of-the-art specialized model, MiVOLO. We also updated MiVOLO and provide details and new metrics in this article. This comparison has yielded some interesting results and insights about the strengths and weaknesses of the participating models. Furthermore, we attempted various ways to fine-tune 
    
[^3]: 基于Transformer模型的单目视觉里程计：一种视频理解方法

    Transformer-based model for monocular visual odometry: a video understanding approach. (arXiv:2305.06121v1 [cs.CV])

    [http://arxiv.org/abs/2305.06121](http://arxiv.org/abs/2305.06121)

    本文提出了一种基于Transformer模型的TSformer-VO方法，将单目视觉里程计作为一项视频理解任务并通过时空自注意机制从视频片段中提取特征，以实现端到端的运动估计，达到了最新成果。

    

    在移动机器人和自主车辆中，给定单个摄像机图像估计摄像机姿势是一项传统任务。这个问题称为单目视觉里程计，通常依赖于需要针对特定场景进行工程化的几何方法。经过适当训练和足够的数据可用性，深度学习方法已被证明是具有普适性的。Transformer架构已统治了自然语言处理和计算机视觉任务的最前沿，例如图像和视频理解。本文将单目视觉里程计作为一项视频理解任务进行处理，以估计6-DoF摄像机的姿势，提出了基于时空自注意机制的TSformer-VO模型，以端到端的方式从视频片段中提取特征并估计运动，与几何和深度学习方法相比，我们的方法在KITTI数据集上取得了有竞争力的最新成果。

    Estimating the camera pose given images of a single camera is a traditional task in mobile robots and autonomous vehicles. This problem is called monocular visual odometry and it often relies on geometric approaches that require engineering effort for a specific scenario. Deep learning methods have shown to be generalizable after proper training and a considerable amount of available data. Transformer-based architectures have dominated the state-of-the-art in natural language processing and computer vision tasks, such as image and video understanding. In this work, we deal with the monocular visual odometry as a video understanding task to estimate the 6-DoF camera's pose. We contribute by presenting the TSformer-VO model based on spatio-temporal self-attention mechanisms to extract features from clips and estimate the motions in an end-to-end manner. Our approach achieved competitive state-of-the-art performance compared with geometry-based and deep learning-based methods on the KITTI
    

