# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LEyes: A Lightweight Framework for Deep Learning-Based Eye Tracking using Synthetic Eye Images.](http://arxiv.org/abs/2309.06129) | 本研究提出了一种名为LEyes的轻量级深度学习眼动跟踪框架，利用合成眼部图像进行训练，解决了由于训练数据集不足和眼部图像变异导致的模型泛化问题。实验结果表明，LEyes训练的模型在瞳孔和CR定位方面优于其他算法。 |

# 详细

[^1]: LEyes：一种轻量级深度学习眼动跟踪框架，使用合成眼部图像

    LEyes: A Lightweight Framework for Deep Learning-Based Eye Tracking using Synthetic Eye Images. (arXiv:2309.06129v1 [cs.CV])

    [http://arxiv.org/abs/2309.06129](http://arxiv.org/abs/2309.06129)

    本研究提出了一种名为LEyes的轻量级深度学习眼动跟踪框架，利用合成眼部图像进行训练，解决了由于训练数据集不足和眼部图像变异导致的模型泛化问题。实验结果表明，LEyes训练的模型在瞳孔和CR定位方面优于其他算法。

    

    深度学习已经加强了凝视估计技术，但实际部署受到不足的训练数据集的限制。眼部图像的硬件引起的变异以及记录的参与者之间固有的生物差异会导致特征和像素级别的差异，阻碍了在特定数据集上训练的模型的泛化能力。虚拟数据集可以是一个解决方案，但创建虚拟数据集既需要时间又需要资源。为了解决这个问题，我们提出了一个名为Light Eyes or "LEyes"的框架，与传统的逼真方法不同，LEyes仅模拟视频眼动跟踪所需的关键图像特征。LEyes便于在多样化的凝视估计任务上训练神经网络。我们证明，使用LEyes训练的模型在眼睛瞳孔和CR定位方面优于其他最先进的算法。

    Deep learning has bolstered gaze estimation techniques, but real-world deployment has been impeded by inadequate training datasets. This problem is exacerbated by both hardware-induced variations in eye images and inherent biological differences across the recorded participants, leading to both feature and pixel-level variance that hinders the generalizability of models trained on specific datasets. While synthetic datasets can be a solution, their creation is both time and resource-intensive. To address this problem, we present a framework called Light Eyes or "LEyes" which, unlike conventional photorealistic methods, only models key image features required for video-based eye tracking using simple light distributions. LEyes facilitates easy configuration for training neural networks across diverse gaze-estimation tasks. We demonstrate that models trained using LEyes outperform other state-of-the-art algorithms in terms of pupil and CR localization across well-known datasets. In addit
    

