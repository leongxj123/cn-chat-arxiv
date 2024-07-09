# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting](https://arxiv.org/abs/2403.08551) | 通过2D高斯喷涂实现图像表示和压缩，在GPU内存占用降低的情况下，提供了更快的渲染速度，并在表示性能上与INR相匹敌。 |
| [^2] | [TbExplain: A Text-based Explanation Method for Scene Classification Models with the Statistical Prediction Correction.](http://arxiv.org/abs/2307.10003) | 本文提出了一种名为TbExplain的框架，它利用XAI技术和预训练的对象检测器，通过文本形式解释场景分类模型，并引入了一种新的方法来纠正预测和进行文本解释。 |
| [^3] | [Towards Multimodal Prediction of Spontaneous Humour: A Novel Dataset and First Results.](http://arxiv.org/abs/2209.14272) | 本研究提出了Passau-SFCH数据集，包含了11小时的录音，用于自发幽默的预测。通过多模态的分析和特征融合，实现了对幽默以及幽默情感的自动识别。 |

# 详细

[^1]: 高斯图像：通过2D高斯喷涂进行1000帧每秒的图像表示和压缩

    GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting

    [https://arxiv.org/abs/2403.08551](https://arxiv.org/abs/2403.08551)

    通过2D高斯喷涂实现图像表示和压缩，在GPU内存占用降低的情况下，提供了更快的渲染速度，并在表示性能上与INR相匹敌。

    

    最近，隐式神经表示（INR）在图像表示和压缩方面取得了巨大成功，提供了高视觉质量和快速渲染速度，每秒10-1000帧，假设有足够的GPU资源可用。然而，这种要求常常阻碍了它们在内存有限的低端设备上的使用。为此，我们提出了一种通过2D高斯喷涂进行图像表示和压缩的开创性范式，名为GaussianImage。我们首先引入2D高斯来表示图像，其中每个高斯具有8个参数，包括位置、协方差和颜色。随后，我们揭示了一种基于累积求和的新颖渲染算法。值得注意的是，我们的方法使用GPU内存至少降低3倍，拟合时间快5倍，不仅在表示性能上与INR（例如WIRE，I-NGP）不相上下，而且无论参数大小如何都能提供1500-2000帧每秒的更快渲染速度。

    arXiv:2403.08551v1 Announce Type: cross  Abstract: Implicit neural representations (INRs) recently achieved great success in image representation and compression, offering high visual quality and fast rendering speeds with 10-1000 FPS, assuming sufficient GPU resources are available. However, this requirement often hinders their use on low-end devices with limited memory. In response, we propose a groundbreaking paradigm of image representation and compression by 2D Gaussian Splatting, named GaussianImage. We first introduce 2D Gaussian to represent the image, where each Gaussian has 8 parameters including position, covariance and color. Subsequently, we unveil a novel rendering algorithm based on accumulated summation. Remarkably, our method with a minimum of 3$\times$ lower GPU memory usage and 5$\times$ faster fitting time not only rivals INRs (e.g., WIRE, I-NGP) in representation performance, but also delivers a faster rendering speed of 1500-2000 FPS regardless of parameter size. 
    
[^2]: TbExplain: 一种场景分类模型的基于文本的解释方法与统计预测校正

    TbExplain: A Text-based Explanation Method for Scene Classification Models with the Statistical Prediction Correction. (arXiv:2307.10003v1 [cs.CV])

    [http://arxiv.org/abs/2307.10003](http://arxiv.org/abs/2307.10003)

    本文提出了一种名为TbExplain的框架，它利用XAI技术和预训练的对象检测器，通过文本形式解释场景分类模型，并引入了一种新的方法来纠正预测和进行文本解释。

    

    可解释性人工智能(XAI)的领域旨在提高黑盒机器学习模型的可解释性。建立基于输入特征重要性值的热图是解释这些模型产生预测的基本方法之一。热图在人类中几乎可以理解，但并非没有缺陷。例如，非专业用户可能不完全理解热图的逻辑（即使用不同强度或颜色突出显示与模型预测相关的像素的逻辑）。此外，与模型预测相关的输入图像的对象和区域通常无法完全通过热图区分。本文提出了一种称为TbExplain的框架，采用XAI技术和预训练的对象检测器，以文本形式解释场景分类模型。此外，TbExplain还采用了一种新的方法来纠正预测和进行文本解释。

    The field of Explainable Artificial Intelligence (XAI) aims to improve the interpretability of black-box machine learning models. Building a heatmap based on the importance value of input features is a popular method for explaining the underlying functions of such models in producing their predictions. Heatmaps are almost understandable to humans, yet they are not without flaws. Non-expert users, for example, may not fully understand the logic of heatmaps (the logic in which relevant pixels to the model's prediction are highlighted with different intensities or colors). Additionally, objects and regions of the input image that are relevant to the model prediction are frequently not entirely differentiated by heatmaps. In this paper, we propose a framework called TbExplain that employs XAI techniques and a pre-trained object detector to present text-based explanations of scene classification models. Moreover, TbExplain incorporates a novel method to correct predictions and textually exp
    
[^3]: 迈向多模态预测自发幽默：一份新颖的数据集和初步结果

    Towards Multimodal Prediction of Spontaneous Humour: A Novel Dataset and First Results. (arXiv:2209.14272v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.14272](http://arxiv.org/abs/2209.14272)

    本研究提出了Passau-SFCH数据集，包含了11小时的录音，用于自发幽默的预测。通过多模态的分析和特征融合，实现了对幽默以及幽默情感的自动识别。

    

    幽默是人类情感和认知的重要元素。其自动理解可以促进更自然的人机交互和人工智能的人性化。目前的幽默检测方法仅基于策划数据，不能满足“现实世界”应用的需求。我们通过引入新颖的Passau-Spontaneous Football Coach Humour（Passau-SFCH）数据集，该数据集包含约11小时的录音，解决了这一缺陷。Passau-SFCH数据集的注释根据Martin的幽默风格问卷提出的幽默存在及其维度（情感和方向）。我们进行了一系列实验，采用预训练的Transformer、卷积神经网络和专家设计的特征。分析了自发幽默识别的每种模态（文本、音频、视频）的性能，并研究了它们之间的互补性。我们的研究结果表明，对于幽默及其情感的自动分析，多模态联合使用效果更好。

    Humour is a substantial element of human affect and cognition. Its automatic understanding can facilitate a more naturalistic human-device interaction and the humanisation of artificial intelligence. Current methods of humour detection are solely based on staged data making them inadequate for 'real-world' applications. We address this deficiency by introducing the novel Passau-Spontaneous Football Coach Humour (Passau-SFCH) dataset, comprising of about 11 hours of recordings. The Passau-SFCH dataset is annotated for the presence of humour and its dimensions (sentiment and direction) as proposed in Martin's Humor Style Questionnaire. We conduct a series of experiments, employing pretrained Transformers, convolutional neural networks, and expert-designed features. The performance of each modality (text, audio, video) for spontaneous humour recognition is analysed and their complementarity is investigated. Our findings suggest that for the automatic analysis of humour and its sentiment, 
    

