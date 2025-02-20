# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Transfer Attack to Image Watermarks](https://arxiv.org/abs/2403.15365) | 水印领域的研究表明，即使在攻击者无法访问水印模型或检测API的情况下，水印基础的AI生成图像检测器也无法抵抗对抗攻击。 |
| [^2] | [CARSO: Counter-Adversarial Recall of Synthetic Observations.](http://arxiv.org/abs/2306.06081) | 本文提出了一种新的图像分类的对抗性防御机制CARSO，该方法可以比最先进的对抗性训练更好地保护分类器，通过利用生成模型进行对抗净化来进行最终分类，并成功地保护自己免受未预见的威胁和最终攻击。 |

# 详细

[^1]: 一种针对图像水印的转移攻击

    A Transfer Attack to Image Watermarks

    [https://arxiv.org/abs/2403.15365](https://arxiv.org/abs/2403.15365)

    水印领域的研究表明，即使在攻击者无法访问水印模型或检测API的情况下，水印基础的AI生成图像检测器也无法抵抗对抗攻击。

    

    水印已被广泛应用于工业领域，用于检测由人工智能生成的图像。文献中对这种基于水印的检测器在白盒和黑盒环境下对抗攻击的稳健性有很好的理解。然而，在无盒环境下的稳健性却知之甚少。具体来说，多项研究声称图像水印在这种环境下是稳健的。在这项工作中，我们提出了一种新的转移对抗攻击来针对无盒环境下的图像水印。我们的转移攻击向带水印的图像添加微扰，以躲避被攻击者训练的多个替代水印模型，并且经过扰动的带水印图像也能躲避目标水印模型。我们的主要贡献是理论上和经验上展示了，基于水印的人工智能生成图像检测器即使攻击者没有访问水印模型或检测API，也不具有对抗攻击的稳健性。

    arXiv:2403.15365v1 Announce Type: cross  Abstract: Watermark has been widely deployed by industry to detect AI-generated images. The robustness of such watermark-based detector against evasion attacks in the white-box and black-box settings is well understood in the literature. However, the robustness in the no-box setting is much less understood. In particular, multiple studies claimed that image watermark is robust in such setting. In this work, we propose a new transfer evasion attack to image watermark in the no-box setting. Our transfer attack adds a perturbation to a watermarked image to evade multiple surrogate watermarking models trained by the attacker itself, and the perturbed watermarked image also evades the target watermarking model. Our major contribution is to show that, both theoretically and empirically, watermark-based AI-generated image detector is not robust to evasion attacks even if the attacker does not have access to the watermarking model nor the detection API.
    
[^2]: CARSO: 对抗性合成观测的反对抗性召回

    CARSO: Counter-Adversarial Recall of Synthetic Observations. (arXiv:2306.06081v1 [cs.CV])

    [http://arxiv.org/abs/2306.06081](http://arxiv.org/abs/2306.06081)

    本文提出了一种新的图像分类的对抗性防御机制CARSO，该方法可以比最先进的对抗性训练更好地保护分类器，通过利用生成模型进行对抗净化来进行最终分类，并成功地保护自己免受未预见的威胁和最终攻击。

    

    本文提出了一种新的对抗性防御机制CARSO，用于图像分类，灵感来自认知神经科学的线索。该方法与对抗训练具有协同互补性，并依赖于被攻击分类器的内部表示的知识。通过利用生成模型进行对抗净化，该方法采样输入的重构来进行最终分类。在各种图像数据集和分类器体系结构上进行的实验评估表明，CARSO能够比最先进的对抗性训练更好地保护分类器——同时具有可接受的清洁准确度损失。此外，防御体系结构成功地保护自己免受未预见的威胁和最终攻击。代码和预训练模型可在https://github.com/获得。

    In this paper, we propose a novel adversarial defence mechanism for image classification -- CARSO -- inspired by cues from cognitive neuroscience. The method is synergistically complementary to adversarial training and relies on knowledge of the internal representation of the attacked classifier. Exploiting a generative model for adversarial purification, conditioned on such representation, it samples reconstructions of inputs to be finally classified. Experimental evaluation by a well-established benchmark of varied, strong adaptive attacks, across diverse image datasets and classifier architectures, shows that CARSO is able to defend the classifier significantly better than state-of-the-art adversarial training alone -- with a tolerable clean accuracy toll. Furthermore, the defensive architecture succeeds in effectively shielding itself from unforeseen threats, and end-to-end attacks adapted to fool stochastic defences. Code and pre-trained models are available at https://github.com/
    

