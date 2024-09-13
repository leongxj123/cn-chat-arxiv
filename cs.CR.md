# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Transfer Attack to Image Watermarks](https://arxiv.org/abs/2403.15365) | 水印领域的研究表明，即使在攻击者无法访问水印模型或检测API的情况下，水印基础的AI生成图像检测器也无法抵抗对抗攻击。 |

# 详细

[^1]: 一种针对图像水印的转移攻击

    A Transfer Attack to Image Watermarks

    [https://arxiv.org/abs/2403.15365](https://arxiv.org/abs/2403.15365)

    水印领域的研究表明，即使在攻击者无法访问水印模型或检测API的情况下，水印基础的AI生成图像检测器也无法抵抗对抗攻击。

    

    水印已被广泛应用于工业领域，用于检测由人工智能生成的图像。文献中对这种基于水印的检测器在白盒和黑盒环境下对抗攻击的稳健性有很好的理解。然而，在无盒环境下的稳健性却知之甚少。具体来说，多项研究声称图像水印在这种环境下是稳健的。在这项工作中，我们提出了一种新的转移对抗攻击来针对无盒环境下的图像水印。我们的转移攻击向带水印的图像添加微扰，以躲避被攻击者训练的多个替代水印模型，并且经过扰动的带水印图像也能躲避目标水印模型。我们的主要贡献是理论上和经验上展示了，基于水印的人工智能生成图像检测器即使攻击者没有访问水印模型或检测API，也不具有对抗攻击的稳健性。

    arXiv:2403.15365v1 Announce Type: cross  Abstract: Watermark has been widely deployed by industry to detect AI-generated images. The robustness of such watermark-based detector against evasion attacks in the white-box and black-box settings is well understood in the literature. However, the robustness in the no-box setting is much less understood. In particular, multiple studies claimed that image watermark is robust in such setting. In this work, we propose a new transfer evasion attack to image watermark in the no-box setting. Our transfer attack adds a perturbation to a watermarked image to evade multiple surrogate watermarking models trained by the attacker itself, and the perturbed watermarked image also evades the target watermarking model. Our major contribution is to show that, both theoretically and empirically, watermark-based AI-generated image detector is not robust to evasion attacks even if the attacker does not have access to the watermarking model nor the detection API.
    

