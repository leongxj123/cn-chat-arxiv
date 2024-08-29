# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Examining Pathological Bias in a Generative Adversarial Network Discriminator: A Case Study on a StyleGAN3 Model](https://arxiv.org/abs/2402.09786) | 这项研究发现了StyleGAN3模型中判别器的病态偏见，它在图像和面部质量上的得分分层影响了不同性别、种族和其他类别的图像。 |

# 详细

[^1]: 检查生成对抗网络判别器中的病态偏见：以StyleGAN3模型为例的案例研究

    Examining Pathological Bias in a Generative Adversarial Network Discriminator: A Case Study on a StyleGAN3 Model

    [https://arxiv.org/abs/2402.09786](https://arxiv.org/abs/2402.09786)

    这项研究发现了StyleGAN3模型中判别器的病态偏见，它在图像和面部质量上的得分分层影响了不同性别、种族和其他类别的图像。

    

    生成对抗网络可以生成逼真的人脸，往往难以被人类区分出来。我们发现预训练的StyleGAN3模型中的判别器在图像和面部质量上系统地对得分进行分层，并且这不成比例地影响了不同性别、种族和其他类别的图像。我们检查了判别器在色彩和亮度方面对感知的种族和性别的偏见，然后检查了社会心理学中关于刻板印象研究中常见的偏见。

    arXiv:2402.09786v1 Announce Type: cross  Abstract: Generative adversarial networks generate photorealistic faces that are often indistinguishable by humans from real faces. We find that the discriminator in the pre-trained StyleGAN3 model, a popular GAN network, systematically stratifies scores by both image- and face-level qualities and that this disproportionately affects images across gender, race, and other categories. We examine the discriminator's bias for color and luminance across axes perceived race and gender; we then examine axes common in research on stereotyping in social psychology.
    

