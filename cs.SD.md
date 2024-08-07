# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PBSCSR: The Piano Bootleg Score Composer Style Recognition Dataset.](http://arxiv.org/abs/2401.16803) | 本研究介绍了PBSCSR数据集，用于研究钢琴乐谱作曲家风格识别。数据集包含了盗版乐谱图像和相关元数据，可以进行多个研究任务。 |

# 详细

[^1]: PBSCSR：钢琴黑市乐谱作曲家风格识别数据集

    PBSCSR: The Piano Bootleg Score Composer Style Recognition Dataset. (arXiv:2401.16803v1 [cs.SD])

    [http://arxiv.org/abs/2401.16803](http://arxiv.org/abs/2401.16803)

    本研究介绍了PBSCSR数据集，用于研究钢琴乐谱作曲家风格识别。数据集包含了盗版乐谱图像和相关元数据，可以进行多个研究任务。

    

    本文介绍了PBSCSR数据集，用于研究钢琴乐谱作曲家风格识别。我们的目标是创建一个研究作曲家风格识别的数据集，它既像MNIST一样易于获取，又像ImageNet一样具有挑战性。为了实现这个目标，我们从IMSLP的钢琴乐谱图像中采样固定长度的盗版乐谱片段。数据集本身包含40,000个62x64的盗版乐谱图像，用于进行9分类任务，以及100,000个62x64的盗版乐谱图像，用于进行100分类任务，还有29,310个无标签的可变长度的盗版乐谱图像，用于预训练。标记数据以与MNIST图像类似的形式呈现，以便极其方便地可视化、操作和训练模型。此外，我们还包括相关的元数据，以允许访问IMSLP上的原始乐谱图像和其他相关数据。我们描述了几个可以使用该数据进行研究的任务。

    This article motivates, describes, and presents the PBSCSR dataset for studying composer style recognition of piano sheet music. Our overarching goal was to create a dataset for studying composer style recognition that is "as accessible as MNIST and as challenging as ImageNet." To achieve this goal, we sample fixed-length bootleg score fragments from piano sheet music images on IMSLP. The dataset itself contains 40,000 62x64 bootleg score images for a 9-way classification task, 100,000 62x64 bootleg score images for a 100-way classification task, and 29,310 unlabeled variable-length bootleg score images for pretraining. The labeled data is presented in a form that mirrors MNIST images, in order to make it extremely easy to visualize, manipulate, and train models in an efficient manner. Additionally, we include relevant metadata to allow access to the underlying raw sheet music images and other related data on IMSLP. We describe several research tasks that could be studied with the data
    

