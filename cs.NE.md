# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reset It and Forget It: Relearning Last-Layer Weights Improves Continual and Transfer Learning.](http://arxiv.org/abs/2310.07996) | 本研究发展了一种重置最后一层权重的方法，称为"zapping"，通过这种方法可以提供更好的持续和迁移学习效果，同时具备简单实施和高效计算的特点。 |

# 详细

[^1]: 重置并忘却：重新学习最后一层权重改善持续和迁移学习

    Reset It and Forget It: Relearning Last-Layer Weights Improves Continual and Transfer Learning. (arXiv:2310.07996v1 [cs.LG])

    [http://arxiv.org/abs/2310.07996](http://arxiv.org/abs/2310.07996)

    本研究发展了一种重置最后一层权重的方法，称为"zapping"，通过这种方法可以提供更好的持续和迁移学习效果，同时具备简单实施和高效计算的特点。

    

    本研究发现了一种简单的预训练机制，能够导致具有更好的持续和迁移学习表征。这种机制——在最后一层权重中反复重置，我们称之为“zapping”——最初设计用于元持续学习过程，但我们发现它在许多不同于元学习和持续学习的情况下也非常适用。在我们的实验中，我们希望将预训练的图像分类器迁移到一组新的类别，仅使用少量样本。我们展示了我们的zapping过程在标准微调和持续学习设置中能够获得更好的迁移准确性和/或更快的适应性，同时实现简单的实施和高效的计算。在许多情况下，通过使用zapping和顺序学习的组合，我们可以达到与最先进的元学习相当的性能而无需昂贵的高阶梯度。

    This work identifies a simple pre-training mechanism that leads to representations exhibiting better continual and transfer learning. This mechanism -- the repeated resetting of weights in the last layer, which we nickname "zapping" -- was originally designed for a meta-continual-learning procedure, yet we show it is surprisingly applicable in many settings beyond both meta-learning and continual learning. In our experiments, we wish to transfer a pre-trained image classifier to a new set of classes, in a few shots. We show that our zapping procedure results in improved transfer accuracy and/or more rapid adaptation in both standard fine-tuning and continual learning settings, while being simple to implement and computationally efficient. In many cases, we achieve performance on par with state of the art meta-learning without needing the expensive higher-order gradients, by using a combination of zapping and sequential learning. An intuitive explanation for the effectiveness of this za
    

