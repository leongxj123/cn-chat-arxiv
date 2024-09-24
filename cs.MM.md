# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Boosting Facial Action Unit Detection Through Jointly Learning Facial Landmark Detection and Domain Separation and Reconstruction.](http://arxiv.org/abs/2310.05207) | 本文提出了一种新的面部动作单位（AU）检测框架，通过共享参数和引入多任务学习，在面部标志检测和AU域分离与重建之间实现了更好的性能。实验证明我们方法在野外AU检测方面优于现有方法。 |

# 详细

[^1]: 通过同时学习面部标志检测、域分离和重建来提高面部动作单位检测的精度

    Boosting Facial Action Unit Detection Through Jointly Learning Facial Landmark Detection and Domain Separation and Reconstruction. (arXiv:2310.05207v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2310.05207](http://arxiv.org/abs/2310.05207)

    本文提出了一种新的面部动作单位（AU）检测框架，通过共享参数和引入多任务学习，在面部标志检测和AU域分离与重建之间实现了更好的性能。实验证明我们方法在野外AU检测方面优于现有方法。

    

    最近，如何将大量的在野非标记面部图像引入监督式面部动作单位（AU）检测框架中成为一个具有挑战性的问题。本文提出了一种新的AU检测框架，通过共享同构面部提取模块的参数，引入多任务学习，同时学习AU域分离和重建以及面部标志检测。另外，我们提出了一种基于对比学习的新特征对齐方案，通过简单的投影器和改进的对比损失添加了四个额外的中间监督器来促进特征重建的过程。在两个基准测试上的实验结果表明，我们在野外AU检测方面优于现有的方法。

    Recently how to introduce large amounts of unlabeled facial images in the wild into supervised Facial Action Unit (AU) detection frameworks has become a challenging problem. In this paper, we propose a new AU detection framework where multi-task learning is introduced to jointly learn AU domain separation and reconstruction and facial landmark detection by sharing the parameters of homostructural facial extraction modules. In addition, we propose a new feature alignment scheme based on contrastive learning by simple projectors and an improved contrastive loss, which adds four additional intermediate supervisors to promote the feature reconstruction process. Experimental results on two benchmarks demonstrate our superiority against the state-of-the-art methods for AU detection in the wild.
    

