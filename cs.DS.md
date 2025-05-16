# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning-Augmented B-Trees.](http://arxiv.org/abs/2211.09251) | 这是一个学习增强的B树，通过使用具有复合优先级的Treaps，每个项目的深度由其预测权重确定，推广了最近的学习增强BST，并且是第一个可以利用访问序列中的局部性的B树数据结构。 |

# 详细

[^1]: 学习增强的B树

    Learning-Augmented B-Trees. (arXiv:2211.09251v2 [cs.DS] UPDATED)

    [http://arxiv.org/abs/2211.09251](http://arxiv.org/abs/2211.09251)

    这是一个学习增强的B树，通过使用具有复合优先级的Treaps，每个项目的深度由其预测权重确定，推广了最近的学习增强BST，并且是第一个可以利用访问序列中的局部性的B树数据结构。

    

    本研究通过使用具有复合优先级的Treaps来研究学习增强的二叉搜索树（BST）和B树。结果是一个简单的搜索树，其中每个项目的深度由其预测权重$w_x$确定。为了实现这个结果，每个项目$x$都有其复合优先级$-\lfloor\log\log(1/w_x)\rfloor + U(0, 1)$，其中$U(0, 1)$是均匀分布的随机变量。这将最近的学习增强BST（Lin-Luo-Woodruff ICML`22）推广到任意输入和预测，而不仅仅适用于Zipfian分布。它还提供了第一个可以根据访问序列中的局部性进行在线自我重组的B树数据结构。该数据结构对于预测错误是健壮的，可以处理插入、删除以及预测更新。

    We study learning-augmented binary search trees (BSTs) and B-Trees via Treaps with composite priorities. The result is a simple search tree where the depth of each item is determined by its predicted weight $w_x$. To achieve the result, each item $x$ has its composite priority $-\lfloor\log\log(1/w_x)\rfloor + U(0, 1)$ where $U(0, 1)$ is the uniform random variable. This generalizes the recent learning-augmented BSTs [Lin-Luo-Woodruff ICML`22], which only work for Zipfian distributions, to arbitrary inputs and predictions. It also gives the first B-Tree data structure that can provably take advantage of localities in the access sequence via online self-reorganization. The data structure is robust to prediction errors and handles insertions, deletions, as well as prediction updates.
    

