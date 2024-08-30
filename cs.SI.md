# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Normalized mutual information is a biased measure for classification and community detection.](http://arxiv.org/abs/2307.01282) | 标准化归一互信息是一种偏倚度量，因为它忽略了条件表的信息内容并且对算法输出有噪声依赖。本文提出了一种修正版本的互信息，并通过对网络社区检测算法的测试证明了使用无偏度量的重要性。 |

# 详细

[^1]: 标准化归一互信息是分类和社区检测的一种偏倚度量

    Normalized mutual information is a biased measure for classification and community detection. (arXiv:2307.01282v1 [cs.SI] CROSS LISTED)

    [http://arxiv.org/abs/2307.01282](http://arxiv.org/abs/2307.01282)

    标准化归一互信息是一种偏倚度量，因为它忽略了条件表的信息内容并且对算法输出有噪声依赖。本文提出了一种修正版本的互信息，并通过对网络社区检测算法的测试证明了使用无偏度量的重要性。

    

    标准归一互信息被广泛用作评估聚类和分类算法性能的相似性度量。本文表明标准化归一互信息的结果有两个偏倚因素：首先，因为它们忽略了条件表的信息内容；其次，因为它们的对称归一化引入了对算法输出的噪声依赖。我们提出了一种修正版本的互信息，解决了这两个缺陷。通过对网络社区检测中一篮子流行算法进行大量数值测试，我们展示了使用无偏度量的重要性，并且显示传统互信息中的偏倚对选择最佳算法的结论产生了显著影响。

    Normalized mutual information is widely used as a similarity measure for evaluating the performance of clustering and classification algorithms. In this paper, we show that results returned by the normalized mutual information are biased for two reasons: first, because they ignore the information content of the contingency table and, second, because their symmetric normalization introduces spurious dependence on algorithm output. We introduce a modified version of the mutual information that remedies both of these shortcomings. As a practical demonstration of the importance of using an unbiased measure, we perform extensive numerical tests on a basket of popular algorithms for network community detection and show that one's conclusions about which algorithm is best are significantly affected by the biases in the traditional mutual information.
    

