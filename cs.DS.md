# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning-Augmented Skip Lists](https://arxiv.org/abs/2402.10457) | 通过将机器学习建议与跳表设计整合，提出了一种学习增强型跳表，能够实现最优期望搜索时间，在处理搜索查询时具有显著改进。 |

# 详细

[^1]: 学习增强型跳表

    Learning-Augmented Skip Lists

    [https://arxiv.org/abs/2402.10457](https://arxiv.org/abs/2402.10457)

    通过将机器学习建议与跳表设计整合，提出了一种学习增强型跳表，能够实现最优期望搜索时间，在处理搜索查询时具有显著改进。

    

    我们研究了将机器学习建议整合到跳表设计中，以改进传统数据结构设计。通过访问可能有误的预测分数频率的预测值的预言神谕，我们构建了一个跳表，可以证明提供最佳的期望搜索时间，几乎有二倍的优势。事实上，我们的学习增强型跳表仍然是最佳的，即使神谕只在常数因子内准确。我们表明，如果搜索查询遵循普遍存在的Zipfian分布，那么我们的跳表对于一个项目的期望搜索时间仅为一个常数，与项目总数n无关，即O(1)，而传统的跳表的期望搜索时间为O(log n)。我们还展示了我们的数据结构的鲁棒性，通过展示我们的数据结构实现了一个期望搜索时间，

    arXiv:2402.10457v1 Announce Type: cross  Abstract: We study the integration of machine learning advice into the design of skip lists to improve upon traditional data structure design. Given access to a possibly erroneous oracle that outputs estimated fractional frequencies for search queries on a set of items, we construct a skip list that provably provides the optimal expected search time, within nearly a factor of two. In fact, our learning-augmented skip list is still optimal up to a constant factor, even if the oracle is only accurate within a constant factor. We show that if the search queries follow the ubiquitous Zipfian distribution, then the expected search time for an item by our skip list is only a constant, independent of the total number $n$ of items, i.e., $\mathcal{O}(1)$, whereas a traditional skip list will have an expected search time of $\mathcal{O}(\log n)$. We also demonstrate robustness by showing that our data structure achieves an expected search time that is wi
    

