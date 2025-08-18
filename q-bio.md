# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Planning to Learn: A Novel Algorithm for Active Learning during Model-Based Planning.](http://arxiv.org/abs/2308.08029) | 本论文提出了一种新颖的算法，称为SI SL，用于主动学习和模型优化过程中的规划。该算法通过与贝叶斯强化学习方案的比较证明了其性能的优越性。 |

# 详细

[^1]: 规划学习：一种新颖的模型优化过程中的主动学习算法

    Planning to Learn: A Novel Algorithm for Active Learning during Model-Based Planning. (arXiv:2308.08029v1 [cs.AI])

    [http://arxiv.org/abs/2308.08029](http://arxiv.org/abs/2308.08029)

    本论文提出了一种新颖的算法，称为SI SL，用于主动学习和模型优化过程中的规划。该算法通过与贝叶斯强化学习方案的比较证明了其性能的优越性。

    

    主动推理是一种近期的对不确定性情境下规划建模的框架。现在人们已经开始评估这种方法的优缺点以及如何改进它。最近的一个拓展-复杂模型优化算法通过递归决策树搜索在多步规划问题上提高了性能。然而，迄今为止很少有工作对比SI与其他已建立的规划算法。SI算法也主要关注推理而不是学习。本文有两个目标。首先，我们比较SI与旨在解决相似问题的贝叶斯强化学习（RL）方案的性能。其次，我们提出了SI复杂学习（SL）的拓展，该拓展在规划过程中更加充分地引入了主动学习。SL维持对未来观测下每个策略下模型参数如何变化的信念。这允许了一种反事实的回顾性评估。

    Active Inference is a recent framework for modeling planning under uncertainty. Empirical and theoretical work have now begun to evaluate the strengths and weaknesses of this approach and how it might be improved. A recent extension - the sophisticated inference (SI) algorithm - improves performance on multi-step planning problems through recursive decision tree search. However, little work to date has been done to compare SI to other established planning algorithms. SI was also developed with a focus on inference as opposed to learning. The present paper has two aims. First, we compare performance of SI to Bayesian reinforcement learning (RL) schemes designed to solve similar problems. Second, we present an extension of SI sophisticated learning (SL) - that more fully incorporates active learning during planning. SL maintains beliefs about how model parameters would change under the future observations expected under each policy. This allows a form of counterfactual retrospective in
    

