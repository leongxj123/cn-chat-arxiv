# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Collective Counterfactual Explanations via Optimal Transport](https://arxiv.org/abs/2402.04579) | 本论文提出了一种集体方法来形成反事实解释，通过利用个体的当前密度来指导推荐的行动，解决了个体为中心的方法可能导致的新的竞争和意想不到的成本问题，并改进了经典反事实解释的期望。 |

# 详细

[^1]: 通过最优传输实现集体反事实解释

    Collective Counterfactual Explanations via Optimal Transport

    [https://arxiv.org/abs/2402.04579](https://arxiv.org/abs/2402.04579)

    本论文提出了一种集体方法来形成反事实解释，通过利用个体的当前密度来指导推荐的行动，解决了个体为中心的方法可能导致的新的竞争和意想不到的成本问题，并改进了经典反事实解释的期望。

    

    反事实解释提供个体的成本最优行动，以改变标签为所需的类别。然而，如果大量实例寻求状态修改，这种个体为中心的方法可能导致新的竞争和意想不到的成本。此外，这些推荐忽视了基础数据分布，可能会建议用户认为是异常值的行动。为了解决这些问题，我们的工作提出了一种集体方法来形成反事实解释，重点是利用个体的当前密度来指导推荐的行动。我们的问题自然地转化为一个最优传输问题。借鉴最优传输的广泛文献，我们说明了这种集体方法如何改进经典反事实解释的期望。我们通过数值模拟支持我们的提议，展示了所提方法的有效性以及与经典方法的关系。

    Counterfactual explanations provide individuals with cost-optimal actions that can alter their labels to desired classes. However, if substantial instances seek state modification, such individual-centric methods can lead to new competitions and unanticipated costs. Furthermore, these recommendations, disregarding the underlying data distribution, may suggest actions that users perceive as outliers. To address these issues, our work proposes a collective approach for formulating counterfactual explanations, with an emphasis on utilizing the current density of the individuals to inform the recommended actions. Our problem naturally casts as an optimal transport problem. Leveraging the extensive literature on optimal transport, we illustrate how this collective method improves upon the desiderata of classical counterfactual explanations. We support our proposal with numerical simulations, illustrating the effectiveness of the proposed approach and its relation to classic methods.
    

