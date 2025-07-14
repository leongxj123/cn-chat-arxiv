# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Estimation of conditional average treatment effects on distributed data: A privacy-preserving approach](https://arxiv.org/abs/2402.02672) | 本论文提出了一种数据协作双机器学习（DC-DML）方法，该方法可以在保护分布式数据隐私的情况下估计条件平均治疗效果（CATE）模型。通过数值实验验证了该方法的有效性。该方法的三个主要贡献是：实现了对分布式数据上的非迭代通信的半参数CATE模型的估计和测试，提高了模型的鲁棒性。 |

# 详细

[^1]: 对分布式数据的条件平均治疗效果估计：一种保护隐私的方法

    Estimation of conditional average treatment effects on distributed data: A privacy-preserving approach

    [https://arxiv.org/abs/2402.02672](https://arxiv.org/abs/2402.02672)

    本论文提出了一种数据协作双机器学习（DC-DML）方法，该方法可以在保护分布式数据隐私的情况下估计条件平均治疗效果（CATE）模型。通过数值实验验证了该方法的有效性。该方法的三个主要贡献是：实现了对分布式数据上的非迭代通信的半参数CATE模型的估计和测试，提高了模型的鲁棒性。

    

    在医学和社会科学等各个领域中，对条件平均治疗效果（CATEs）的估计是一个重要的课题。如果分布在多个参与方之间的数据可以集中，可以对CATEs进行高精度的估计。然而，如果这些数据包含隐私信息，则很难进行数据聚合。为了解决这个问题，我们提出了数据协作双机器学习（DC-DML）方法，该方法可以在保护分布式数据隐私的情况下估计CATE模型，并通过数值实验对该方法进行了评估。我们的贡献总结如下三点。首先，我们的方法能够在分布式数据上进行非迭代通信的半参数CATE模型的估计和测试。半参数或非参数的CATE模型能够比参数模型更稳健地进行估计和测试，对于模型偏差的鲁棒性更强。然而，据我们所知，目前还没有提出有效的通信方法来估计和测试这些模型。

    Estimation of conditional average treatment effects (CATEs) is an important topic in various fields such as medical and social sciences. CATEs can be estimated with high accuracy if distributed data across multiple parties can be centralized. However, it is difficult to aggregate such data if they contain privacy information. To address this issue, we proposed data collaboration double machine learning (DC-DML), a method that can estimate CATE models with privacy preservation of distributed data, and evaluated the method through numerical experiments. Our contributions are summarized in the following three points. First, our method enables estimation and testing of semi-parametric CATE models without iterative communication on distributed data. Semi-parametric or non-parametric CATE models enable estimation and testing that is more robust to model mis-specification than parametric models. However, to our knowledge, no communication-efficient method has been proposed for estimating and 
    

