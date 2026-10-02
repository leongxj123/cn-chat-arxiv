# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Nowcasting with signature methods.](http://arxiv.org/abs/2305.10256) | 该论文提出了利用路径签名进行回归建模，通过线性模型嵌入连续时间处理了混杂频率和不规则采样带来的缺失数据问题，相较于卡尔曼滤波器更为优秀。该方法已应用于金融、医学和网络安全领域达到最先进水平。 |
| [^2] | [An axiomatic theory for anonymized risk sharing.](http://arxiv.org/abs/2208.07533) | 本文研究了匿名风险分享的公理框架，提出和讨论了四个公理，并发现这四个公理特征化了条件平均风险分享规则，在相关应用中具有独特而重要的地位。 |

# 详细

[^1]: 基于特征数据的实时预测方法

    Nowcasting with signature methods. (arXiv:2305.10256v1 [econ.EM])

    [http://arxiv.org/abs/2305.10256](http://arxiv.org/abs/2305.10256)

    该论文提出了利用路径签名进行回归建模，通过线性模型嵌入连续时间处理了混杂频率和不规则采样带来的缺失数据问题，相较于卡尔曼滤波器更为优秀。该方法已应用于金融、医学和网络安全领域达到最先进水平。

    

    许多重要的经济变量经常延迟一个月以上才能公布。当前预测方法已经在快速、可靠地估算经济滞后指标方面发挥了作用，与信号处理中的滤波方法密切相关。路径签名是一种数学对象，它捕捉序列数据的几何属性；它通过将观察到的数据嵌入连续时间，自然地处理混合频率和/或不规则采样带来的缺失数据问题，在金融、医学和网络安全等领域的应用已经达到了最先进的水平。我们通过对签名回归进行简单的线性建模来研究现在预测问题，这种方法比流行的卡尔曼滤波器更优秀。我们通过模拟实验量化了性能，并通过对美国GDP增长的预测应用进行了说明。

    Key economic variables are often published with a significant delay of over a month. The nowcasting literature has arisen to provide fast, reliable estimates of delayed economic indicators and is closely related to filtering methods in signal processing. The path signature is a mathematical object which captures geometric properties of sequential data; it naturally handles missing data from mixed frequency and/or irregular sampling -- issues often encountered when merging multiple data sources -- by embedding the observed data in continuous time. Calculating path signatures and using them as features in models has achieved state-of-the-art results in fields such as finance, medicine, and cyber security. We look at the nowcasting problem by applying regression on signatures, a simple linear model on these nonlinear objects that we show subsumes the popular Kalman filter. We quantify the performance via a simulation exercise, and through application to nowcasting US GDP growth, where we 
    
[^2]: 匿名风险分享的公理理论研究

    An axiomatic theory for anonymized risk sharing. (arXiv:2208.07533v4 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2208.07533](http://arxiv.org/abs/2208.07533)

    本文研究了匿名风险分享的公理框架，提出和讨论了四个公理，并发现这四个公理特征化了条件平均风险分享规则，在相关应用中具有独特而重要的地位。

    

    我们研究了匿名风险分享的公理框架。与传统的风险分享设置不同，我们的框架不需要有关个体代理的偏好、身份、私人操作和已实现损失的任何信息，因此它对于建模去中心化系统的风险分享是有用的。我们提出和讨论了在这种框架下自然的四个公理——精算公平性、风险公平性、风险匿名性和操作匿名性。我们发现这四个公理特征化了条件平均风险分享规则，揭示了这种广泛使用的风险分享规则在相关的匿名风险分享应用中的独特而重要的地位。我们研究了几个其他性质及其与四个公理的关系，以及它们在合理化实践中某些分享机制的设计中的影响。

    We study an axiomatic framework for anonymized risk sharing. In contrast to traditional risk sharing settings, our framework requires no information on preferences, identities, private operations and realized losses from the individual agents, and thereby it is useful for modeling risk sharing in decentralized systems. Four axioms natural in such a framework -- actuarial fairness, risk fairness, risk anonymity, and operational anonymity -- are put forward and discussed. We establish the remarkable fact that the four axioms characterizes the conditional mean risk sharing rule, revealing the unique and prominent role of this popular risk sharing rule among all others in relevant applications of anonymized risk sharing. Several other properties and their relations to the four axioms are studied, as well as their implications in rationalizing the design of some sharing mechanisms in practice.
    

