# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Does AI help humans make better decisions? A methodological framework for experimental evaluation](https://arxiv.org/abs/2403.12108) | 引入一种新的实验框架用于评估人类是否通过使用AI可以做出更好的决策，在单盲实验设计中比较了三种决策系统的表现 |
| [^2] | [Bayesian Federated Inference for regression models with heterogeneous multi-center populations](https://arxiv.org/abs/2402.02898) | 这项研究提出了一种利用贝叶斯联合推断方法，在不同中心分别分析本地数据，并将统计推断结果组合起来，以解决样本量不足的问题，并准确估计回归模型的参数。 |

# 详细

[^1]: AI是否有助于人类做出更好的决策？一种用于实验评估的方法论框架

    Does AI help humans make better decisions? A methodological framework for experimental evaluation

    [https://arxiv.org/abs/2403.12108](https://arxiv.org/abs/2403.12108)

    引入一种新的实验框架用于评估人类是否通过使用AI可以做出更好的决策，在单盲实验设计中比较了三种决策系统的表现

    

    基于数据驱动算法的人工智能（AI）在当今社会变得无处不在。然而，在许多情况下，尤其是当利益高昂时，人类仍然作出最终决策。因此，关键问题是AI是否有助于人类比单独的人类或单独的AI做出更好的决策。我们引入了一种新的方法论框架，用于实验性地回答这个问题，而不需要额外的假设。我们使用基于基准潜在结果的标准分类指标测量决策者做出正确决策的能力。我们考虑了一个单盲实验设计，在这个设计中，提供AI生成的建议在不同案例中被随机分配给最终决策的人类。在这种实验设计下，我们展示了如何比较三种替代决策系统的性能--仅人类、人类与AI、仅AI。

    arXiv:2403.12108v1 Announce Type: new  Abstract: The use of Artificial Intelligence (AI) based on data-driven algorithms has become ubiquitous in today's society. Yet, in many cases and especially when stakes are high, humans still make final decisions. The critical question, therefore, is whether AI helps humans make better decisions as compared to a human alone or AI an alone. We introduce a new methodological framework that can be used to answer experimentally this question with no additional assumptions. We measure a decision maker's ability to make correct decisions using standard classification metrics based on the baseline potential outcome. We consider a single-blinded experimental design, in which the provision of AI-generated recommendations is randomized across cases with a human making final decisions. Under this experimental design, we show how to compare the performance of three alternative decision-making systems--human-alone, human-with-AI, and AI-alone. We apply the pr
    
[^2]: 具有异质多中心人群的回归模型的贝叶斯联合推断

    Bayesian Federated Inference for regression models with heterogeneous multi-center populations

    [https://arxiv.org/abs/2402.02898](https://arxiv.org/abs/2402.02898)

    这项研究提出了一种利用贝叶斯联合推断方法，在不同中心分别分析本地数据，并将统计推断结果组合起来，以解决样本量不足的问题，并准确估计回归模型的参数。

    

    为了准确估计回归模型的参数，样本量必须相对于可能的预测变量个数足够大。在实际应用中，通常缺乏足够的数据，这可能导致模型过拟合，并因此无法对新患者的结果进行可靠预测。合并来自不同（医疗）中心收集的数据可以缓解这个问题，但通常由于隐私法规或物流问题而不可行。另一种方法是分析各个中心的本地数据，然后使用贝叶斯联合推断（BFI）方法将统计推断结果进行组合。这种方法的目标是从各个中心的推断结果中计算出如果对组合数据进行了统计分析后会得到什么结果。我们解释了同质和异质中心人群下的方法，并给出了真实的示例。

    To estimate accurately the parameters of a regression model, the sample size must be large enough relative to the number of possible predictors for the model. In practice, sufficient data is often lacking, which can lead to overfitting of the model and, as a consequence, unreliable predictions of the outcome of new patients. Pooling data from different data sets collected in different (medical) centers would alleviate this problem, but is often not feasible due to privacy regulation or logistic problems. An alternative route would be to analyze the local data in the centers separately and combine the statistical inference results with the Bayesian Federated Inference (BFI) methodology. The aim of this approach is to compute from the inference results in separate centers what would have been found if the statistical analysis was performed on the combined data. We explain the methodology under homogeneity and heterogeneity across the populations in the separate centers, and give real lif
    

