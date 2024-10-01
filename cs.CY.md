# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Understanding random forests and overfitting: a visualization and simulation study](https://arxiv.org/abs/2402.18612) | 这项研究通过可视化和模拟研究探讨了随机森林的行为，发现在训练集存在过拟合的情况下，模型在测试数据上表现出了竞争力。 |
| [^2] | [Interpretable Stereotype Identification through Reasoning.](http://arxiv.org/abs/2308.00071) | 本研究通过使用推理方法，在零射击刻板印象识别中取得了重要的进展，并发现推理的性能增益远远超过模型规模扩展的增益。推理不仅提高了准确性，还提高了决策的可解释性。 |

# 详细

[^1]: 理解随机森林和过拟合：一项可视化和模拟研究

    Understanding random forests and overfitting: a visualization and simulation study

    [https://arxiv.org/abs/2402.18612](https://arxiv.org/abs/2402.18612)

    这项研究通过可视化和模拟研究探讨了随机森林的行为，发现在训练集存在过拟合的情况下，模型在测试数据上表现出了竞争力。

    

    随机森林在临床风险预测建模中变得流行。在一项关于预测卵巢恶性的案例研究中，我们观察到训练集上的c-统计值接近1。尽管这表明存在过拟合，但在测试数据上表现竞争力。我们旨在通过（1）在三个真实案例研究中可视化数据空间和（2）进行模拟研究来理解随机森林的行为。在案例研究中，使用热力图在二维子空间中可视化风险估计。模拟研究包括48个逻辑数据生成机制（DGM），变化预测变量分布、预测变量数量、预测变量之间的相关性、真实c-统计值和真实预测变量的强度。对于每个DGM，模拟生成大小为200或4000的1000个训练数据集，并使用ranger包训练最小节点大小为2或20的随机森林模型，总共得到了192个场景。可视化结果表明，模型…

    arXiv:2402.18612v1 Announce Type: cross  Abstract: Random forests have become popular for clinical risk prediction modelling. In a case study on predicting ovarian malignancy, we observed training c-statistics close to 1. Although this suggests overfitting, performance was competitive on test data. We aimed to understand the behaviour of random forests by (1) visualizing data space in three real world case studies and (2) a simulation study. For the case studies, risk estimates were visualised using heatmaps in a 2-dimensional subspace. The simulation study included 48 logistic data generating mechanisms (DGM), varying the predictor distribution, the number of predictors, the correlation between predictors, the true c-statistic and the strength of true predictors. For each DGM, 1000 training datasets of size 200 or 4000 were simulated and RF models trained with minimum node size 2 or 20 using ranger package, resulting in 192 scenarios in total. The visualizations suggested that the mod
    
[^2]: 可解释的推理方法用于刻板印象识别

    Interpretable Stereotype Identification through Reasoning. (arXiv:2308.00071v1 [cs.CL])

    [http://arxiv.org/abs/2308.00071](http://arxiv.org/abs/2308.00071)

    本研究通过使用推理方法，在零射击刻板印象识别中取得了重要的进展，并发现推理的性能增益远远超过模型规模扩展的增益。推理不仅提高了准确性，还提高了决策的可解释性。

    

    鉴于语言模型训练使用了包含固有偏见的大量数据集，可能会不经意地持续系统性歧视，因此，审查和解决语言模型中的偏见变得至关重要，将公平性整合到它们的发展中，以确保这些模型具有公正和无偏的特性。在这项工作中，我们展示了基于Vicuna-13B-v1.3的零射击刻板印象识别中推理的重要性。尽管我们观察到从13B到33B的规模扩展会提高准确性，但我们表明推理的性能增益远远超过规模扩展的增益。我们的研究结果表明，推理可能是使LLMs在刻板印象等领域任务上超越规模定律的关键因素。此外，通过对选定的推理追踪进行定性分析，我们突出显示了推理不仅提高了准确性，还提高了决策的可解释性。

    Given that language models are trained on vast datasets that may contain inherent biases, there is a potential danger of inadvertently perpetuating systemic discrimination. Consequently, it becomes essential to examine and address biases in language models, integrating fairness into their development to ensure these models are equitable and free from bias. In this work, we demonstrate the importance of reasoning in zero-shot stereotype identification based on Vicuna-13B-v1.3. While we do observe improved accuracy by scaling from 13B to 33B, we show that the performance gain from reasoning significantly exceeds the gain from scaling up. Our findings suggest that reasoning could be a key factor that enables LLMs to trescend the scaling law on out-of-domain tasks such as stereotype identification. Additionally, through a qualitative analysis of select reasoning traces, we highlight how reasoning enhances not just accuracy but also the interpretability of the decision.
    

