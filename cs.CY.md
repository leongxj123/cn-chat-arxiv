# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Counterfactual Fairness for Predictions using Generative Adversarial Networks.](http://arxiv.org/abs/2310.17687) | 这篇论文提出了一种使用生成对抗网络实现对照因果公平性的方法，通过学习敏感属性的后代的对照分布来确保公平预测。 |

# 详细

[^1]: 使用生成对抗网络进行预测的对照因果公平性

    Counterfactual Fairness for Predictions using Generative Adversarial Networks. (arXiv:2310.17687v1 [cs.LG])

    [http://arxiv.org/abs/2310.17687](http://arxiv.org/abs/2310.17687)

    这篇论文提出了一种使用生成对抗网络实现对照因果公平性的方法，通过学习敏感属性的后代的对照分布来确保公平预测。

    

    由于法律、伦理和社会原因，预测中的公平性在实践中非常重要。通常通过对照因果公平性来实现，该公平性确保个体的预测与在不同敏感属性下的对照世界中的预测相同。然而，要实现对照因果公平性是具有挑战性的，因为对照是不可观察的。在本文中，我们开发了一种新颖的深度神经网络，称为对照因果公平性生成对抗网络（GCFN），用于在对照因果公平性下进行预测。具体而言，我们利用一个量身定制的生成对抗网络直接学习敏感属性的后代的对照分布，然后通过一种新颖的对照媒介正则化来实施公平预测。如果对照分布学习得足够好，我们的方法在数学上确保对照因果公平性的概念。因此，我们的GCFN解决了对照因果公平性问题。

    Fairness in predictions is of direct importance in practice due to legal, ethical, and societal reasons. It is often achieved through counterfactual fairness, which ensures that the prediction for an individual is the same as that in a counterfactual world under a different sensitive attribute. However, achieving counterfactual fairness is challenging as counterfactuals are unobservable. In this paper, we develop a novel deep neural network called Generative Counterfactual Fairness Network (GCFN) for making predictions under counterfactual fairness. Specifically, we leverage a tailored generative adversarial network to directly learn the counterfactual distribution of the descendants of the sensitive attribute, which we then use to enforce fair predictions through a novel counterfactual mediator regularization. If the counterfactual distribution is learned sufficiently well, our method is mathematically guaranteed to ensure the notion of counterfactual fairness. Thereby, our GCFN addre
    

