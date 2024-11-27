# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Conformal Prediction is Robust to Dispersive Label Noise.](http://arxiv.org/abs/2209.14295) | 本研究研究了Conformal Prediction方法对于标签噪声具有鲁棒性。我们找出了构建可以正确覆盖无噪声真实标签的不确定性集合的条件，并提出了对具有噪声标签的一般损失函数进行正确控制的要求。实验证明，在对抗性案例之外，使用Conformal Prediction和风险控制技术可以实现对干净真实标签的保守风险。我们还提出了一种有界尺寸噪声修正的方法，以确保实现正确的真实标签风险。 |

# 详细

[^1]: Conformal Prediction对分散标签噪声具有稳健性

    Conformal Prediction is Robust to Dispersive Label Noise. (arXiv:2209.14295v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.14295](http://arxiv.org/abs/2209.14295)

    本研究研究了Conformal Prediction方法对于标签噪声具有鲁棒性。我们找出了构建可以正确覆盖无噪声真实标签的不确定性集合的条件，并提出了对具有噪声标签的一般损失函数进行正确控制的要求。实验证明，在对抗性案例之外，使用Conformal Prediction和风险控制技术可以实现对干净真实标签的保守风险。我们还提出了一种有界尺寸噪声修正的方法，以确保实现正确的真实标签风险。

    

    我们研究了对标签噪声具有鲁棒性的Conformal Prediction方法，该方法是一种用于不确定性量化的强大工具。我们的分析涵盖了回归和分类问题，对于如何构建能够正确覆盖未观察到的无噪声真实标签的不确定性集合进行了界定。我们进一步扩展了我们的理论，并提出了对于带有噪声标签正确控制一般损失函数（如假阴性比例）的要求。我们的理论和实验表明，在带有噪声标签的情况下，Conformal Prediction和风险控制技术能够实现对干净真实标签的保守风险，除了在对抗性案例中。在这种情况下，我们还可以通过对Conformal Prediction算法进行有界尺寸的噪声修正，以确保实现正确的真实标签风险，而无需考虑分数或数据的规则性。

    We study the robustness of conformal prediction, a powerful tool for uncertainty quantification, to label noise. Our analysis tackles both regression and classification problems, characterizing when and how it is possible to construct uncertainty sets that correctly cover the unobserved noiseless ground truth labels. We further extend our theory and formulate the requirements for correctly controlling a general loss function, such as the false negative proportion, with noisy labels. Our theory and experiments suggest that conformal prediction and risk-controlling techniques with noisy labels attain conservative risk over the clean ground truth labels except in adversarial cases. In such cases, we can also correct for noise of bounded size in the conformal prediction algorithm in order to ensure achieving the correct risk of the ground truth labels without score or data regularity.
    

