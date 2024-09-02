# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Invariant Causal Prediction with Locally Linear Models.](http://arxiv.org/abs/2401.05218) | 本文扩展了ICP原则，考虑了在不同环境下具有局部线性模型的不变因果预测任务。通过提供因果父节点的可辨识性条件和引入LoLICaP方法，实现了在观察数据中识别目标变量的因果父节点。 |

# 详细

[^1]: 具有局部线性模型的不变因果预测

    Invariant Causal Prediction with Locally Linear Models. (arXiv:2401.05218v1 [cs.LG])

    [http://arxiv.org/abs/2401.05218](http://arxiv.org/abs/2401.05218)

    本文扩展了ICP原则，考虑了在不同环境下具有局部线性模型的不变因果预测任务。通过提供因果父节点的可辨识性条件和引入LoLICaP方法，实现了在观察数据中识别目标变量的因果父节点。

    

    本文考虑通过观察数据，从一组候选变量中识别出目标变量的因果父节点的任务。我们的主要假设是候选变量在不同的环境中被观察到，这些环境可以对应于机器的不同设置或者动态过程中的不同时间间隔等。在一定的假设条件下，不同的环境可以被视为对观察系统的干预。我们假设目标变量和协变量之间存在线性关系，在每个环境下可能不同，但因果结构在不同环境中是不变的。这是Peters等人[2016]提出的ICP（不变因果预测）原则的扩展，后者假设所有环境下存在一个固定的线性关系。在我们提出的设置下，我们给出了因果父节点可辨识性的充分条件，并引入了一个名为LoLICaP的实用方法。

    We consider the task of identifying the causal parents of a target variable among a set of candidate variables from observational data. Our main assumption is that the candidate variables are observed in different environments which may, for example, correspond to different settings of a machine or different time intervals in a dynamical process. Under certain assumptions different environments can be regarded as interventions on the observed system. We assume a linear relationship between target and covariates, which can be different in each environment with the only restriction that the causal structure is invariant across environments. This is an extension of the ICP ($\textbf{I}$nvariant $\textbf{C}$ausal $\textbf{P}$rediction) principle by Peters et al. [2016], who assumed a fixed linear relationship across all environments. Within our proposed setting we provide sufficient conditions for identifiability of the causal parents and introduce a practical method called LoLICaP ($\text
    

