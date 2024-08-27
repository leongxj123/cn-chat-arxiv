# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Extract Mechanisms from Heterogeneous Effects: Identification Strategy for Mediation Analysis](https://arxiv.org/abs/2403.04131) | 该论文开发了一种新的识别策略，通过将总处理效应进行分解，将复杂的中介问题转化为简单的线性回归问题，实现了因果效应和中介效应的同时估计，建立了新的因果中介和因果调节之间的联系。 |
| [^2] | [When accurate prediction models yield harmful self-fulfilling prophecies](https://arxiv.org/abs/2312.01210) | 本研究通过调查预测模型的部署对决策产生有害影响的情况，发现这些模型可能会成为有害的自我实现预言。这些模型不会因为对某些患者造成更糟糕的结果而使其预测能力变无效。 |
| [^3] | [Who Are We Missing? A Principled Approach to Characterizing the Underrepresented Population.](http://arxiv.org/abs/2401.14512) | 本文提出了一种基于优化的方法，Rashomon Set of Optimal Trees (ROOT)，用于识别和描述随机对照试验中的少数人群。该方法通过最小化目标平均处理效应估计的方差来优化目标子群体分布，从而提供更精确和可解释的处理效应估计。与其他方法相比，该方法具有更高的精度和可解释性，通过合成数据实验进行了验证。 |
| [^4] | [Invariant Causal Prediction with Locally Linear Models.](http://arxiv.org/abs/2401.05218) | 本文扩展了ICP原则，考虑了在不同环境下具有局部线性模型的不变因果预测任务。通过提供因果父节点的可辨识性条件和引入LoLICaP方法，实现了在观察数据中识别目标变量的因果父节点。 |
| [^5] | [On the Efficiency of Finely Stratified Experiments.](http://arxiv.org/abs/2307.15181) | 本文研究了在实验分析中涉及的大类处理效应参数的有效估计，包括平均处理效应、分位数处理效应、局部平均处理效应等。 |
| [^6] | [Causal Estimation of Exposure Shifts with Neural Networks: Evaluating the Health Benefits of Stricter Air Quality Standards in the US.](http://arxiv.org/abs/2302.02560) | 本研究提出了一种神经网络方法，利用其理论基础和实施的可行性，从而估计连续暴露/治疗的分布对政策相关结果的因果效应。我们将此方法应用于包含6800万个个体和2700万个美国境内死亡事件的数据中，通过评估美国国家环境保护局（EPA）对PM2.5的国家环境空气质量标准（NAAQS）进行修订后的健康效益。 |

# 详细

[^1]: 从异质效应中提取机制：中介分析的识别策略

    Extract Mechanisms from Heterogeneous Effects: Identification Strategy for Mediation Analysis

    [https://arxiv.org/abs/2403.04131](https://arxiv.org/abs/2403.04131)

    该论文开发了一种新的识别策略，通过将总处理效应进行分解，将复杂的中介问题转化为简单的线性回归问题，实现了因果效应和中介效应的同时估计，建立了新的因果中介和因果调节之间的联系。

    

    理解因果机制对于解释和概括经验现象至关重要。因果中介分析提供了量化中介效应的统计技术。然而，现有方法通常需要强大的识别假设或复杂的研究设计。我们开发了一种新的识别策略，简化了这些假设，实现了因果效应和中介效应的同时估计。该策略基于总处理效应的新型分解，将具有挑战性的中介问题转化为简单的线性回归问题。新方法建立了因果中介和因果调节之间的新联系。我们讨论了几种研究设计和估计器，以增加我们的识别策略在各种实证研究中的可用性。我们通过在实验中估计因果中介效应来演示我们方法的应用。

    arXiv:2403.04131v1 Announce Type: cross  Abstract: Understanding causal mechanisms is essential for explaining and generalizing empirical phenomena. Causal mediation analysis offers statistical techniques to quantify mediation effects. However, existing methods typically require strong identification assumptions or sophisticated research designs. We develop a new identification strategy that simplifies these assumptions, enabling the simultaneous estimation of causal and mediation effects. The strategy is based on a novel decomposition of total treatment effects, which transforms the challenging mediation problem into a simple linear regression problem. The new method establishes a new link between causal mediation and causal moderation. We discuss several research designs and estimators to increase the usability of our identification strategy for a variety of empirical studies. We demonstrate the application of our method by estimating the causal mediation effect in experiments concer
    
[^2]: 当准确的预测模型导致有害的自我实现预言

    When accurate prediction models yield harmful self-fulfilling prophecies

    [https://arxiv.org/abs/2312.01210](https://arxiv.org/abs/2312.01210)

    本研究通过调查预测模型的部署对决策产生有害影响的情况，发现这些模型可能会成为有害的自我实现预言。这些模型不会因为对某些患者造成更糟糕的结果而使其预测能力变无效。

    

    目标：预测模型在医学研究和实践中非常受欢迎。通过为特定患者预测感兴趣的结果，这些模型可以帮助决策困难的治疗决策，并且通常被誉为个性化的、数据驱动的医疗保健的杰出代表。许多预测模型在验证研究中基于其预测准确性而部署用于决策支持。我们调查这是否是一种安全和有效的方法。材料和方法：我们展示了使用预测模型进行决策可以导致有害的决策，即使在部署后这些预测表现出良好的区分度。这些模型是有害的自我实现预言：它们的部署损害了一群患者，但这些患者的更糟糕的结果并不使模型的预测能力无效。结果：我们的主要结果是对这些预测模型集合的形式化描述。接下来，我们展示了在部署前后都进行了良好校准的模型

    Objective: Prediction models are popular in medical research and practice. By predicting an outcome of interest for specific patients, these models may help inform difficult treatment decisions, and are often hailed as the poster children for personalized, data-driven healthcare. Many prediction models are deployed for decision support based on their prediction accuracy in validation studies. We investigate whether this is a safe and valid approach.   Materials and Methods: We show that using prediction models for decision making can lead to harmful decisions, even when the predictions exhibit good discrimination after deployment. These models are harmful self-fulfilling prophecies: their deployment harms a group of patients but the worse outcome of these patients does not invalidate the predictive power of the model.   Results: Our main result is a formal characterization of a set of such prediction models. Next we show that models that are well calibrated before and after deployment 
    
[^3]: 我们错过了谁？一种基于原则的揭示少数人群特征的方法

    Who Are We Missing? A Principled Approach to Characterizing the Underrepresented Population. (arXiv:2401.14512v1 [stat.ME])

    [http://arxiv.org/abs/2401.14512](http://arxiv.org/abs/2401.14512)

    本文提出了一种基于优化的方法，Rashomon Set of Optimal Trees (ROOT)，用于识别和描述随机对照试验中的少数人群。该方法通过最小化目标平均处理效应估计的方差来优化目标子群体分布，从而提供更精确和可解释的处理效应估计。与其他方法相比，该方法具有更高的精度和可解释性，通过合成数据实验进行了验证。

    

    随机对照试验在理解因果效应方面起到了关键作用，然而将推论扩展到目标人群时面临效应异质性和代表性不足的挑战。我们的论文解决了在随机对照试验中识别和描述少数人群的关键问题，提出了一种改进目标人群以提升普适性的创新框架。我们引入了一种基于优化的方法——Rashomon Set of Optimal Trees (ROOT)，来描述少数人群。ROOT通过最小化目标平均处理效应估计的方差来优化目标子群体分布，从而确保更精确的处理效应估计。值得注意的是，ROOT生成可解释的少数人群特征，有助于研究人员有效沟通。我们的方法在精度和可解释性方面相对于其他方法展现了改进，通过合成数据实验进行了验证。

    Randomized controlled trials (RCTs) serve as the cornerstone for understanding causal effects, yet extending inferences to target populations presents challenges due to effect heterogeneity and underrepresentation. Our paper addresses the critical issue of identifying and characterizing underrepresented subgroups in RCTs, proposing a novel framework for refining target populations to improve generalizability. We introduce an optimization-based approach, Rashomon Set of Optimal Trees (ROOT), to characterize underrepresented groups. ROOT optimizes the target subpopulation distribution by minimizing the variance of the target average treatment effect estimate, ensuring more precise treatment effect estimations. Notably, ROOT generates interpretable characteristics of the underrepresented population, aiding researchers in effective communication. Our approach demonstrates improved precision and interpretability compared to alternatives, as illustrated with synthetic data experiments. We ap
    
[^4]: 具有局部线性模型的不变因果预测

    Invariant Causal Prediction with Locally Linear Models. (arXiv:2401.05218v1 [cs.LG])

    [http://arxiv.org/abs/2401.05218](http://arxiv.org/abs/2401.05218)

    本文扩展了ICP原则，考虑了在不同环境下具有局部线性模型的不变因果预测任务。通过提供因果父节点的可辨识性条件和引入LoLICaP方法，实现了在观察数据中识别目标变量的因果父节点。

    

    本文考虑通过观察数据，从一组候选变量中识别出目标变量的因果父节点的任务。我们的主要假设是候选变量在不同的环境中被观察到，这些环境可以对应于机器的不同设置或者动态过程中的不同时间间隔等。在一定的假设条件下，不同的环境可以被视为对观察系统的干预。我们假设目标变量和协变量之间存在线性关系，在每个环境下可能不同，但因果结构在不同环境中是不变的。这是Peters等人[2016]提出的ICP（不变因果预测）原则的扩展，后者假设所有环境下存在一个固定的线性关系。在我们提出的设置下，我们给出了因果父节点可辨识性的充分条件，并引入了一个名为LoLICaP的实用方法。

    We consider the task of identifying the causal parents of a target variable among a set of candidate variables from observational data. Our main assumption is that the candidate variables are observed in different environments which may, for example, correspond to different settings of a machine or different time intervals in a dynamical process. Under certain assumptions different environments can be regarded as interventions on the observed system. We assume a linear relationship between target and covariates, which can be different in each environment with the only restriction that the causal structure is invariant across environments. This is an extension of the ICP ($\textbf{I}$nvariant $\textbf{C}$ausal $\textbf{P}$rediction) principle by Peters et al. [2016], who assumed a fixed linear relationship across all environments. Within our proposed setting we provide sufficient conditions for identifiability of the causal parents and introduce a practical method called LoLICaP ($\text
    
[^5]: 关于细分实验效率的研究

    On the Efficiency of Finely Stratified Experiments. (arXiv:2307.15181v1 [econ.EM])

    [http://arxiv.org/abs/2307.15181](http://arxiv.org/abs/2307.15181)

    本文研究了在实验分析中涉及的大类处理效应参数的有效估计，包括平均处理效应、分位数处理效应、局部平均处理效应等。

    

    本文研究了在实验分析中涉及的大类处理效应参数的有效估计。在这里，效率是指对于一类广泛的处理分配方案而言的，其中任何单位被分配到处理的边际概率等于预先指定的值，例如一半。重要的是，我们不要求处理状态是以i.i.d.的方式分配的，因此可以适应实践中使用的复杂处理分配方案，如分层随机化和匹配对。所考虑的参数类别是可以表示为已知观测数据的一个已知函数的期望的约束的解的那些参数，其中可能包括处理分配边际概率的预先指定值。我们证明了这类参数包括平均处理效应、分位数处理效应、局部平均处理效应等。

    This paper studies the efficient estimation of a large class of treatment effect parameters that arise in the analysis of experiments. Here, efficiency is understood to be with respect to a broad class of treatment assignment schemes for which the marginal probability that any unit is assigned to treatment equals a pre-specified value, e.g., one half. Importantly, we do not require that treatment status is assigned in an i.i.d. fashion, thereby accommodating complicated treatment assignment schemes that are used in practice, such as stratified block randomization and matched pairs. The class of parameters considered are those that can be expressed as the solution to a restriction on the expectation of a known function of the observed data, including possibly the pre-specified value for the marginal probability of treatment assignment. We show that this class of parameters includes, among other things, average treatment effects, quantile treatment effects, local average treatment effect
    
[^6]: 神经网络在因果估计中的应用: 在美国评估更严格的空气质量标准的健康效益

    Causal Estimation of Exposure Shifts with Neural Networks: Evaluating the Health Benefits of Stricter Air Quality Standards in the US. (arXiv:2302.02560v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.02560](http://arxiv.org/abs/2302.02560)

    本研究提出了一种神经网络方法，利用其理论基础和实施的可行性，从而估计连续暴露/治疗的分布对政策相关结果的因果效应。我们将此方法应用于包含6800万个个体和2700万个美国境内死亡事件的数据中，通过评估美国国家环境保护局（EPA）对PM2.5的国家环境空气质量标准（NAAQS）进行修订后的健康效益。

    

    在政策研究中，估计连续性暴露/治疗的分布对感兴趣的结果的因果效应是最关键的分析任务之一。我们称之为偏移-响应函数（SRF）估计问题。现有的涉及强健因果效应估计器的神经网络方法缺乏理论保证和实际实现，用于SRF估计。受公共卫生中的关键政策问题的启发，我们开发了一种神经网络方法及其理论基础，以提供具有强健性和效率保证的SRF估计。然后，我们将我们的方法应用于包含6800万个个体和2700万个美国境内死亡事件的数据中，以估计将美国国家环境保护局（EPA）最近提议从12 μg/m³改为9 μg/m³的PM2.5的美国国家环境空气质量标准（NAAQS）的修订对结果的因果效应。我们的目标是首次估计

    In policy research, one of the most critical analytic tasks is to estimate the causal effect of a policy-relevant shift to the distribution of a continuous exposure/treatment on an outcome of interest. We call this problem shift-response function (SRF) estimation. Existing neural network methods involving robust causal-effect estimators lack theoretical guarantees and practical implementations for SRF estimation. Motivated by a key policy-relevant question in public health, we develop a neural network method and its theoretical underpinnings to estimate SRFs with robustness and efficiency guarantees. We then apply our method to data consisting of 68 million individuals and 27 million deaths across the U.S. to estimate the causal effect from revising the US National Ambient Air Quality Standards (NAAQS) for PM 2.5 from 12 $\mu g/m^3$ to 9 $\mu g/m^3$. This change has been recently proposed by the US Environmental Protection Agency (EPA). Our goal is to estimate, for the first time, the 
    

