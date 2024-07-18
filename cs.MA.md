# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Rethinking Integration of Prediction and Planning in Deep Learning-Based Automated Driving Systems: A Review.](http://arxiv.org/abs/2308.05731) | 这项综述重新思考了基于深度学习的自动驾驶系统中预测和规划的整合问题，提出了将其作为相互依赖的联合步骤来提高安全性、效率性和舒适性的必要性。 |
| [^2] | [Professional Basketball Player Behavior Synthesis via Planning with Diffusion.](http://arxiv.org/abs/2306.04090) | 本文中提出了一个名为PLAYBEST的方法，通过使用扩散概率模型从篮球比赛历史数据中学习策略，生成更加真实和多样化的球员行为，从而提高球员的决策制定效果，并在实验中得到了验证。 |

# 详细

[^1]: 重新思考基于深度学习的自动驾驶系统中的预测和规划的整合：一项综述

    Rethinking Integration of Prediction and Planning in Deep Learning-Based Automated Driving Systems: A Review. (arXiv:2308.05731v1 [cs.RO])

    [http://arxiv.org/abs/2308.05731](http://arxiv.org/abs/2308.05731)

    这项综述重新思考了基于深度学习的自动驾驶系统中预测和规划的整合问题，提出了将其作为相互依赖的联合步骤来提高安全性、效率性和舒适性的必要性。

    

    自动驾驶有可能彻底改变个人、公共和货运交通的方式。除了感知环境的巨大挑战外，即准确地使用可用的传感器数据感知环境，自动驾驶还包括规划一个安全、舒适和高效的运动轨迹。为了促进安全和进步，许多工作依赖于模块化的交通未来运动的预测。模块化的自动驾驶系统通常将预测和规划作为顺序的独立任务处理。虽然这考虑了周围交通对自车的影响，但它未能预测交通参与者对自车行为的反应。最近的研究表明，将预测和规划整合为相互依赖的联合步骤是实现安全、高效和舒适驾驶所必需的。虽然有各种模型实现了这种集成系统，但对不同原理的全面概述和理论理解仍然缺乏。

    Automated driving has the potential to revolutionize personal, public, and freight mobility. Besides the enormous challenge of perception, i.e. accurately perceiving the environment using available sensor data, automated driving comprises planning a safe, comfortable, and efficient motion trajectory. To promote safety and progress, many works rely on modules that predict the future motion of surrounding traffic. Modular automated driving systems commonly handle prediction and planning as sequential separate tasks. While this accounts for the influence of surrounding traffic on the ego-vehicle, it fails to anticipate the reactions of traffic participants to the ego-vehicle's behavior. Recent works suggest that integrating prediction and planning in an interdependent joint step is necessary to achieve safe, efficient, and comfortable driving. While various models implement such integrated systems, a comprehensive overview and theoretical understanding of different principles are lacking.
    
[^2]: 通过扩散规划进行职业篮球运动员行为合成

    Professional Basketball Player Behavior Synthesis via Planning with Diffusion. (arXiv:2306.04090v1 [cs.AI])

    [http://arxiv.org/abs/2306.04090](http://arxiv.org/abs/2306.04090)

    本文中提出了一个名为PLAYBEST的方法，通过使用扩散概率模型从篮球比赛历史数据中学习策略，生成更加真实和多样化的球员行为，从而提高球员的决策制定效果，并在实验中得到了验证。

    

    在多智能体系统中动态规划已经被应用于改善各种领域的决策制定。职业篮球作为一个包含隐蔽性战略策略和决策制定的动态时空博弈的引人注目的例子。然而，处理多样的场上信号和导航潜在动作和结果的广阔空间使得现有方法很难迅速识别响应不断变化的情况下的最佳策略。在本研究中，我们首先将序列决策制定过程定义为条件轨迹生成过程。我们进一步引入PLAYBEST（PLAYER BEhavior SynThesis），这是一种提高球员决策制定的方法。我们扩展了最先进的生成模型——扩散概率模型，以从历史的美国职业篮球联赛(NBA)球员运动跟踪数据中学习具有挑战性的多智能体环境动态。为了融合数据驱动的策略，我们引入了一个辅助变量到PLAYBEST中，以适应外部输入并生成真实和多样化的球员轨迹。实验结果表明，PLAYBEST可以生成高质量的球员行为，并在各种评估场景中优于基线模型。

    Dynamically planning in multi-agent systems has been explored to improve decision-making in various domains. Professional basketball serves as a compelling example of a dynamic spatio-temporal game, encompassing both concealed strategic policies and decision-making. However, processing the diverse on-court signals and navigating the vast space of potential actions and outcomes makes it difficult for existing approaches to swiftly identify optimal strategies in response to evolving circumstances. In this study, we first formulate the sequential decision-making process as a conditional trajectory generation process. We further introduce PLAYBEST (PLAYer BEhavior SynThesis), a method for enhancing player decision-making. We extend the state-of-the-art generative model, diffusion probabilistic model, to learn challenging multi-agent environmental dynamics from historical National Basketball Association (NBA) player motion tracking data. To incorporate data-driven strategies, an auxiliary v
    

