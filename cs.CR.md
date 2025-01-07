# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Shifting the Lens: Detecting Malware in npm Ecosystem with Large Language Models](https://arxiv.org/abs/2403.12196) | 通过大型语言模型在npm生态系统中进行实证研究，以协助安全分析师识别恶意软件包 |
| [^2] | [Automated Security Response through Online Learning with Adaptive Conjectures](https://arxiv.org/abs/2402.12499) | 该论文通过自适应猜想的在线学习，提出了一种适用于IT基础设施的自动化安全响应方法，其中游戏参与者通过Bayesian学习调整猜想，并通过推演更新策略，最终实现了最佳拟合，提高了推演在猜想模型下的性能。 |
| [^3] | [Leverage Staking with Liquid Staking Derivatives (LSDs): Opportunities and Risks.](http://arxiv.org/abs/2401.08610) | 这项研究系统地研究了Liquid Staking Derivatives (LSDs)的杠杆质押机会与风险。他们发现杠杆质押在Lido-Aave生态系统中能够实现较高的回报，并有潜力通过优化策略获得更多收益。 |

# 详细

[^1]: 用大型语言模型在npm生态系统中检测恶意软件

    Shifting the Lens: Detecting Malware in npm Ecosystem with Large Language Models

    [https://arxiv.org/abs/2403.12196](https://arxiv.org/abs/2403.12196)

    通过大型语言模型在npm生态系统中进行实证研究，以协助安全分析师识别恶意软件包

    

    Gartner 2022年的报告预测，到2025年，全球45%的组织将遭遇软件供应链攻击，凸显了改善软件供应链安全对社区和国家利益的迫切性。当前的恶意软件检测技术通过过滤良性和恶意软件包来辅助手动审核过程，然而这种技术存在较高的误报率和有限的自动化支持。因此，恶意软件检测技术可以受益于先进、更自动化的方法，得到准确且误报较少的结果。该研究的目标是通过对大型语言模型（LLMs）进行实证研究，帮助安全分析师识别npm生态系统中的恶意软件。

    arXiv:2403.12196v1 Announce Type: cross  Abstract: The Gartner 2022 report predicts that 45% of organizations worldwide will encounter software supply chain attacks by 2025, highlighting the urgency to improve software supply chain security for community and national interests. Current malware detection techniques aid in the manual review process by filtering benign and malware packages, yet such techniques have high false-positive rates and limited automation support. Therefore, malware detection techniques could benefit from advanced, more automated approaches for accurate and minimally false-positive results. The goal of this study is to assist security analysts in identifying malicious packages through the empirical study of large language models (LLMs) to detect potential malware in the npm ecosystem.   We present SocketAI Scanner, a multi-stage decision-maker malware detection workflow using iterative self-refinement and zero-shot-role-play-Chain of Thought (CoT) prompting techni
    
[^2]: 通过自适应猜想的在线学习实现自动化安全响应

    Automated Security Response through Online Learning with Adaptive Conjectures

    [https://arxiv.org/abs/2402.12499](https://arxiv.org/abs/2402.12499)

    该论文通过自适应猜想的在线学习，提出了一种适用于IT基础设施的自动化安全响应方法，其中游戏参与者通过Bayesian学习调整猜想，并通过推演更新策略，最终实现了最佳拟合，提高了推演在猜想模型下的性能。

    

    我们研究了针对IT基础设施的自动化安全响应，并将攻击者和防御者之间的互动形式表述为一个部分观测、非平稳博弈。我们放宽了游戏模型正确规定的标准假设，并考虑每个参与者对模型有一个概率性猜想，可能在某种意义上错误规定，即真实模型的概率为0。这种形式允许我们捕捉关于基础设施和参与者意图的不确定性。为了在线学习有效的游戏策略，我们设计了一种新颖的方法，其中一个参与者通过贝叶斯学习迭代地调整其猜想，并通过推演更新其策略。我们证明了猜想会收敛到最佳拟合，并提供了在具有猜测模型的情况下推演实现性能改进的上限。为了刻画游戏的稳定状态，我们提出了Berk-Nash平衡的一个变种。

    arXiv:2402.12499v1 Announce Type: cross  Abstract: We study automated security response for an IT infrastructure and formulate the interaction between an attacker and a defender as a partially observed, non-stationary game. We relax the standard assumption that the game model is correctly specified and consider that each player has a probabilistic conjecture about the model, which may be misspecified in the sense that the true model has probability 0. This formulation allows us to capture uncertainty about the infrastructure and the intents of the players. To learn effective game strategies online, we design a novel method where a player iteratively adapts its conjecture using Bayesian learning and updates its strategy through rollout. We prove that the conjectures converge to best fits, and we provide a bound on the performance improvement that rollout enables with a conjectured model. To characterize the steady state of the game, we propose a variant of the Berk-Nash equilibrium. We 
    
[^3]: 使用Liquid Staking Derivatives (LSDs)进行杠杆质押: 机会与风险

    Leverage Staking with Liquid Staking Derivatives (LSDs): Opportunities and Risks. (arXiv:2401.08610v1 [q-fin.GN])

    [http://arxiv.org/abs/2401.08610](http://arxiv.org/abs/2401.08610)

    这项研究系统地研究了Liquid Staking Derivatives (LSDs)的杠杆质押机会与风险。他们发现杠杆质押在Lido-Aave生态系统中能够实现较高的回报，并有潜力通过优化策略获得更多收益。

    

    Lido是以太坊上最主要的Liquid Staking Derivative (LSD)提供商，允许用户抵押任意数量的ETH来获得stETH，这可以与DeFi协议如Aave进行整合。Lido与Aave之间的互通性使得一种新型策略“杠杆质押”得以实现，用户在Lido上质押ETH获取stETH，将stETH作为Aave上的抵押品借入ETH，然后将借入的ETH重新投入Lido。用户可以迭代执行此过程，根据自己的风险偏好来优化潜在回报。本文系统地研究了杠杆质押所涉及的机会和风险。我们是第一个在Lido-Aave生态系统中对杠杆质押策略进行形式化的研究。我们的经验研究发现，在以太坊上有262个杠杆质押头寸，总质押金额为295,243 ETH（482M USD）。我们发现，90.13%的杠杆质押头寸实现了比传统质押更高的回报。

    Lido, the leading Liquid Staking Derivative (LSD) provider on Ethereum, allows users to stake an arbitrary amount of ETH to receive stETH, which can be integrated with Decentralized Finance (DeFi) protocols such as Aave. The composability between Lido and Aave enables a novel strategy called "leverage staking", where users stake ETH on Lido to acquire stETH, utilize stETH as collateral on Aave to borrow ETH, and then restake the borrowed ETH on Lido. Users can iteratively execute this process to optimize potential returns based on their risk profile.  This paper systematically studies the opportunities and risks associated with leverage staking. We are the first to formalize the leverage staking strategy within the Lido-Aave ecosystem. Our empirical study identifies 262 leverage staking positions on Ethereum, with an aggregated staking amount of 295,243 ETH (482M USD). We discover that 90.13% of leverage staking positions have achieved higher returns than conventional staking. Furtherm
    

