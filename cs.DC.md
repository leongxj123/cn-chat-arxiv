# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Analyzing the Impact of Partial Sharing on the Resilience of Online Federated Learning Against Model Poisoning Attacks](https://arxiv.org/abs/2403.13108) | PSO-Fed算法的部分共享机制不仅可以降低通信负载，还能增强算法对模型投毒攻击的抵抗力，并且在面对拜占庭客户端的情况下依然能保持收敛。 |

# 详细

[^1]: 分析部分共享对在线联邦学习抵抗模型投毒攻击的影响

    Analyzing the Impact of Partial Sharing on the Resilience of Online Federated Learning Against Model Poisoning Attacks

    [https://arxiv.org/abs/2403.13108](https://arxiv.org/abs/2403.13108)

    PSO-Fed算法的部分共享机制不仅可以降低通信负载，还能增强算法对模型投毒攻击的抵抗力，并且在面对拜占庭客户端的情况下依然能保持收敛。

    

    我们审查了部分共享的在线联邦学习（PSO-Fed）算法对抵抗模型投毒攻击的韧性。 PSO-Fed通过使客户端在每个更新轮次仅与服务器交换部分模型估计来减少通信负载。模型估计的部分共享还增强了算法对模型投毒攻击的强度。为了更好地理解这一现象，我们分析了PSO-Fed算法在存在拜占庭客户端的情况下的性能，这些客户端可能会在与服务器共享之前通过添加噪声轻微篡改其本地模型。通过我们的分析，我们证明了PSO-Fed在均值和均方意义上都能保持收敛，即使在模型投毒攻击的压力下也是如此。我们进一步推导了PSO-Fed的理论均方误差（MSE），将其与步长、攻击概率、数字等各种参数联系起来。

    arXiv:2403.13108v1 Announce Type: new  Abstract: We scrutinize the resilience of the partial-sharing online federated learning (PSO-Fed) algorithm against model-poisoning attacks. PSO-Fed reduces the communication load by enabling clients to exchange only a fraction of their model estimates with the server at each update round. Partial sharing of model estimates also enhances the robustness of the algorithm against model-poisoning attacks. To gain better insights into this phenomenon, we analyze the performance of the PSO-Fed algorithm in the presence of Byzantine clients, malicious actors who may subtly tamper with their local models by adding noise before sharing them with the server. Through our analysis, we demonstrate that PSO-Fed maintains convergence in both mean and mean-square senses, even under the strain of model-poisoning attacks. We further derive the theoretical mean square error (MSE) of PSO-Fed, linking it to various parameters such as stepsize, attack probability, numb
    

