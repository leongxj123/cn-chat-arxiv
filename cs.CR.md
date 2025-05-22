# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Adversarial ModSecurity: Countering Adversarial SQL Injections with Robust Machine Learning.](http://arxiv.org/abs/2308.04964) | 这篇论文介绍了Adversarial ModSecurity，它是一个使用强大的机器学习来对抗SQL注入攻击的防火墙。通过将核心规则集作为输入特征，该模型可以识别并防御对抗性SQL注入攻击。实验结果表明，AdvModSec在训练后能够有效地应对这类攻击。 |

# 详细

[^1]: Adversarial ModSecurity: 使用强大的机器学习对抗SQL注入攻击

    Adversarial ModSecurity: Countering Adversarial SQL Injections with Robust Machine Learning. (arXiv:2308.04964v1 [cs.LG])

    [http://arxiv.org/abs/2308.04964](http://arxiv.org/abs/2308.04964)

    这篇论文介绍了Adversarial ModSecurity，它是一个使用强大的机器学习来对抗SQL注入攻击的防火墙。通过将核心规则集作为输入特征，该模型可以识别并防御对抗性SQL注入攻击。实验结果表明，AdvModSec在训练后能够有效地应对这类攻击。

    

    ModSecurity被广泛认可为标准的开源Web应用防火墙(WAF)，由OWASP基金会维护。它通过与核心规则集进行匹配来检测恶意请求，识别出常见的攻击模式。每个规则在CRS中都被手动分配一个权重，基于相应攻击的严重程度，如果触发规则的权重之和超过给定的阈值，就会被检测为恶意请求。然而，我们的研究表明，这种简单的策略在检测SQL注入攻击方面很不有效，因为它往往会阻止许多合法请求，同时还容易受到对抗性SQL注入攻击的影响，即故意操纵以逃避检测的攻击。为了克服这些问题，我们设计了一个名为AdvModSec的强大机器学习模型，它将CRS规则作为输入特征，并经过训练以检测对抗性SQL注入攻击。我们的实验表明，AdvModSec在针对该攻击的流量上进行训练后表现出色。

    ModSecurity is widely recognized as the standard open-source Web Application Firewall (WAF), maintained by the OWASP Foundation. It detects malicious requests by matching them against the Core Rule Set, identifying well-known attack patterns. Each rule in the CRS is manually assigned a weight, based on the severity of the corresponding attack, and a request is detected as malicious if the sum of the weights of the firing rules exceeds a given threshold. In this work, we show that this simple strategy is largely ineffective for detecting SQL injection (SQLi) attacks, as it tends to block many legitimate requests, while also being vulnerable to adversarial SQLi attacks, i.e., attacks intentionally manipulated to evade detection. To overcome these issues, we design a robust machine learning model, named AdvModSec, which uses the CRS rules as input features, and it is trained to detect adversarial SQLi attacks. Our experiments show that AdvModSec, being trained on the traffic directed towa
    

