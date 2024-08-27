# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [UAMM: UBET Automated Market Maker.](http://arxiv.org/abs/2308.06375) | UAMM是一种新的自动市场做市商方法，通过考虑外部市场价格和流动性池的暂时损失来定价，并且有效消除了套利机会。 |

# 详细

[^1]: UAMM: UBET自动市场做市商

    UAMM: UBET Automated Market Maker. (arXiv:2308.06375v1 [cs.LG])

    [http://arxiv.org/abs/2308.06375](http://arxiv.org/abs/2308.06375)

    UAMM是一种新的自动市场做市商方法，通过考虑外部市场价格和流动性池的暂时损失来定价，并且有效消除了套利机会。

    

    自动市场做市商（AMM）是去中心化交易所（DEX）使用的定价机制。传统的AMM方法仅基于其自身的流动性池进行定价，而不考虑外部市场或流动性提供者的风险管理。在本文中，我们提出了一种称为UBET AMM（UAMM）的新方法，通过考虑外部市场价格和流动性池的暂时损失来计算价格。尽管依赖于外部市场价格，我们的方法在计算滑点时仍然保持了恒定产品曲线的期望属性。UAMM的关键要素是根据期望的目标余额确定合适的滑点金额，以鼓励流动性池最小化暂时损失。我们证明了当外部市场价格有效时，我们的方法消除了套利机会。

    Automated market makers (AMMs) are pricing mechanisms utilized by decentralized exchanges (DEX). Traditional AMM approaches are constrained by pricing solely based on their own liquidity pool, without consideration of external markets or risk management for liquidity providers. In this paper, we propose a new approach known as UBET AMM (UAMM), which calculates prices by considering external market prices and the impermanent loss of the liquidity pool. Despite relying on external market prices, our method maintains the desired properties of a constant product curve when computing slippages. The key element of UAMM is determining the appropriate slippage amount based on the desired target balance, which encourages the liquidity pool to minimize impermanent loss. We demonstrate that our approach eliminates arbitrage opportunities when external market prices are efficient.
    

