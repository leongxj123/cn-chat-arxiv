# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SportsNGEN: Sustained Generation of Multi-player Sports Gameplay](https://arxiv.org/abs/2403.12977) | SportsNGEN是一种基于Transformer解码器的模型，经过训练能持续生成逼真的多人体育游戏，包括模拟整个网球比赛和为教练和广播员提供洞察力的能力。 |
| [^2] | [Probabilistic forecasting with Factor Quantile Regression: Application to electricity trading.](http://arxiv.org/abs/2303.08565) | 本文提出了一种新的概率预测方法，将分位数回归平均法（QRA）方法和主成分分析（PCA）平均方案相结合。在德国EPEX SPOT和波兰电力交易所（TGE）两个欧洲能源市场的数据集上进行了评估，证明该方法优于其他方法，能带来高达10欧元的利润。 |
| [^3] | [Electricity price forecasting with Smoothing Quantile Regression Averaging: Quantifying economic benefits of probabilistic forecasts.](http://arxiv.org/abs/2302.00411) | 本文介绍了一种名为平滑分位数回归平均（SQR Averaging）的新方法，用于准确预测复杂电力市场的电力价格，通过评估可靠性和锐度指标表明其性能优于现有的基准方法，并引入了一种评估方案来量化预测所带来的经济效益。 |

# 详细

[^1]: SportsNGEN: 持续生成多人体育游戏

    SportsNGEN: Sustained Generation of Multi-player Sports Gameplay

    [https://arxiv.org/abs/2403.12977](https://arxiv.org/abs/2403.12977)

    SportsNGEN是一种基于Transformer解码器的模型，经过训练能持续生成逼真的多人体育游戏，包括模拟整个网球比赛和为教练和广播员提供洞察力的能力。

    

    我们提出了一种基于Transformer解码器的模型SportsNGEN，该模型经过训练使用运动员和球追踪序列，能够生成逼真且持续的游戏场景。我们在大量专业网球追踪数据上训练和评估SportsNGEN，并展示通过将生成的模拟与射击分类器和逻辑相结合来开始和结束球赛，系统能够模拟整个网球比赛。此外，SportsNGEN的通用版本可以通过在包含该球员的比赛数据上微调来定制特定球员。我们展示了我们的模型经过良好校准，可以通过评估反事实或假设选项为教练和广播员提供洞察力。最后，我们展示了质量结果表明相同的方法适用于足球。

    arXiv:2403.12977v1 Announce Type: cross  Abstract: We present a transformer decoder based model, SportsNGEN, that is trained on sports player and ball tracking sequences that is capable of generating realistic and sustained gameplay. We train and evaluate SportsNGEN on a large database of professional tennis tracking data and demonstrate that by combining the generated simulations with a shot classifier and logic to start and end rallies, the system is capable of simulating an entire tennis match. In addition, a generic version of SportsNGEN can be customized to a specific player by fine-tuning on match data that includes that player. We show that our model is well calibrated and can be used to derive insights for coaches and broadcasters by evaluating counterfactual or what if options. Finally, we show qualitative results indicating the same approach works for football.
    
[^2]: 基于因子分位数回归的概率预测：应用于电力交易

    Probabilistic forecasting with Factor Quantile Regression: Application to electricity trading. (arXiv:2303.08565v1 [stat.AP])

    [http://arxiv.org/abs/2303.08565](http://arxiv.org/abs/2303.08565)

    本文提出了一种新的概率预测方法，将分位数回归平均法（QRA）方法和主成分分析（PCA）平均方案相结合。在德国EPEX SPOT和波兰电力交易所（TGE）两个欧洲能源市场的数据集上进行了评估，证明该方法优于其他方法，能带来高达10欧元的利润。

    

    本文提出了一种新的概率预测方法，将分位数回归平均法（QRA）方法和主成分分析（PCA）平均方案相结合。该方法在德国EPEX SPOT和波兰电力交易所（TGE）两个欧洲能源市场的数据集上进行了评估。结果表明，该方法比文献基准更准确。此外，实证证据表明，该方法在经验覆盖范围和Christoffersen测试方面优于其竞争对手。此外，根据财务指标评估了概率预测的经济价值。我们测试了考虑日前市场交易策略的预测模型的绩效，该策略利用概率价格预测和储能系统。结果表明，每1 MWh交易的利润高达10欧元。

    This paper presents a novel approach for constructing probabilistic forecasts, which combines both the Quantile Regression Averaging (QRA) method and the Principal Component Analysis (PCA) averaging scheme. The performance of the approach is evaluated on datasets from two European energy markets - the German EPEX SPOT and the Polish Power Exchange (TGE). The results indicate that newly proposed solutions yield results, which are more accurate than the literature benchmarks. Additionally, empirical evidence indicates that the proposed method outperforms its competitors in terms of the empirical coverage and the Christoffersen test. In addition, the economic value of the probabilistic forecast is evaluated on the basis of financial metrics. We test the performance of forecasting models taking into account a day-ahead market trading strategy that utilizes probabilistic price predictions and an energy storage system. The results indicate that profits of up to 10 EUR per 1 MWh transaction c
    
[^3]: 使用平滑分位数回归平均的电力价格预测：量化概率预测的经济效益

    Electricity price forecasting with Smoothing Quantile Regression Averaging: Quantifying economic benefits of probabilistic forecasts. (arXiv:2302.00411v2 [stat.AP] UPDATED)

    [http://arxiv.org/abs/2302.00411](http://arxiv.org/abs/2302.00411)

    本文介绍了一种名为平滑分位数回归平均（SQR Averaging）的新方法，用于准确预测复杂电力市场的电力价格，通过评估可靠性和锐度指标表明其性能优于现有的基准方法，并引入了一种评估方案来量化预测所带来的经济效益。

    

    在复杂的电力市场中，准确的电力价格预测对战略投标至关重要，影响着日常运营和长期投资。本文介绍了一种名为平滑分位数回归平均（SQR Averaging）的新方法，它改进了概率预测的高性能方案。为了展示其效用，我们在包括COVID-19疫情和俄罗斯对乌克兰的入侵的最新数据基础上，对四个电力市场进行了全面研究。通过可靠性和锐度指标，评估了SQR Averaging的性能，并将其与最先进的基准方法进行了比较。此外，还引入了一种评估方案，用于量化SQR Averaging预测所带来的经济效益。该方案可以应用于任何日前电力市场，并基于一种利用电池储存的交易策略，通过使用选定的预测分布分位数设定限价订单。

    In the world of the complex power market, accurate electricity price forecasting is essential for strategic bidding and affects both daily operations and long-term investments. This article introduce a new method dubbed Smoothing Quantile Regression (SQR) Averaging, that improves on well-performing schemes for probabilistic forecasting. To showcase its utility, a comprehensive study is conducted across four power markets, including recent data encompassing the COVID-19 pandemic and the Russian invasion on Ukraine. The performance of SQR Averaging is evaluated and compared to state-of-the-art benchmark methods in terms of the reliability and sharpness measures. Additionally, an evaluation scheme is introduced to quantify the economic benefits derived from SQR Averaging predictions. This scheme can be applied in any day-ahead electricity market and is based on a trading strategy that leverages battery storage and sets limit orders using selected quantiles of the predictive distribution. 
    

