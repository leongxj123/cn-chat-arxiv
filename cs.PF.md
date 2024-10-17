# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [cedar: Composable and Optimized Machine Learning Input Data Pipelines.](http://arxiv.org/abs/2401.08895) | cedar是一个编程模型和框架，可以轻松构建、优化和执行机器学习输入数据管道。它提供了易于使用的编程接口和可组合运算符，支持任意ML框架和库。通过解决当前输入数据系统无法充分利用性能优化的问题，cedar提高了资源利用效率，满足了庞大数据量和高训练吞吐量的需求。 |

# 详细

[^1]: cedar：可组合和优化的机器学习输入数据管道

    cedar: Composable and Optimized Machine Learning Input Data Pipelines. (arXiv:2401.08895v1 [cs.LG])

    [http://arxiv.org/abs/2401.08895](http://arxiv.org/abs/2401.08895)

    cedar是一个编程模型和框架，可以轻松构建、优化和执行机器学习输入数据管道。它提供了易于使用的编程接口和可组合运算符，支持任意ML框架和库。通过解决当前输入数据系统无法充分利用性能优化的问题，cedar提高了资源利用效率，满足了庞大数据量和高训练吞吐量的需求。

    

    输入数据管道是每个机器学习（ML）训练任务的重要组成部分。它负责读取大量的训练数据，使用复杂的变换处理样本批次，并以低延迟和高吞吐量将其加载到训练节点上。高性能的输入数据系统变得越来越关键，原因是数据量急剧增加和训练吞吐量的要求。然而，当前的输入数据系统无法充分利用关键的性能优化，导致资源利用效率极低的基础设施，或者更糟糕地，浪费昂贵的加速器。为了满足这些需求，我们提出了cedar，一个编程模型和框架，允许用户轻松构建、优化和执行输入数据管道。cedar提供了易于使用的编程接口，允许用户使用可组合运算符来定义支持任意ML框架和库的输入数据管道。

    The input data pipeline is an essential component of each machine learning (ML) training job. It is responsible for reading massive amounts of training data, processing batches of samples using complex of transformations, and loading them onto training nodes at low latency and high throughput. Performant input data systems are becoming increasingly critical, driven by skyrocketing data volumes and training throughput demands. Unfortunately, current input data systems cannot fully leverage key performance optimizations, resulting in hugely inefficient infrastructures that require significant resources -- or worse -- underutilize expensive accelerators.  To address these demands, we present cedar, a programming model and framework that allows users to easily build, optimize, and execute input data pipelines. cedar presents an easy-to-use programming interface, allowing users to define input data pipelines using composable operators that support arbitrary ML frameworks and libraries. Mean
    

