# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Interoperable synthetic health data with SyntHIR to enable the development of CDSS tools.](http://arxiv.org/abs/2308.02613) | 本论文提出了一种利用合成EHR数据开发CDSS工具的体系架构，通过使用SyntHIR系统和FHIR标准实现数据互操作性和工具可迁移性。 |

# 详细

[^1]: 用SyntHIR实现互操作性合成健康数据，以便开发CDSS工具

    Interoperable synthetic health data with SyntHIR to enable the development of CDSS tools. (arXiv:2308.02613v1 [cs.LG])

    [http://arxiv.org/abs/2308.02613](http://arxiv.org/abs/2308.02613)

    本论文提出了一种利用合成EHR数据开发CDSS工具的体系架构，通过使用SyntHIR系统和FHIR标准实现数据互操作性和工具可迁移性。

    

    利用高质量的患者日志和健康登记来开发基于机器学习的临床决策支持系统（CDSS）有很大的机会。为了在临床工作流程中实施CDSS工具，需要将该工具集成、验证和测试在用于存储和管理患者数据的电子健康记录（EHR）系统上。然而，由于合规法规，通常不可能获得对EHR系统的必要访问权限。我们提出了一种用于生成和使用CDSS工具开发的合成EHR数据的体系架构。该体系结构在一个称为SyntHIR的系统中实现。SyntHIR系统使用Fast Healthcare Interoperability Resources (FHIR)标准进行数据互操作性，使用Gretel框架生成合成数据，使用Microsoft Azure FHIR服务器作为基于FHIR的EHR系统，以及使用SMART on FHIR框架进行工具可迁移性。我们通过使用数据开发机器学习基于CDSS工具来展示SyntHIR的实用性。

    There is a great opportunity to use high-quality patient journals and health registers to develop machine learning-based Clinical Decision Support Systems (CDSS). To implement a CDSS tool in a clinical workflow, there is a need to integrate, validate and test this tool on the Electronic Health Record (EHR) systems used to store and manage patient data. However, it is often not possible to get the necessary access to an EHR system due to legal compliance. We propose an architecture for generating and using synthetic EHR data for CDSS tool development. The architecture is implemented in a system called SyntHIR. The SyntHIR system uses the Fast Healthcare Interoperability Resources (FHIR) standards for data interoperability, the Gretel framework for generating synthetic data, the Microsoft Azure FHIR server as the FHIR-based EHR system and SMART on FHIR framework for tool transportability. We demonstrate the usefulness of SyntHIR by developing a machine learning-based CDSS tool using data
    

