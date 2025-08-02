# Chapter 25: 用户行为建模

## 开篇段落

用户行为建模是现代搜索引擎理解和优化搜索体验的核心技术。通过分析用户的点击、停留时间、查询重写等交互信号，搜索系统能够推断用户的真实意图，纠正排序算法中的各种偏差，并提供个性化的搜索结果。本章将深入探讨从传统点击模型到现代神经行为预测的演进，介绍多维度偏差建模的架构设计，以及如何构建跨设备和会话感知的用户表示。我们还将探讨反事实学习在无偏排序优化中的应用，帮助你设计能够从有偏的用户反馈中学习无偏排序策略的系统。

## 章节大纲

### 1. 点击模型：从位置偏差到神经行为预测
- 1.1 传统点击模型架构
  - Position-Based Model (PBM)
  - Cascade Model (CM)
  - Dynamic Bayesian Network (DBN)
  - User Browsing Model (UBM)
- 1.2 点击预测的特征工程
  - 查询-文档相关性特征
  - 用户历史行为特征
  - 上下文特征设计
- 1.3 深度学习点击模型
  - Neural Click Model (NCM)
  - Transformer-based Click Prediction
  - 多任务学习架构
- 1.4 实时点击预测系统设计

### 2. 多偏差建模：展现、信任与查询意图偏差
- 2.1 偏差类型与建模策略
  - 位置偏差 (Position Bias)
  - 展现偏差 (Presentation Bias)
  - 选择偏差 (Selection Bias)
  - 信任偏差 (Trust Bias)
- 2.2 多偏差联合建模架构
  - Propensity Score Estimation
  - Inverse Propensity Scoring (IPS)
  - Doubly Robust Estimation
- 2.3 查询意图偏差处理
  - 导航型 vs 信息型查询
  - 意图感知的偏差校正
  - 多意图查询的建模挑战
- 2.4 偏差感知的训练策略

### 3. 跨设备与会话感知的用户建模
- 3.1 跨设备用户识别架构
  - 确定性匹配策略
  - 概率性用户链接
  - 隐私保护的设备关联
- 3.2 会话建模与表示学习
  - Session-based Recommendation
  - Hierarchical RNN架构
  - Self-Attention会话编码
- 3.3 长短期兴趣建模
  - 用户画像的增量更新
  - 兴趣漂移检测
  - 多粒度时间建模
- 3.4 跨设备行为融合策略

### 4. 反事实学习与无偏排序优化
- 4.1 因果推断框架
  - Potential Outcome Framework
  - 反事实推理在搜索中的应用
  - Treatment Effect估计
- 4.2 无偏学习排序算法
  - Unbiased Learning to Rank
  - Counterfactual Risk Minimization
  - 偏差-方差权衡
- 4.3 在线学习与探索策略
  - Contextual Bandits
  - Thompson Sampling
  - 探索-利用平衡
- 4.4 离线评估与在线实验设计

### 5. 本章小结

### 6. 练习题

### 7. 常见陷阱与错误 (Gotchas)

### 8. 最佳实践检查清单

## 1. 点击模型：从位置偏差到神经行为预测

用户点击行为是搜索引擎最直接的反馈信号，但这些信号往往包含各种偏差。理解和建模这些偏差对于构建有效的排序系统至关重要。

### 1.1 传统点击模型架构

#### Position-Based Model (PBM)
PBM 将点击概率分解为两个独立因素：文档的内在吸引力和位置的检查概率。

```ocaml
module type PositionBasedModel = sig
  type position = int
  type relevance = float
  
  val examination_probability : position -> float
  val attractiveness : doc_features -> query_features -> relevance
  val click_probability : position -> relevance -> float
end
```

关键设计考虑：
- 位置检查概率通常呈指数衰减
- 吸引力估计需要考虑查询-文档特征交互
- 参数学习使用 EM 算法或梯度下降

#### Cascade Model (CM)
级联模型假设用户从上到下顺序浏览结果，直到找到满意答案。

架构特点：
- 顺序依赖的点击概率计算
- 停止概率建模用户满意度
- 适合导航型查询的行为建模

#### Dynamic Bayesian Network (DBN)
DBN 通过隐变量建模用户的浏览和满意状态，捕捉更复杂的用户行为模式。

状态变量：
- Examination (E): 用户是否查看了结果
- Satisfaction (S): 用户是否满意
- Click (C): 观察到的点击行为

#### User Browsing Model (UBM)
UBM 扩展了级联模型，允许用户跳过某些结果后继续浏览。

```ocaml
module type UserBrowsingModel = sig
  type browsing_state = {
    examined_positions : int list;
    click_positions : int list;
    skip_probability : float;
  }
  
  val update_state : browsing_state -> position -> action -> browsing_state
  val predict_examination : browsing_state -> position -> float
end
```

### 1.2 点击预测的特征工程

有效的点击预测需要精心设计的特征体系：

#### 查询-文档相关性特征
- BM25 分数与变体
- 查询词在标题、正文、URL 中的匹配
- 语义相似度（Word2Vec, BERT embeddings）
- 查询意图与文档类型匹配度

#### 用户历史行为特征
- 用户对相似查询的历史点击率
- 用户对特定域名的偏好
- 个性化的停留时间模式
- 重复访问和书签行为

#### 上下文特征设计
- 时间因素（工作日/周末，具体时间）
- 设备类型和屏幕尺寸
- 地理位置相关性
- 查询会话的上下文

特征交互策略：
```ocaml
module type FeatureInteraction = sig
  type feature_vector
  
  val cartesian_product : feature_vector -> feature_vector -> feature_vector
  val factorization_machines : feature_vector list -> int -> feature_vector
  val deep_crossing : feature_vector list -> neural_config -> feature_vector
end
```

### 1.3 深度学习点击模型

#### Neural Click Model (NCM)
NCM 使用深度神经网络自动学习特征表示和交互：

架构组件：
- Embedding 层：将离散特征映射到连续空间
- Interaction 层：建模查询-文档交互
- Position encoding：注入位置信息
- Output 层：预测点击概率

```ocaml
module type NeuralClickModel = sig
  type embedding_config = {
    vocab_size : int;
    embedding_dim : int;
    pretrained_weights : weight_matrix option;
  }
  
  type model_architecture = {
    query_encoder : encoder_config;
    doc_encoder : encoder_config;
    interaction_layers : layer_config list;
    position_encoding : position_encoding_type;
  }
  
  val forward : model_architecture -> query -> document -> position -> click_probability
end
```

#### Transformer-based Click Prediction
利用自注意力机制捕捉结果列表中的相互影响：

关键创新：
- 多头注意力建模结果间关系
- 位置编码保留排序信息
- 跨结果的特征传播
- 端到端可微分架构

实现考虑：
- 计算复杂度 O(n²) 需要优化
- 局部注意力减少计算量
- 知识蒸馏加速推理

#### 多任务学习架构
同时优化多个用户行为指标：

任务设计：
- 主任务：点击预测
- 辅助任务：停留时间预测、跳出率预测
- 共享表示学习
- 任务特定的输出头

```ocaml
module type MultiTaskClickModel = sig
  type task_weight = float
  
  type task_config = {
    click_weight : task_weight;
    dwell_weight : task_weight;
    satisfaction_weight : task_weight;
  }
  
  val joint_loss : predictions -> ground_truth -> task_config -> float
  val gradient_normalization : gradient list -> normalized_gradient
end
```

### 1.4 实时点击预测系统设计

构建生产级的点击预测系统需要考虑：

#### 特征服务架构
- 低延迟特征获取（<10ms）
- 特征缓存策略
- 实时特征计算管道
- 特征版本管理

#### 模型服务优化
- 模型量化和剪枝
- 批处理预测请求
- GPU/TPU 推理加速
- 模型热更新机制

#### 在线学习管道
```ocaml
module type OnlineClickLearning = sig
  type model_state
  type update_batch
  
  val incremental_update : model_state -> update_batch -> model_state
  val exploration_strategy : model_state -> query -> document list -> document list
  val convergence_monitoring : model_state -> metrics
end
```

系统集成要点：
- A/B 测试框架集成
- 实时指标监控
- 降级策略设计
- 数据质量保障

## 2. 多偏差建模：展现、信任与查询意图偏差

搜索结果的用户反馈包含多种相互作用的偏差。准确识别和校正这些偏差是构建公平、有效排序系统的关键。

### 2.1 偏差类型与建模策略

#### 位置偏差 (Position Bias)
用户倾向于点击排在前面的结果，即使相关性较低。

建模策略：
- 检查概率随位置指数衰减
- 考虑不同设备的衰减模式差异
- 区分垂直结果（图片、视频）的位置效应

```ocaml
module type PositionBias = sig
  type device_type = Mobile | Desktop | Tablet
  type result_type = Web | Image | Video | News
  
  val examination_decay : position -> device_type -> float
  val vertical_position_effect : position -> result_type -> float
  val position_normalization : click_data -> normalized_clicks
end
```

#### 展现偏差 (Presentation Bias)
结果的视觉呈现影响用户行为：

影响因素：
- 标题长度和格式
- 摘要质量和高亮
- 富媒体元素（图片、评分、价格）
- 结果块的视觉突出度

建模考虑：
- 特征工程捕捉展现特征
- A/B 测试分离展现效果
- 多模态特征融合

#### 选择偏差 (Selection Bias)
搜索系统只能观察到被展示结果的反馈：

问题本质：
- Missing Not At Random (MNAR)
- 反事实：如果展示其他结果会怎样？
- 长尾查询的数据稀疏性

缓解策略：
- 探索性展示收集反馈
- 重要性采样校正
- 迁移学习利用相似查询

#### 信任偏差 (Trust Bias)
用户对特定来源或品牌的固有偏好：

```ocaml
module type TrustBias = sig
  type domain = string
  type user_trust_profile = {
    trusted_domains : (domain * float) list;
    domain_visit_history : domain visit_record list;
    trust_decay_factor : float;
  }
  
  val estimate_trust_score : user_trust_profile -> domain -> float
  val debiased_relevance : raw_click_rate -> trust_score -> adjusted_relevance
end
```

### 2.2 多偏差联合建模架构

#### Propensity Score Estimation
倾向分数估计点击的"可观察性"：

估计方法：
- 随机化实验收集无偏数据
- 双塔模型分离相关性和倾向性
- EM 算法迭代估计

```ocaml
module type PropensityEstimation = sig
  type propensity_model = {
    position_component : position -> float;
    presentation_component : presentation_features -> float;
    user_component : user_features -> float;
  }
  
  val estimate_propensity : propensity_model -> context -> float
  val randomization_strategy : query -> exploration_config -> ranking
end
```

#### Inverse Propensity Scoring (IPS)
使用倾向分数进行无偏估计：

核心公式：
- 无偏估计 = Σ(click × relevance) / propensity
- 方差控制通过 clipping 或正则化
- 自归一化 IPS 提高稳定性

实现挑战：
- 倾向分数接近零导致高方差
- 需要准确的倾向分数估计
- 在线系统的增量更新

#### Doubly Robust Estimation
结合直接估计和倾向分数校正：

```ocaml
module type DoublyRobust = sig
  type imputation_model
  type propensity_model
  
  val impute_missing_feedback : imputation_model -> unobserved_pairs -> predictions
  val doubly_robust_estimate : 
    observations -> imputation_model -> propensity_model -> unbiased_estimate
  val variance_reduction : estimate list -> reduced_variance_estimate
end
```

优势：
- 两个模型只要一个准确就能得到无偏估计
- 方差通常低于纯 IPS
- 适合部署在生产系统

### 2.3 查询意图偏差处理

#### 导航型 vs 信息型查询
不同查询意图导致不同的用户行为模式：

导航型查询特征：
- 高位置点击集中度
- 短停留时间
- 低结果多样性需求

信息型查询特征：
- 分散的点击分布
- 长停留时间和多次交互
- 高结果多样性价值

```ocaml
module type IntentAwareBias = sig
  type query_intent = Navigational | Informational | Transactional | Local
  
  val classify_intent : query -> query_features -> query_intent
  val intent_specific_bias_model : query_intent -> bias_parameters
  val adaptive_debiasing : query_intent -> click_data -> debiased_signal
end
```

#### 意图感知的偏差校正
根据查询意图调整偏差模型：

策略设计：
- 导航查询降低位置偏差权重
- 信息查询增强多样性信号
- 交易查询关注转化信号
- 本地查询结合地理偏好

#### 多意图查询的建模挑战
许多查询包含混合意图：

处理方法：
- 软意图分类（概率分布）
- 多任务学习框架
- 意图特定的子排序融合
- 动态意图识别基于用户交互

### 2.4 偏差感知的训练策略

#### 加权损失函数设计
```ocaml
module type BiasAwareLoss = sig
  type loss_config = {
    position_weight : position -> float;
    propensity_weight : float -> float;
    intent_weight : query_intent -> float;
  }
  
  val weighted_cross_entropy : predictions -> labels -> weights -> float
  val listwise_debiased_loss : ranking -> click_data -> propensities -> float
end
```

#### 多阶段训练流程
1. 预训练阶段：使用历史数据学习基础相关性
2. 偏差估计阶段：通过随机化实验估计偏差
3. 联合优化阶段：同时学习相关性和偏差校正
4. 在线微调阶段：持续适应用户行为变化

#### 评估指标设计
传统指标的偏差感知版本：
- Debiased CTR
- Position-normalized DCG
- Intent-aware MAP
- Propensity-weighted AUC

在线实验考虑：
- 长期用户满意度 vs 短期点击率
- 不同用户群体的公平性
- 查询覆盖率和多样性
- 系统整体健康度指标