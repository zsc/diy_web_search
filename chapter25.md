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

现代用户在多个设备间切换搜索，其行为模式呈现复杂的时空特征。构建统一的用户表示需要解决设备识别、会话理解和兴趣演化等挑战。

### 3.1 跨设备用户识别架构

跨设备用户识别是理解用户完整搜索旅程的基础。这涉及在保护隐私的前提下，准确关联来自不同设备的用户行为。

#### 确定性匹配策略
基于明确信号的用户关联：

```ocaml
module type DeterministicMatching = sig
  type user_id = string
  type device_id = string
  type matching_signal = 
    | LoginID of user_id
    | EmailHash of string
    | PhoneHash of string
    | CrossDeviceToken of string
  
  val match_devices : matching_signal list -> device_id list -> user_id option
  val confidence_score : matching_signal -> float
  val signal_freshness : matching_signal -> timestamp -> float
end
```

匹配信号优先级：
- 登录账号（最可靠）
- 加密的邮箱/电话（中等可靠）
- 跨设备同步令牌（需验证）
- 行为模式相似度（辅助信号）

#### 概率性用户链接
当缺乏确定性信号时，使用概率模型推断设备关联：

行为特征提取：
- 搜索查询模式（主题分布、时间模式）
- 点击偏好（域名、内容类型）
- 地理位置轨迹（隐私安全处理后）
- 设备切换时间窗口

```ocaml
module type ProbabilisticLinking = sig
  type behavior_embedding = float array
  type device_pair = device_id * device_id
  
  val extract_behavior_features : device_history -> behavior_embedding
  val similarity_computation : behavior_embedding -> behavior_embedding -> float
  val graph_clustering : (device_pair * float) list -> device_cluster list
  val merge_threshold : float  (* 通常设为 0.8-0.9 *)
end
```

概率推断架构：
- 设备相似度计算（余弦相似度、编辑距离）
- 图聚类算法（Connected Components, Louvain）
- 时间衰减因子（近期行为权重更高）
- 不确定性量化（输出概率而非二元决策）

#### 隐私保护的设备关联
隐私合规是跨设备追踪的核心约束：

技术方案：
- 差分隐私添加噪声
- 联邦学习避免原始数据传输
- 同态加密计算相似度
- k-匿名性保证

```ocaml
module type PrivacyPreservingLinking = sig
  type privacy_budget = float
  type encrypted_features
  
  val add_laplace_noise : features -> privacy_budget -> noisy_features
  val federated_similarity : local_model list -> global_similarity_model
  val homomorphic_distance : encrypted_features -> encrypted_features -> encrypted_distance
  val k_anonymity_check : user_cluster -> int -> bool
end
```

实施要点：
- GDPR/CCPA 合规性审查
- 用户明确同意机制
- 数据最小化原则
- 定期数据清理策略

### 3.2 会话建模与表示学习

会话是用户搜索意图的自然单元，准确理解会话内的行为序列对个性化至关重要。

#### Session-based Recommendation
会话推荐系统的核心是捕捉短期行为模式：

会话定义策略：
- 时间间隔分割（如30分钟无活动）
- 查询相似度分割（主题突变）
- 混合分割（时间+语义）

```ocaml
module type SessionModeling = sig
  type session = {
    session_id : string;
    queries : query list;
    clicks : (query * document * timestamp) list;
    duration : time_span;
    device_type : device_type;
  }
  
  val segment_sessions : user_log -> session list
  val extract_session_intent : session -> intent_distribution
  val session_embedding : session -> embedding
end
```

#### Hierarchical RNN架构
层次化建模捕捉会话内和会话间的依赖：

架构设计：
- 底层 RNN：编码会话内查询序列
- 顶层 RNN：建模会话序列演化
- 注意力机制：识别关键查询和转折点

```ocaml
module type HierarchicalRNN = sig
  type intra_session_state
  type inter_session_state
  
  val encode_query_sequence : query list -> intra_session_state
  val encode_session_sequence : session list -> inter_session_state
  val attention_weights : state -> query -> float array
  val predict_next_query : inter_session_state -> query_distribution
end
```

实现优化：
- LSTM/GRU 缓解梯度问题
- 截断反向传播控制内存
- 批处理变长序列
- 模型压缩部署移动端

#### Self-Attention会话编码
Transformer 架构在会话建模中的应用：

优势：
- 并行计算效率高
- 长距离依赖建模
- 位置编码保留时序信息
- 多头注意力捕捉不同关系

```ocaml
module type SessionTransformer = sig
  type attention_config = {
    num_heads : int;
    hidden_dim : int;
    position_encoding : position_encoding_type;
    max_sequence_length : int;
  }
  
  val self_attention : query_sequence -> attention_config -> attended_representation
  val cross_session_attention : session list -> global_context
  val masked_prediction : partial_session -> query_distribution
end
```

架构变体：
- BERT-style：双向编码理解上下文
- GPT-style：自回归生成预测下一查询
- Hybrid：结合双向理解和单向预测

### 3.3 长短期兴趣建模

用户兴趣具有多时间尺度特征，需要区分稳定偏好和临时需求。

#### 用户画像的增量更新
实时维护用户兴趣表示：

```ocaml
module type UserProfileUpdate = sig
  type interest_profile = {
    short_term : weighted_topics;  (* 最近1天 *)
    medium_term : weighted_topics; (* 最近1周 *)
    long_term : weighted_topics;   (* 历史累积 *)
  }
  
  val incremental_update : interest_profile -> new_interaction -> interest_profile
  val decay_function : timestamp -> timestamp -> float
  val merge_profiles : interest_profile list -> interest_profile
end
```

更新策略：
- 指数衰减降低历史权重
- 主题层次化聚合
- 异常检测过滤噪声
- 定期重算防止漂移

#### 兴趣漂移检测
识别用户兴趣的显著变化：

检测方法：
- 分布距离度量（KL散度、JS散度）
- 变点检测算法（CUSUM、贝叶斯方法）
- 时间序列异常检测
- 主题演化追踪

```ocaml
module type InterestDrift = sig
  type drift_score = float
  type change_point = timestamp * topic_change
  
  val compute_topic_distance : topic_dist -> topic_dist -> drift_score
  val detect_change_points : topic_sequence -> change_point list
  val classify_drift_type : change_point -> DriftType.t
  val adapt_recommendation : drift_score -> recommendation_strategy
end
```

应用场景：
- 生活事件检测（搬家、换工作）
- 季节性兴趣变化
- 探索新领域识别
- 推荐策略自适应

#### 多粒度时间建模
不同时间尺度的兴趣表示：

```ocaml
module type MultiGranularityModeling = sig
  type time_granularity = Hour | Day | Week | Month | Year
  
  type temporal_profile = {
    hourly_pattern : activity_distribution;    (* 日内模式 *)
    weekly_pattern : day_preference array;     (* 周模式 *)
    seasonal_trend : season_interests;         (* 季节趋势 *)
    life_stage : life_stage_interests;         (* 生命阶段 *)
  }
  
  val extract_temporal_patterns : user_history -> temporal_profile
  val time_aware_scoring : temporal_profile -> timestamp -> score_modifier
end
```

时间特征工程：
- 周期性模式提取（傅里叶变换）
- 节假日效应建模
- 工作日/周末区分
- 时区感知处理

### 3.4 跨设备行为融合策略

整合多设备数据构建统一用户视图：

#### 设备特征对齐
不同设备的行为特征标准化：

```ocaml
module type DeviceAlignment = sig
  type device_features = {
    screen_size : dimensions;
    input_method : InputType.t;
    network_speed : bandwidth;
    typical_session_length : duration;
  }
  
  val normalize_click_behavior : device_features -> raw_clicks -> normalized_clicks
  val adjust_dwell_time : device_features -> raw_dwell -> adjusted_dwell
  val calibrate_scroll_depth : device_features -> scroll_data -> normalized_scroll
end
```

对齐策略：
- 移动设备点击率通常更高
- 桌面设备停留时间更长
- 平板介于两者之间
- 语音设备交互模式特殊

#### 多设备会话拼接
识别跨设备的连续搜索任务：

拼接信号：
- 时间邻近性（设备切换时间窗口）
- 查询相似性（编辑距离、语义相似）
- 结果重叠度（共同点击文档）
- 地理位置连续性

```ocaml
module type CrossDeviceSession = sig
  type device_session = device_id * session
  type unified_session = {
    sessions : device_session list;
    transition_points : (timestamp * device_id * device_id) list;
    task_completion : completion_status;
  }
  
  val stitch_sessions : device_session list -> unified_session list
  val compute_transition_probability : session -> session -> float
  val extract_cross_device_patterns : unified_session list -> pattern list
end
```

#### 统一用户表示学习
融合多设备数据的深度学习架构：

```ocaml
module type UnifiedUserEmbedding = sig
  type device_embedding = embedding
  type user_embedding = embedding
  
  type fusion_architecture = 
    | Concatenation
    | Attention_Pooling
    | Graph_Neural_Network
    | Hierarchical_Fusion
  
  val encode_device_behavior : device_history -> device_embedding
  val fuse_device_embeddings : device_embedding list -> fusion_architecture -> user_embedding
  val contrastive_learning : positive_pairs -> negative_pairs -> loss
end
```

融合策略比较：
- 简单拼接：快速但忽略设备间关系
- 注意力池化：自适应权重组合
- 图神经网络：建模设备交互拓扑
- 层次化融合：保留设备特定和共享特征

训练目标：
- 设备间行为一致性
- 任务完成度预测
- 下一设备预测
- 跨设备查询推荐

## 4. 反事实学习与无偏排序优化

搜索系统的核心挑战是从有偏的用户反馈中学习无偏的排序策略。反事实学习提供了一个原则性的框架，使我们能够回答"如果展示了不同的结果会怎样"这一关键问题。

### 4.1 因果推断框架

在搜索场景中应用因果推断需要仔细定义处理（treatment）、结果（outcome）和混杂因素（confounders）。

#### Potential Outcome Framework
潜在结果框架是理解反事实推理的基础：

```ocaml
module type PotentialOutcome = sig
  type treatment = ranking
  type outcome = click list
  type unit = query * user_context
  
  (* Y(1) - 展示排序1时的潜在点击 *)
  (* Y(0) - 展示排序0时的潜在点击 *)
  type potential_outcomes = {
    factual : outcome;      (* 观察到的结果 *)
    counterfactual : outcome option;  (* 未观察到的结果 *)
  }
  
  val causal_effect : potential_outcomes -> treatment_effect
  val fundamental_problem : unit -> (* 只能观察一个结果 *)
end
```

核心假设：
- SUTVA（稳定单元处理值假设）：用户之间无干扰
- 一致性：相同处理产生相同结果
- 可忽略性：给定特征，处理分配独立于潜在结果

#### 反事实推理在搜索中的应用
搜索场景的特殊性：

处理定义：
- 文档的排序位置
- 是否展示某个结果
- 结果的呈现方式

结果定义：
- 点击行为
- 停留时间
- 查询重写
- 会话结束

```ocaml
module type SearchCausality = sig
  type search_treatment = {
    ranking : document list;
    presentation : display_format;
    position : int;
  }
  
  type search_outcome = {
    clicks : click list;
    dwell_time : duration;
    satisfaction : satisfaction_signal option;
  }
  
  val estimate_position_effect : document -> position -> position -> effect_size
  val presentation_effect : display_format -> display_format -> effect_size
end
```

混杂因素识别：
- 查询意图（导航/信息/交易）
- 用户历史偏好
- 时间上下文
- 设备类型

#### Treatment Effect估计
估计展示不同结果的因果效应：

平均处理效应（ATE）：
```ocaml
module type TreatmentEffect = sig
  type ate = float  (* E[Y(1) - Y(0)] *)
  type cate = features -> float  (* Conditional ATE *)
  
  val estimate_ate : population -> treatment -> control -> ate
  val heterogeneous_effects : subgroups -> treatment -> cate
  val individual_treatment_effect : unit -> treatment -> ite
end
```

估计方法：
- 匹配方法（Propensity Score Matching）
- 加权方法（IPW, AIPW）
- 回归方法（Meta-learners）
- 机器学习方法（Causal Forests）

实践考虑：
- 协变量平衡检查
- 敏感性分析
- 置信区间计算
- 多重检验校正

### 4.2 无偏学习排序算法

从有偏的点击数据中学习无偏的排序函数是现代搜索引擎的核心技术。

#### Unbiased Learning to Rank
无偏 LTR 的核心思想是校正观察偏差：

```ocaml
module type UnbiasedLTR = sig
  type biased_feedback = (query * document * click * position) list
  type relevance_model
  type bias_model
  
  val joint_optimization : biased_feedback -> (relevance_model * bias_model)
  val debiased_loss : predictions -> labels -> propensities -> float
  val variance_regularization : float -> regularized_loss
end
```

关键算法：
- Dual Learning Algorithm：联合学习相关性和偏差
- EM-based Methods：交替优化两个模型
- Regression-based Approaches：直接回归建模

实现细节：
- 梯度裁剪防止数值不稳定
- 批归一化稳定训练
- 早停防止过拟合
- 正则化控制模型复杂度

#### Counterfactual Risk Minimization
最小化反事实风险的原则性方法：

理论基础：
```ocaml
module type CounterfactualRisk = sig
  type policy = context -> action
  type logging_policy = policy
  type target_policy = policy
  
  (* 反事实风险 = E_π[loss] *)
  val counterfactual_risk : target_policy -> logging_policy -> data -> risk
  val importance_sampling : target_policy -> logging_policy -> weight
  val clipped_importance_sampling : weight -> clip_threshold -> clipped_weight
end
```

优化目标：
- 经验风险最小化（ERM）的反事实版本
- 方差正则化项
- 策略约束（避免极端分布偏移）

算法变体：
- POEM（Policy Optimizer for Exponential Models）
- BanditNet（深度学习版本）
- SNIPS（自归一化重要性采样）

#### 偏差-方差权衡
无偏估计往往具有高方差，需要仔细权衡：

```ocaml
module type BiasVarianceTradeoff = sig
  type estimator
  type hyperparameter = float
  
  val bias : estimator -> ground_truth -> float
  val variance : estimator -> float
  val mse : estimator -> ground_truth -> float  (* bias² + variance *)
  
  val regularized_estimator : hyperparameter -> estimator
  val cross_validation : data -> hyperparameter list -> optimal_hyperparameter
end
```

权衡策略：
- Shrinkage：向有偏但低方差估计收缩
- Clipping：限制重要性权重范围
- Doubly Robust：结合多个估计器
- Bootstrap：通过重采样估计不确定性

实践指南：
- 小数据集倾向保守（接受一定偏差）
- 大数据集可追求更低偏差
- 在线系统需要稳定性
- 定期离线评估真实性能

### 4.3 在线学习与探索策略

在线环境需要平衡探索新排序和利用已知最佳排序。

#### Contextual Bandits
上下文赌博机框架自然适合搜索排序：

```ocaml
module type ContextualBandit = sig
  type context = query_features * user_features
  type arm = document
  type reward = click_signal
  
  type policy = context -> arm list -> probability_distribution
  
  val thompson_sampling : posterior -> policy
  val ucb_style : confidence_bound -> policy  
  val epsilon_greedy : epsilon -> random_policy -> greedy_policy -> policy
end
```

搜索特定挑战：
- 组合动作空间（排序而非单个选择）
- 延迟反馈（满意度信号）
- 非平稳环境（用户偏好演化）
- 多目标优化（相关性、多样性、新颖性）

#### Thompson Sampling
贝叶斯方法优雅处理探索-利用权衡：

实现架构：
```ocaml
module type ThompsonSampling = sig
  type prior_distribution
  type posterior_distribution
  
  val update_posterior : 
    posterior_distribution -> observation -> posterior_distribution
  val sample_parameters : posterior_distribution -> model_parameters
  val compute_ranking : model_parameters -> context -> ranking
  
  (* 实践优化 *)
  val batch_update : posterior_distribution -> observation list -> posterior_distribution
  val approximate_posterior : exact_posterior -> approximation_method -> approximate_posterior
end
```

后验近似方法：
- 共轭先验（Beta-Bernoulli）
- Laplace 近似
- 变分推断
- MCMC 采样

实践优化：
- 批量更新减少计算
- 分布式后验维护
- 在线-离线混合更新
- 冷启动先验设计

#### 探索-利用平衡
不同阶段需要不同的探索策略：

```ocaml
module type ExplorationSchedule = sig
  type exploration_rate = float
  type time_step = int
  
  val constant_exploration : exploration_rate
  val decay_schedule : initial_rate -> decay_factor -> time_step -> exploration_rate
  val adaptive_exploration : performance_history -> exploration_rate
  val contextual_exploration : query_type -> user_segment -> exploration_rate
end
```

探索策略设计：
- 新用户：更多探索了解偏好
- 老用户：更多利用提供稳定体验
- 长尾查询：探索发现相关文档
- 头部查询：利用已验证排序

自适应机制：
- 根据预测不确定性调整
- 根据用户反馈质量调整
- 根据业务指标调整
- 根据竞争环境调整

### 4.4 离线评估与在线实验设计

可靠的评估体系是反事实学习成功的关键。

#### 离线评估方法
使用历史数据评估新策略：

```ocaml
module type OfflineEvaluation = sig
  type logged_data = (context * action * reward * logging_probability) list
  type evaluation_policy
  
  val ips_estimator : evaluation_policy -> logged_data -> estimated_performance
  val direct_method : reward_predictor -> evaluation_policy -> logged_data -> estimated_performance
  val doubly_robust : reward_predictor -> evaluation_policy -> logged_data -> estimated_performance
  
  val confidence_interval : estimator -> logged_data -> (lower_bound * upper_bound)
end
```

评估指标：
- CTR@k（前k个结果的点击率）
- nDCG（归一化折损累积增益）
- MAP（平均精度均值）
- 用户满意度代理指标

数据要求：
- 日志包含展示概率
- 随机化流量收集
- 充分的动作覆盖
- 代表性用户样本

#### 在线实验设计
A/B 测试的因果推断视角：

```ocaml
module type OnlineExperiment = sig
  type experiment_config = {
    treatment_allocation : allocation_strategy;
    sample_size : int;
    duration : time_period;
    metrics : metric list;
  }
  
  val randomization_unit : UserLevel | QueryLevel | SessionLevel
  val stratified_randomization : strata -> allocation
  val sequential_testing : interim_analysis -> continue_or_stop
  val multiple_testing_correction : p_values -> adjusted_p_values
end
```

实验设计原则：
- 功效分析确定样本量
- 分层随机化提高精度
- 序贯测试早期决策
- 多重比较校正

因果推断增强：
- 利用协变量提高功效（CUPED）
- 工具变量处理不依从
- 中介分析理解机制
- 异质性效应分析

#### 长期效应评估
短期指标可能误导长期影响：

```ocaml
module type LongTermEvaluation = sig
  type short_term_metric = float
  type long_term_outcome = float
  
  val surrogate_index : short_term_metric list -> predicted_long_term
  val user_retention_analysis : experiment_data -> retention_curves
  val ecosystem_effects : treatment -> spillover_effects
  val novelty_effect_correction : time_series -> corrected_estimate
end
```

方法论：
- 代理指标方法
- 用户留存分析  
- 生态系统效应评估
- 新颖性效应校正

实践建议：
- 结合多个时间窗口
- 关注用户细分差异
- 监控意外副作用
- 定期回顾长期影响

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