# Chapter 22: 查询改写系统

## 章节概述

查询改写系统是现代搜索引擎提升用户体验的核心组件。用户输入的查询往往存在拼写错误、语义模糊、表达不完整等问题。本章深入探讨查询改写的架构设计，从传统的拼写纠错到基于深度学习的语义理解，从单轮改写到多轮对话优化，以及跨语言查询的标准化处理。我们将分析各种改写策略的设计权衡，探讨如何构建一个高效、准确且可扩展的查询改写系统。

## 目录

1. 拼写纠错与自动补全架构
   - 编辑距离算法与优化
   - 统计语言模型的应用
   - 用户行为驱动的纠错
   - 实时补全的系统设计

2. 基于 BERT 的同义词扩展与意图理解
   - 预训练模型的查询理解
   - 同义词挖掘与扩展策略
   - 查询意图分类架构
   - 上下文感知的语义匹配

3. 多轮查询上下文改写策略
   - 会话状态管理设计
   - 指代消解与省略补全
   - 查询重构的决策机制
   - 长会话的记忆管理

4. 跨语言查询标准化与代码混合处理
   - 多语言检测与切换
   - 查询翻译架构设计
   - 代码混合的处理策略
   - 跨语言同义词映射

## 1. 拼写纠错与自动补全架构

### 1.1 编辑距离算法与优化

拼写纠错的核心是计算用户输入与候选词之间的相似度。经典的编辑距离算法提供了基础框架，但在大规模搜索系统中需要多层优化。

#### 算法选择与权衡

**Levenshtein 距离** 是最基础的选择，支持插入、删除、替换三种操作。对于键盘输入场景，可以考虑 **Damerau-Levenshtein 距离**，额外支持相邻字符交换操作。

```ocaml
module type EDIT_DISTANCE = sig
  type config = {
    max_distance : int;
    weights : {
      insertion : float;
      deletion : float;
      substitution : float;
      transposition : float option;
    };
    keyboard_layout : keyboard_model option;
  }
  
  val compute : config -> string -> string -> float option
  val compute_batch : config -> string -> string array -> (string * float) array
end
```

在实际实现中，还需要考虑更复杂的距离度量：

- **Jaro-Winkler 距离**: 对前缀匹配给予更高权重，适合人名和品牌纠错
- **音似距离（Soundex/Metaphone）**: 处理同音异形词，如 "iPhone" vs "aifone"
- **视觉相似度**: OCR 场景下的字符相似度，如 "0" vs "O", "1" vs "l"
- **键盘距离**: 基于键盘布局的误按概率模型

#### 优化策略

1. **前缀树加速**: 使用 Trie 结构存储词典，支持早期剪枝
   - 分支限界：当前缀的最小可能距离已超过阈值时剪枝
   - 共享计算：相同前缀的编辑距离可复用
   - 内存优化：使用 DAWG（Directed Acyclic Word Graph）压缩存储

2. **BK-Tree 索引**: 利用度量空间的三角不等式快速过滤候选词
   - 构建时间 O(n log n)，查询时间 O(log n) 平均情况
   - 支持任意满足三角不等式的距离度量
   - 可以并行构建多个子树提升构建速度

3. **并行计算**: 动态规划矩阵的对角线并行化
   - GPU 加速：利用 CUDA/OpenCL 并行计算编辑距离矩阵
   - SIMD 指令：使用向量化指令同时处理多个字符比较
   - 分块计算：将大矩阵分解为小块，减少 cache miss

4. **近似算法**: 使用 LSH 或 MinHash 进行粗筛
   - SimHash：将字符串映射到汉明空间，快速计算相似度
   - MinHash + LSH：处理集合相似度，适合 n-gram 表示
   - 学习型哈希：使用神经网络学习最优的哈希函数

#### 实时性能优化

在搜索引擎中，拼写纠错必须在毫秒级完成：

```ocaml
module type SPELL_CHECKER = sig
  type t
  type candidate = {
    word : string;
    score : float;
    source : [`Dictionary | `QueryLog | `UserHistory];
  }
  
  val create : config -> dictionary -> t
  val check : t -> string -> candidate list
  val check_with_context : t -> string -> context:string list -> candidate list
  
  type optimization = 
    | CachingStrategy of { cache_size : int; eviction : [`LRU | `LFU] }
    | Precomputation of { common_errors : (string * string) list }
    | TrigramIndex of { false_positive_rate : float }
end
```

### 1.2 统计语言模型的应用

纯编辑距离无法处理同音异形词和上下文相关的纠错。统计语言模型提供了更智能的纠错能力。

#### N-gram 模型架构

```ocaml
module type LANGUAGE_MODEL = sig
  type model
  type score = float
  
  val score_sequence : model -> string array -> score
  val candidates : model -> string -> context:string array -> (string * score) list
  val interpolate : (model * float) list -> model
end
```

实践中常用的架构：
- **Unigram 频率**: 处理独立词的常见性
- **Bigram/Trigram**: 捕捉局部上下文
- **Skip-gram**: 处理非连续依赖
- **神经语言模型**: 使用 LSTM/Transformer 建模长距离依赖

#### Noisy Channel Model

经典的拼写纠错使用噪声信道模型：

```
P(correct|observed) ∝ P(observed|correct) × P(correct)
```

其中：
- `P(observed|correct)`: 错误模型，可从用户纠错行为学习
- `P(correct)`: 语言模型，可从大规模语料统计

```ocaml
module type NOISY_CHANNEL = sig
  type error_model
  type language_model
  
  val train_error_model : (string * string) list -> error_model
  val train_language_model : corpus -> language_model
  
  val correct : 
    error_model -> 
    language_model -> 
    string -> 
    (string * float) list
    
  type feature = 
    | EditDistance of float
    | Phonetic of float  
    | Keyboard of float
    | Frequency of float
    | Context of float
end
```

#### 上下文感知纠错

现代搜索引擎的查询纠错必须考虑上下文：

1. **会话上下文**: "买了 ipone" → "iphone"（基于之前搜索 "iPhone 13 价格"）
2. **地理上下文**: "pikza" → "pizza"（用户在意大利）
3. **时间上下文**: "kobe" → "covid"（2020年疫情期间）
4. **领域上下文**: "jave" → "java"（在编程相关搜索中）

#### 混合模型架构

实践中通常组合多种信号：

```ocaml
module type HYBRID_CORRECTOR = sig
  type evidence = 
    | Statistical of { score : float; source : string }
    | Neural of { score : float; model : string }
    | Behavioral of { score : float; signal_type : string }
    
  type correction = {
    original : string;
    candidates : (string * evidence list) list;
  }
  
  val combine_evidences : evidence list -> float
  val rank_corrections : correction -> (string * float) list
end
```

### 1.3 用户行为驱动的纠错

用户的历史行为提供了宝贵的纠错信号。点击日志中的查询-点击对可以挖掘出高质量的纠错映射。

#### 行为信号的类型

1. **查询重写序列**: 用户在短时间内的连续查询
   - 时间窗口：通常 30 秒内的连续查询
   - 编辑距离：相邻查询的相似度
   - 结果改善：后续查询是否获得更多点击

2. **点击跳过模式**: 某些结果被系统性跳过
   - 位置偏差校正：考虑结果展示位置
   - 群体行为：多用户的一致跳过行为
   - 时间演化：跳过模式的时间稳定性

3. **停留时间**: 区分好结果和坏结果
   - 短停留（<10s）：可能表示结果不相关
   - 长停留（>30s）：用户找到有用信息
   - 回访行为：是否返回搜索结果页

4. **会话结束信号**: 找到满意结果的标志
   - 搜索终止：不再继续搜索
   - 任务完成：后续行为表明任务完成
   - 满意度信号：显式或隐式的满意度反馈

```ocaml
module type BEHAVIOR_MODEL = sig
  type correction_candidate = {
    original : string;
    corrected : string;
    confidence : float;
    evidence : evidence list;
  }
  
  and evidence = 
    | QueryRewrite of { time_gap : float; session_id : string }
    | ClickPattern of { ctr_diff : float; num_impressions : int }
    | DwellTime of { avg_dwell : float; num_clicks : int }
    
  val mine_corrections : query_log -> correction_candidate list
  val validate : correction_candidate -> validation_result
end
```

#### 纠错对挖掘算法

从用户行为中挖掘纠错对的关键步骤：

1. **会话分割**: 识别相关的查询序列
   ```ocaml
   type session_splitter = {
     time_threshold : duration;
     similarity_threshold : float;
     task_classifier : query list -> task_type option;
   }
   ```

2. **候选对生成**: 找出潜在的纠错关系
   - 连续查询对：(q1, q2) where edit_distance(q1, q2) < threshold
   - 跳跃查询对：考虑中间有探索性查询的情况
   - 多跳纠错：q1 → q2 → q3 的传递关系

3. **置信度计算**: 评估纠错对的可靠性
   ```ocaml
   type confidence_factors = {
     frequency : int;           (* 出现次数 *)
     consistency : float;       (* 不同用户的一致性 *)
     improvement : float;       (* 结果质量提升 *)
     bidirectional : bool;      (* 是否存在反向纠错 *)
   }
   ```

4. **噪声过滤**: 排除错误的纠错关系
   - 品牌变体：iPhone vs iPhone 13（不是纠错）
   - 查询细化：手机 → 华为手机（不是纠错）
   - 同义变换：汽车 → 轿车（不是纠错）

#### 实时学习架构

用户行为驱动的纠错需要实时更新能力：

```ocaml
module type ONLINE_LEARNING = sig
  type model
  type update = 
    | NewCorrection of { query : string; correction : string; signal : signal }
    | Feedback of { correction : string; accepted : bool }
    
  val update : model -> update -> model
  val merge : model list -> model  (* 分布式合并 *)
  val checkpoint : model -> unit   (* 持久化 *)
  val rollback : model -> version -> model
end
```

### 1.4 实时补全的系统设计

自动补全需要在用户输入的同时提供实时建议，这对系统的响应时间和资源消耗提出了严格要求。

#### 索引结构设计

```ocaml
module type AUTOCOMPLETE_INDEX = sig
  type t
  type suggestion = {
    text : string;
    score : float;
    metadata : metadata;
  }
  
  val build : (string * float * metadata) list -> t
  val suggest : t -> prefix:string -> limit:int -> suggestion list
  val update : t -> string -> float -> metadata -> t
end
```

常用的数据结构选择：

1. **Trie + Top-K**: 每个节点存储 Top-K 补全
   - 优点：前缀查询 O(m)，m 为前缀长度
   - 缺点：内存占用大，更新代价高
   - 优化：只在叶节点存储完整信息

2. **压缩前缀树（Radix Tree）**: 减少内存占用
   - 路径压缩：单子节点路径合并
   - 适合稀疏数据和长尾查询
   - 支持高效的范围查询

3. **三元搜索树（TST）**: 平衡内存和查询效率
   - 结合 BST 和 Trie 的优点
   - 内存效率优于 Trie
   - 支持近似匹配和通配符

4. **前缀哈希表**: 空间换时间的极致
   - 预计算所有可能前缀的结果
   - 查询时间 O(1)
   - 适合有限词表和短前缀

#### 排序与个性化

自动补全的排序需要综合多种信号：

```ocaml
module type SUGGESTION_RANKER = sig
  type ranking_features = {
    query_frequency : float;      (* 全局查询频率 *)
    user_frequency : float;       (* 用户查询频率 *)
    temporal_trend : float;       (* 时间趋势分数 *)
    location_relevance : float;   (* 地理相关性 *)
    context_similarity : float;   (* 上下文相似度 *)
    completion_rate : float;      (* 补全选择率 *)
  }
  
  val extract_features : suggestion -> user_context -> ranking_features
  val rank : suggestion list -> user_context -> suggestion list
  
  type personalization_level = 
    | Global                      (* 无个性化 *)
    | Cohort of string           (* 群组个性化 *)
    | User of user_id            (* 用户个性化 *)
    | Session of session_id      (* 会话个性化 *)
end
```

#### 实时性能优化策略

1. **多级缓存架构**:
   ```ocaml
   type cache_hierarchy = {
     browser_cache : { size : int; ttl : duration };
     edge_cache : { locations : string list; policy : eviction_policy };
     app_cache : { implementation : [`Redis | `Memcached]; sharding : bool };
     index_cache : { hot_prefixes : int; bloom_filter : bool };
   }
   ```

2. **预测性预取**:
   - 基于输入速度预测下一个字符
   - 预取可能的后续前缀结果
   - 利用键盘布局预测可能的误按

3. **流式处理**:
   - 支持逐字符流式返回结果
   - 客户端渐进式渲染
   - 取消过期的请求

4. **负载均衡**:
   ```ocaml
   type load_balancing = 
     | RoundRobin
     | LeastConnections
     | PrefixAffinity of { hash_function : string -> int }
     | GeoProximity of { user_location : location }
   ```

#### 分布式架构考虑

1. **分片策略**: 
   - 前缀分片：相同前缀路由到同一节点
   - 一致性哈希：支持动态扩缩容
   - 复合分片：前缀 + 用户 ID 二级分片

2. **数据同步**:
   ```ocaml
   module type SYNC_PROTOCOL = sig
     type sync_method = 
       | FullSync of { frequency : duration }
       | IncrementalSync of { batch_size : int; latency : duration }
       | EventDriven of { stream : event_stream }
       
     val sync : source:node -> target:node -> sync_method -> unit
     val reconcile : node list -> consistency_level -> unit
   end
   ```

3. **更新机制**: 
   - 增量更新：只更新变化的部分
   - 批量重建：定期完全重建索引
   - 混合策略：热数据增量，冷数据批量

4. **个性化层次**:
   - 全局：所有用户共享
   - 群组：相似用户群体
   - 用户：个人历史和偏好
   - 会话：当前会话上下文

## 2. 基于 BERT 的同义词扩展与意图理解

### 2.1 预训练模型的查询理解

BERT 等预训练模型为查询理解带来了革命性的提升。但如何高效地将这些模型应用于在线查询改写是一个挑战。

#### 模型架构选择

```ocaml
module type QUERY_ENCODER = sig
  type model
  type encoding = float array
  
  val encode : model -> string -> encoding
  val encode_batch : model -> string array -> encoding array
  val similarity : encoding -> encoding -> float
  
  type model_config = {
    model_type : [`BERT | `RoBERTa | `ELECTRA | `Custom of string];
    max_length : int;
    pooling : [`CLS | `Mean | `Max | `Attention];
    device : [`CPU | `GPU of int | `TPU];
  }
end
```

#### 查询特定的模型适配

搜索查询与一般文本有显著差异，需要特殊适配：

1. **查询特征**:
   - 平均长度短（2-4 个词）
   - 语法不完整（缺少冠词、介词）
   - 领域术语多（品牌、型号、专业词汇）
   - 含有特殊符号（价格、型号中的符号）

2. **预训练策略**:
   ```ocaml
   type pretraining_task = 
     | MaskedQueryModeling of { mask_ratio : float }
     | QueryNextQuery of { window_size : int }
     | ClickedDocumentPrediction
     | QueryIntentClassification
   ```

3. **领域适应**:
   - 在查询日志上继续预训练
   - 使用点击数据作为弱监督信号
   - 结合知识图谱增强实体理解

#### 推理优化策略

1. **模型蒸馏**: 
   ```ocaml
   module type DISTILLATION = sig
     type teacher_model
     type student_model
     
     val distill : 
       teacher:teacher_model -> 
       architecture:model_architecture ->
       training_data:dataset ->
       student_model
       
     type distillation_loss = 
       | KLDivergence of { temperature : float }
       | MSE of { layer_mapping : (int * int) list }
       | Attention of { head_mapping : bool }
       | Combined of (distillation_loss * float) list
   end
   ```

2. **量化技术**: 
   - INT8 量化：精度损失小，速度提升 2-4x
   - 混合精度：关键层保持 FP16，其他层 INT8
   - 动态量化：根据输入范围动态调整量化参数

3. **剪枝策略**: 
   - 结构化剪枝：移除整个注意力头或层
   - 非结构化剪枝：移除权重矩阵中的元素
   - 任务相关剪枝：保留对查询理解重要的部分

4. **动态计算**: 
   ```ocaml
   type dynamic_inference = {
     early_exit : float -> bool;  (* 置信度阈值 *)
     layer_skip : query -> int list;  (* 跳过的层 *)
     adaptive_length : string -> int;  (* 动态序列长度 *)
   }
   ```

#### 缓存与服务化

为了支持大规模在线服务，需要精心设计缓存和服务架构：

```ocaml
module type MODEL_SERVER = sig
  type server
  type request = {
    queries : string array;
    model_version : string option;
    timeout : duration;
  }
  
  type response = {
    embeddings : encoding array;
    latency : duration;
    cache_hit : bool array;
  }
  
  val serve : server -> request -> response Lwt.t
  
  type cache_config = {
    embedding_cache : {
      size : int;
      ttl : duration;
      similarity_threshold : float;  (* 近似匹配 *)
    };
    model_cache : {
      versions : int;  (* 同时缓存的模型版本数 *)
      preload : bool;  (* 预加载到 GPU *)
    };
  }
end
```

### 2.2 同义词挖掘与扩展策略

基于 BERT 的同义词扩展不仅考虑词形相似，更重要的是语义相似。

#### 挖掘方法

1. **向量空间近邻**: 在 BERT 嵌入空间中寻找近邻
2. **掩码语言模型**: 使用 MLM 任务生成同义词
3. **对比学习**: 从点击日志中学习查询相似性
4. **知识图谱对齐**: 结合外部知识库

```ocaml
module type SYNONYM_EXPANDER = sig
  type expansion = {
    original : string;
    synonyms : (string * float) list;
    context : string option;
  }
  
  val expand : query:string -> context:string list -> expansion
  val filter : expansion -> criteria -> expansion
  
  type criteria = {
    min_similarity : float;
    max_candidates : int;
    domain_specific : bool;
    preserve_intent : bool;
  }
end
```

### 2.3 查询意图分类架构

理解用户的查询意图是提供准确结果的前提。意图分类帮助系统选择合适的改写策略。

#### 意图体系设计

```ocaml
type query_intent = 
  | Navigational of { target_site : string option }
  | Informational of { topic : string; question_type : question_type option }
  | Transactional of { action : string; object_type : string option }
  | Local of { location : string option; service_type : string option }
  | Multimedia of { media_type : [`Image | `Video | `Audio] }
  | Code of { language : string option; task : string option }

and question_type = 
  | Definition | HowTo | Comparison | List | Factoid | Opinion
```

#### 多任务学习架构

意图分类通常与其他任务联合训练：
- 查询分类
- 命名实体识别
- 领域检测
- 紧急度判断

### 2.4 上下文感知的语义匹配

在实际搜索场景中，查询的含义高度依赖上下文。上下文感知的改写能显著提升搜索质量。

#### 上下文来源

1. **会话历史**: 之前的查询和点击
2. **用户画像**: 长期兴趣和偏好
3. **时空信息**: 位置、时间、设备
4. **任务上下文**: 正在进行的任务类型

```ocaml
module type CONTEXTUAL_REWRITER = sig
  type context = {
    session_queries : (string * timestamp) list;
    clicked_results : (string * url * timestamp) list;
    user_profile : profile option;
    spatiotemporal : spatiotemporal_info;
  }
  
  val rewrite : query:string -> context:context -> string list
  val explain : query:string -> context:context -> rewrite:string -> explanation
end
```

## 3. 多轮查询上下文改写策略

### 3.1 会话状态管理设计

多轮搜索需要维护复杂的会话状态，包括查询历史、实体状态、任务进展等。

#### 状态表示模型

```ocaml
module type SESSION_STATE = sig
  type t
  type entity = { 
    id : string; 
    type_ : string; 
    mentions : (int * string) list;
    attributes : (string * value) list;
  }
  
  type task_state = 
    | Exploring of { topic : string; depth : int }
    | Comparing of { entities : entity list; criteria : string list }
    | Refining of { base_query : string; refinements : string list }
    | Completed of { final_query : string; satisfaction : float option }
    
  val init : unit -> t
  val update : t -> query:string -> results:result list -> t
  val get_entities : t -> entity list
  val get_task : t -> task_state
end
```

#### 存储策略

1. **内存存储**: Redis 中的会话缓存
2. **持久化**: Cassandra 中的长期历史
3. **分级存储**: 热数据在内存，冷数据在磁盘
4. **压缩策略**: 老化session的状态压缩

### 3.2 指代消解与省略补全

用户在多轮对话中常使用指代词和省略表达，系统需要准确理解这些简化表达。

#### 指代消解类型

1. **代词指代**: "它"、"这个"、"那些"
2. **定语省略**: "红色的" → "红色的手机"
3. **动词省略**: "北京的呢" → "北京的天气怎么样"
4. **比较省略**: "更便宜的" → "比iPhone更便宜的手机"

```ocaml
module type COREFERENCE_RESOLVER = sig
  type mention = {
    text : string;
    span : int * int;
    type_ : [`Pronoun | `Definite | `Demonstrative | `Ellipsis];
  }
  
  type resolution = {
    mention : mention;
    antecedent : string;
    confidence : float;
    evidence : string list;
  }
  
  val resolve : query:string -> context:string list -> resolution list
  val expand_ellipsis : query:string -> context:string list -> string
end
```

### 3.3 查询重构的决策机制

不是所有查询都需要改写，过度改写可能改变用户意图。需要智能的决策机制。

#### 改写触发条件

```ocaml
type rewrite_trigger = 
  | LowResultQuality of { avg_score : float; num_results : int }
  | HighAmbiguity of { intent_entropy : float; num_interpretations : int }
  | ExplicitRefinement of { refinement_type : string }
  | ContextualContinuation of { coherence_score : float }
  | SpellingError of { confidence : float; severity : int }
```

#### 改写策略选择

```ocaml
module type REWRITE_PLANNER = sig
  type strategy = 
    | NoRewrite
    | SpellCorrect of { aggressive : bool }
    | SynonymExpand of { num_terms : int; boost_original : bool }
    | IntentClarify of { options : string list }
    | ContextExpand of { context_weight : float }
    | MultiStrategy of strategy list
    
  val plan : query:string -> signals:signal list -> strategy
  val combine : strategy list -> strategy
end
```

### 3.4 长会话的记忆管理

随着会话轮数增加，如何有效管理历史信息成为挑战。需要在保持相关信息和控制内存消耗之间平衡。

#### 记忆压缩策略

1. **重要性评分**: 基于信息增益保留关键查询
2. **实体中心**: 围绕核心实体组织记忆
3. **任务抽象**: 将具体查询抽象为任务表示
4. **滑动窗口**: 结合时间衰减的固定窗口

```ocaml
module type MEMORY_MANAGER = sig
  type memory
  type importance_scorer = query -> context -> float
  
  val create : capacity:int -> importance_scorer -> memory
  val add : memory -> query -> result list -> memory
  val compress : memory -> compression_strategy -> memory
  val retrieve : memory -> query -> context_window
  
  type compression_strategy = 
    | TopK of int
    | EntityCentric of { max_entities : int }
    | TaskAbstraction of { abstractor : query list -> task_state }
    | TemporalDecay of { half_life : duration }
end
```

## 4. 跨语言查询标准化与代码混合处理

### 4.1 多语言检测与切换

在多语言环境中，准确检测查询语言是第一步。但实际查询often包含多种语言混合。

#### 语言检测挑战

1. **短文本**: 查询通常只有几个词
2. **专有名词**: 品牌名、地名的语言归属
3. **代码混合**: "iPhone手机price"
4. **拼音输入**: "beijing tianqi"

```ocaml
module type LANGUAGE_DETECTOR = sig
  type detection = {
    language : language;
    confidence : float;
    script : script;
    segments : (int * int * language) list option;
  }
  
  val detect : string -> detection
  val detect_mixed : string -> detection list
  
  type model = 
    | NgramBased of { n : int; smoothing : smoothing_method }
    | NeuralBased of { architecture : string; checkpoint : string }
    | Hybrid of { models : (model * float) list }
end
```

### 4.2 查询翻译架构设计

跨语言搜索需要高质量的查询翻译，但又不同于通用翻译。

#### 查询翻译的特殊性

1. **领域适应**: 搜索查询的语言分布特殊
2. **保留歧义**: 不要过度消歧
3. **命名实体**: 保持原文或音译
4. **查询结构**: 保持操作符和语法

```ocaml
module type QUERY_TRANSLATOR = sig
  type translation = {
    text : string;
    confidence : float;
    alternatives : (string * float) list;
    preserved_entities : (string * entity_type) list;
  }
  
  val translate : source:language -> target:language -> string -> translation
  val translate_multi : targets:language list -> string -> translation list
  
  type config = {
    model : [`Statistical | `Neural | `Hybrid];
    preserve_entities : bool;
    maintain_ambiguity : bool;
    domain_adaptation : domain option;
  }
end
```

### 4.3 代码混合的处理策略

代码混合(Code-mixing)是多语言用户的常见现象，需要特殊处理。

#### 代码混合类型

```ocaml
type code_mixing = 
  | IntraWord of { word : string; languages : language list }
  | InterWord of { segments : (string * language) list }
  | ScriptMixing of { segments : (string * script) list }
  | Transliteration of { original : string; script : script; language : language }
```

#### 处理策略

1. **分段识别**: 识别不同语言的片段
2. **统一表示**: 转换到目标语言或中间表示
3. **混合索引**: 同时索引多种形式
4. **查询扩展**: 生成单语言变体

### 4.4 跨语言同义词映射

不同语言中的"同义"概念可能有细微差别，需要精确的跨语言映射。

#### 映射来源

1. **平行语料**: 从翻译对中学习
2. **知识图谱**: 多语言知识库对齐
3. **用户行为**: 跨语言点击模式
4. **词向量对齐**: 跨语言词嵌入

```ocaml
module type CROSS_LINGUAL_SYNONYMS = sig
  type mapping = {
    source : string * language;
    targets : (string * language * float) list;
    context_dependent : bool;
    domain : domain option;
  }
  
  val find_synonyms : term:string -> source:language -> target:language -> mapping
  val validate : mapping -> validation_method -> float
  
  type validation_method = 
    | BackTranslation
    | ClickCorrelation of { min_cooccurrence : int }
    | EmbeddingSimilarity of { threshold : float }
    | HumanAnnotation
end
```

## 本章小结

查询改写系统是连接用户表达和系统理解的桥梁。本章探讨了四个核心模块的架构设计：

1. **拼写纠错与自动补全**: 从编辑距离到用户行为，从实时响应到分布式架构
2. **基于BERT的语义理解**: 预训练模型的高效应用，同义词扩展和意图分类
3. **多轮会话改写**: 状态管理、指代消解、智能决策和记忆压缩
4. **跨语言处理**: 语言检测、查询翻译、代码混合和跨语言映射

关键的架构原则：
- **效率优先**: 查询改写必须低延迟
- **准确性与召回率平衡**: 不要过度改写
- **上下文感知**: 充分利用会话和用户信息
- **可解释性**: 用户需要理解改写原因
- **渐进式改进**: 从简单规则到复杂模型

## 练习题

### 基础题

1. **编辑距离优化** 
   - 设计一个支持键盘布局感知的编辑距离算法。相邻键的替换成本应该更低。
   - *Hint*: 考虑QWERTY键盘的物理布局，定义键之间的距离矩阵

<details>
<summary>参考答案</summary>

建立键盘距离矩阵，将物理距离转换为编辑成本。对于QWERTY键盘，可以用曼哈顿距离或欧氏距离计算键位距离。修改替换成本计算：`cost = base_cost * (1 - exp(-distance))`。这样相邻键（如Q和W）的替换成本会显著低于远距离键（如Q和P）。

</details>

2. **自动补全数据结构**
   - 比较Trie、三元搜索树和前缀哈希表在自动补全场景下的优劣。考虑内存占用、查询速度和更新效率。
   - *Hint*: 考虑不同的查询分布和更新频率

<details>
<summary>参考答案</summary>

- Trie: 查询O(m)，内存占用大但支持前缀共享，更新简单
- 三元搜索树: 查询O(m log n)，内存效率更高，适合稀疏数据
- 前缀哈希表: 查询O(1)但仅支持定长前缀，内存占用取决于前缀长度选择

实践中，短前缀(2-3字符)用哈希表，完整补全用压缩Trie。

</details>

3. **查询意图分类体系**
   - 设计一个适用于电商搜索的查询意图分类体系。考虑购买阶段和用户需求的多样性。
   - *Hint*: 考虑用户购买旅程的不同阶段

<details>
<summary>参考答案</summary>

电商查询意图可分为：
- 探索期：类目浏览、趋势了解
- 研究期：产品比较、评测查询、规格咨询  
- 决策期：价格查询、促销信息、库存确认
- 购买后：物流查询、售后服务、使用指南

每个阶段对应不同的改写策略，如研究期扩展同类产品，决策期强调促销信息。

</details>

### 挑战题

4. **混合语言查询处理**
   - 设计一个处理中英混合查询的系统架构。例如："iPhone 13 pro max 价格 深圳"
   - *Hint*: 考虑实体识别、语言边界检测和索引策略

<details>
<summary>参考答案</summary>

架构包括：
1. 语言片段识别：基于字符和词汇特征的序列标注
2. 实体保护：识别品牌、型号等不应翻译的实体
3. 混合索引：同时索引原文、拼音、翻译版本
4. 查询理解：将混合查询规范化为结构化表示
5. 排序时融合：不同语言版本的结果融合

关键是保持语义完整性while支持灵活匹配。

</details>

5. **会话状态压缩**
   - 设计一个会话状态压缩算法，在保持查询改写质量的同时，将存储需求降低90%。
   - *Hint*: 考虑信息论中的有损压缩原理

<details>
<summary>参考答案</summary>

压缩策略：
1. 实体中心表示：只保留关键实体及其属性变化
2. 查询聚类：相似查询合并为一个代表  
3. 任务抽象：将查询序列抽象为任务图
4. 重要性采样：基于信息增益保留关键转折点
5. 增量编码：只存储状态变化而非完整状态

使用注意力机制评估历史查询对当前改写的重要性。

</details>

6. **个性化改写冲突**
   - 当用户的个性化偏好与查询的明确意图冲突时，如何设计改写策略？例如，素食者搜索"牛排餐厅推荐"。
   - *Hint*: 考虑显式意图和隐式偏好的权重

<details>
<summary>参考答案</summary>

冲突解决框架：
1. 意图强度评估：明确词汇("牛排")vs模糊表达("好吃的")
2. 偏好置信度：长期稳定偏好vs短期兴趣
3. 上下文线索：是否为他人查询（礼物、招待）
4. 渐进式策略：主结果满足显式需求，辅助推荐符合偏好
5. 透明化解释：告知用户个性化调整的存在

核心原则：尊重用户显式表达，谨慎应用隐式偏好。

</details>

7. **实时A/B测试框架**
   - 设计一个查询改写策略的在线A/B测试框架，要求支持多臂老虎机算法动态调整流量分配。
   - *Hint*: 考虑探索与利用的平衡，以及统计显著性

<details>
<summary>参考答案</summary>

框架设计：
1. 流量分配器：Thompson采样或UCB算法
2. 特征分桶：按查询类型、用户群体分别测试
3. 指标体系：短期CTR + 长期满意度
4. 早停机制：贝叶斯方法评估显著性
5. 反馈回路：实时更新改写模型权重

关键挑战：处理延迟反馈、网络效应和季节性变化。

</details>

8. **跨语言查询对齐评估**
   - 设计一个评估跨语言查询改写质量的指标体系，不依赖人工标注。
   - *Hint*: 利用用户行为信号和回译一致性

<details>
<summary>参考答案</summary>

评估指标：
1. 回译一致性：改写后回译的语义保持度
2. 点击重合度：不同语言查询的结果点击相似性  
3. 会话延续性：改写后是否减少refinement
4. 结果互信息：不同语言结果集的信息重合
5. 实体保持率：关键实体的识别和保留

组合指标：加权平均，权重通过与人工评估的相关性学习。

</details>

## 常见陷阱与错误 (Gotchas)

1. **过度改写**: 将用户的精确查询改写为模糊查询
   - 解决方案：保留原始查询作为必选项，改写作为扩展

2. **上下文过拟合**: 过度依赖历史，忽视当前查询的独立性
   - 解决方案：设置上下文权重衰减，允许上下文重置

3. **多语言实体混淆**: 同一实体的不同语言表达未能对齐
   - 解决方案：建立多语言实体知识库，统一实体ID

4. **改写解释缺失**: 用户不理解为什么查询被改写
   - 解决方案：提供改写原因提示，支持改写撤销

5. **性能瓶颈**: 复杂的改写模型导致延迟增加
   - 解决方案：分级改写，简单情况快速路径，复杂情况异步改写

6. **隐私泄露**: 查询改写暴露用户历史
   - 解决方案：差分隐私技术，限制历史信息的使用

## 最佳实践检查清单

### 架构设计
- [ ] 是否设计了多级改写策略（快速路径 + 深度改写）？
- [ ] 是否考虑了缓存策略减少重复计算？
- [ ] 是否支持改写策略的A/B测试？
- [ ] 是否有降级方案应对模型服务故障？

### 算法选择  
- [ ] 是否根据查询类型选择合适的改写算法？
- [ ] 是否平衡了准确性和召回率？
- [ ] 是否考虑了不同语言的特殊性？
- [ ] 是否有持续学习机制？

### 用户体验
- [ ] 是否提供改写的透明度和可控性？
- [ ] 是否保留了用户的原始意图？
- [ ] 是否处理了改写失败的情况？
- [ ] 是否支持改写历史的查看？

### 性能优化
- [ ] 是否满足延迟要求（<50ms）？
- [ ] 是否优化了内存使用？
- [ ] 是否支持增量更新？
- [ ] 是否有预计算和预热机制？

### 质量保证
- [ ] 是否有改写质量的自动评估？
- [ ] 是否监控改写后的用户满意度？
- [ ] 是否有bad case的收集和分析？
- [ ] 是否定期更新改写规则和模型？