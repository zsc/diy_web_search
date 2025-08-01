# Chapter 6: 查询处理引擎

## 章节大纲

### 6.1 查询解析器的类型设计
- 查询语法的形式化定义
- 查询 AST 的类型表示
- 语法糖与查询扩展
- 错误处理与容错设计

### 6.2 相关性计算的模块化架构
- 经典 IR 模型的接口抽象
- 特征提取管道的组合性设计
- 机器学习排序器的集成接口
- 相关性信号的层次化组织

### 6.3 查询优化器的设计原理
- 查询计划的代价模型
- 基于规则的优化策略
- 自适应查询执行
- 分布式查询的优化考虑

### 6.4 分布式聚合的接口定义
- 分片间的通信协议
- 聚合算子的代数性质
- 容错与部分结果处理
- 流式聚合与批处理权衡

---

查询处理引擎是搜索系统的大脑，负责理解用户意图、优化执行计划、协调分布式计算。本章深入探讨查询处理的四个核心模块：解析器将自然语言转换为结构化表示，相关性计算器评估文档匹配度，优化器选择最高效的执行路径，分布式聚合器整合多节点结果。我们将使用 OCaml 的类型系统定义清晰的模块接口，展示如何构建可扩展、高性能的查询处理架构。

## 6.1 查询解析器的类型设计

查询解析器将用户输入的自然语言或结构化查询转换为系统内部的抽象语法树（AST）。良好的类型设计能够确保查询表示的正确性，支持灵活的查询语法扩展，并为后续的优化和执行提供坚实基础。

### 查询语法的形式化定义

现代搜索引擎需要支持多种查询语法：布尔查询、短语查询、通配符查询、范围查询等。我们通过代数数据类型（ADT）来形式化这些构造：

```ocaml
type query_term =
  | Term of string
  | Phrase of string list
  | Wildcard of string * wildcard_type
  | Range of field * bound * bound
  | Fuzzy of string * int  (* term, edit distance *)
  
and wildcard_type = Prefix | Suffix | Infix

and field = string

and bound = 
  | Inclusive of string
  | Exclusive of string
  | Unbounded

type query_expr =
  | Leaf of query_term
  | And of query_expr list
  | Or of query_expr list
  | Not of query_expr
  | Boost of query_expr * float
  | Field of field * query_expr
```

这种设计的优势在于：
- **组合性**：复杂查询可以通过简单查询组合而成
- **类型安全**：编译器能够检查查询构造的有效性
- **可扩展**：添加新的查询类型只需扩展 ADT

### 查询 AST 的类型表示

解析器输出的 AST 需要包含足够的信息供后续阶段使用：

```ocaml
type position = {
  line: int;
  column: int;
  offset: int;
}

type 'a node = {
  value: 'a;
  position: position;
}

type typed_query = query_expr node

module type QUERY_PARSER = sig
  type token
  
  val tokenize : string -> token list result
  val parse : token list -> typed_query result
  val parse_string : string -> typed_query result
  
  (* 错误处理 *)
  type parse_error =
    | UnexpectedToken of token * position
    | UnmatchedParen of position
    | InvalidFieldName of string * position
    | SyntaxError of string * position
    
  val explain_error : parse_error -> string
end
```

位置信息的保留对于错误报告和查询分析至关重要。Result 类型确保错误处理的显式性。

### 语法糖与查询扩展

用户友好的查询语言需要支持各种语法糖。查询扩展器负责将简化的语法转换为规范形式：

```ocaml
module type QUERY_EXPANDER = sig
  type expansion_rule
  
  val expand : typed_query -> typed_query
  
  (* 常见的扩展规则 *)
  val synonym_expansion : (string * string list) list -> expansion_rule
  val prefix_completion : trie -> expansion_rule
  val spelling_correction : edit_distance_index -> expansion_rule
  val query_suggestion : query_log -> expansion_rule
  
  (* 组合多个扩展规则 *)
  val combine : expansion_rule list -> expansion_rule
  
  (* 配置扩展行为 *)
  type expansion_config = {
    max_expansions: int;
    boost_original: float;
    use_synonyms: bool;
    use_spelling: bool;
  }
  
  val with_config : expansion_config -> expansion_rule -> expansion_rule
end
```

扩展规则的设计考虑：
- **可组合性**：多个扩展规则可以按顺序应用
- **可配置性**：不同场景需要不同的扩展策略
- **性能约束**：限制扩展数量避免查询爆炸

### 错误处理与容错设计

实际系统中，用户查询经常包含语法错误。优秀的解析器应该：

```ocaml
module type ERROR_RECOVERY = sig
  (* 错误恢复策略 *)
  type recovery_strategy =
    | SkipToken
    | InsertToken of token
    | ReplaceToken of token
    | Backtrack of int
  
  (* 尝试从错误中恢复 *)
  val recover : parse_error -> token list -> (token list * recovery_strategy) option
  
  (* 容错解析：返回最可能的解析结果 *)
  val parse_fuzzy : string -> (typed_query * parse_error list) result
  
  (* 查询建议：基于错误提供修正建议 *)
  val suggest_corrections : parse_error -> string list
end
```

容错设计的权衡：
- **用户体验 vs 精确性**：过度纠错可能导致意外结果
- **性能开销**：错误恢复增加解析复杂度
- **歧义处理**：多种纠错可能时的选择策略

### 查询解析的性能优化

解析器性能直接影响查询延迟：

```ocaml
module type PARSER_CACHE = sig
  type cache_key = string
  type cache_value = typed_query
  
  (* LRU 缓存常见查询的解析结果 *)
  val get : cache_key -> cache_value option
  val put : cache_key -> cache_value -> unit
  
  (* 查询规范化用于提高缓存命中率 *)
  val normalize : string -> string
  
  (* 预编译的查询模板 *)
  type query_template
  val compile_template : string -> query_template
  val instantiate : query_template -> (string * string) list -> typed_query
end
```

优化策略：
- **缓存解析结果**：相同查询避免重复解析
- **查询规范化**：提高缓存命中率（如去除多余空格）
- **模板预编译**：参数化查询的快速实例化
- **并行词法分析**：长查询的分段并行处理

## 6.2 相关性计算的模块化架构

相关性计算是搜索引擎的核心，决定了哪些文档最匹配用户查询。模块化的架构设计允许我们组合不同的相关性信号，支持从经典 IR 模型到现代机器学习方法的平滑过渡。

### 经典 IR 模型的接口抽象

首先定义统一的相关性评分接口，支持各种经典检索模型：

```ocaml
module type SCORING_MODEL = sig
  type doc_stats = {
    term_freq: int;
    doc_length: int;
    unique_terms: int;
  }
  
  type corpus_stats = {
    num_docs: int;
    avg_doc_length: float;
    doc_freq: int;  (* for current term *)
    total_terms: int64;
  }
  
  (* 计算单个词项的得分 *)
  val score_term : doc_stats -> corpus_stats -> float
  
  (* 组合多个词项得分 *)
  val combine_scores : float list -> float
  
  (* 模型参数配置 *)
  type params
  val default_params : params
  val with_params : params -> t -> t
end

(* 具体模型实现 *)
module BM25 : SCORING_MODEL with type params = {k1: float; b: float}
module DFR : SCORING_MODEL with type params = {c: float}
module LM_Dirichlet : SCORING_MODEL with type params = {mu: float}
module LM_JelinekMercer : SCORING_MODEL with type params = {lambda: float}
```

这种抽象的优势：
- **统一接口**：不同模型可以互换使用
- **参数化设计**：支持模型调优
- **可组合性**：便于实现混合模型

### 特征提取管道的组合性设计

现代搜索系统需要提取多维度特征用于相关性计算：

```ocaml
module type FEATURE_EXTRACTOR = sig
  type feature_vector
  type query_context
  type doc_context
  
  (* 提取查询无关特征 *)
  val extract_static : doc_context -> feature_vector
  
  (* 提取查询相关特征 *)
  val extract_dynamic : query_context -> doc_context -> feature_vector
  
  (* 特征组合 *)
  val concat : feature_vector list -> feature_vector
  val merge : (string * feature_vector) list -> feature_vector
end

(* 特征管道的组合器 *)
module type FEATURE_PIPELINE = sig
  type pipeline
  
  (* 构建管道 *)
  val create : unit -> pipeline
  val add_extractor : string -> (module FEATURE_EXTRACTOR) -> pipeline -> pipeline
  
  (* 执行管道 *)
  val run : pipeline -> query -> document -> feature_vector
  
  (* 特征选择与变换 *)
  val select_features : string list -> pipeline -> pipeline
  val add_transformer : (feature_vector -> feature_vector) -> pipeline -> pipeline
end
```

常见的特征类型：
- **文本特征**：TF-IDF、BM25、语言模型得分
- **结构特征**：标题匹配、URL 深度、锚文本
- **用户特征**：点击率、停留时间、跳出率
- **时效特征**：发布时间、更新频率、内容新鲜度
- **语义特征**：词向量相似度、主题分布距离

### 机器学习排序器的集成接口

将机器学习模型集成到相关性计算中：

```ocaml
module type ML_RANKER = sig
  type model
  type prediction = float
  
  (* 模型加载与管理 *)
  val load_model : string -> model result
  val get_version : model -> string
  val get_features : model -> string list
  
  (* 预测接口 *)
  val predict : model -> feature_vector -> prediction
  val predict_batch : model -> feature_vector list -> prediction list
  
  (* 在线学习支持 *)
  val update : model -> feature_vector -> label:float -> model
  val save_checkpoint : model -> string -> unit result
end

(* 模型服务的抽象 *)
module type MODEL_SERVER = sig
  type server
  
  (* 多模型管理 *)
  val create : config -> server
  val register_model : string -> (module ML_RANKER) -> server -> unit
  val get_model : string -> server -> (module ML_RANKER) option
  
  (* A/B 测试支持 *)
  type experiment = {
    name: string;
    models: (string * float) list;  (* model_id, traffic_ratio *)
    hash_seed: string;
  }
  
  val run_experiment : experiment -> user_id:string -> string
  
  (* 模型热更新 *)
  val reload_model : string -> server -> unit result
  val watch_models : string -> (string -> unit) -> unit
end
```

集成考虑：
- **延迟预算**：模型推理时间的严格控制
- **特征对齐**：训练与服务时特征的一致性
- **降级策略**：模型服务不可用时的 fallback
- **版本管理**：模型的灰度发布与回滚

### 相关性信号的层次化组织

复杂的相关性计算需要组织多层次的信号：

```ocaml
module type RELEVANCE_CASCADE = sig
  type stage = 
    | L0_Boolean     (* 布尔匹配 *)
    | L1_Linear      (* 线性评分 *)
    | L2_Learning    (* 机器学习 *)
    | L3_Neural      (* 深度模型 *)
  
  type cascade_config = {
    stages: (stage * int) list;  (* stage, max_candidates *)
    time_budget: float;
    fallback_stage: stage;
  }
  
  (* 构建级联评分器 *)
  val create : cascade_config -> t
  val add_scorer : stage -> (module SCORING_MODEL) -> t -> t
  
  (* 执行级联评分 *)
  val score : t -> query -> document list -> (document * float) list
  
  (* 性能监控 *)
  type stage_stats = {
    candidates_in: int;
    candidates_out: int;
    time_spent: float;
    scorer_calls: int;
  }
  
  val get_stats : t -> (stage * stage_stats) list
end
```

级联设计的权衡：
- **计算效率**：早期阶段过滤大量候选
- **精度保证**：后期阶段提升排序质量
- **资源分配**：根据查询复杂度动态调整
- **延迟控制**：时间预算的合理分配

### 相关性调试与解释

为了理解和调优相关性计算，需要详细的解释机制：

```ocaml
module type RELEVANCE_EXPLAIN = sig
  type explanation
  
  (* 生成评分解释 *)
  val explain : scoring_model -> query -> document -> explanation
  
  (* 解释的结构化表示 *)
  type explain_node = {
    component: string;
    score: float;
    weight: float;
    children: explain_node list;
    metadata: (string * string) list;
  }
  
  val to_tree : explanation -> explain_node
  val to_json : explanation -> json
  val to_human_readable : explanation -> string
  
  (* 对比分析 *)
  val compare : document -> document -> query -> explain_node * explain_node
  
  (* 特征重要性分析 *)
  val feature_importance : ml_model -> feature_vector -> (string * float) list
  
  (* SHAP 值计算 *)
  val shap_values : ml_model -> feature_vector -> feature_vector
end
```

调试功能的应用：
- **相关性调优**：识别评分异常的原因
- **A/B 测试分析**：理解不同模型的差异
- **用户反馈处理**：解释为什么某个结果排名靠前
- **模型诊断**：发现特征工程的问题
