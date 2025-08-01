# Chapter 2: 文本处理管道

文本处理管道是搜索引擎的第一道关卡，决定了后续所有组件能够处理的数据质量。本章深入探讨如何设计一个既高效又可扩展的文本处理系统，从原始文本到可索引的标准化词项。我们将使用 OCaml 的类型系统来定义清晰的模块接口，确保各组件之间的解耦与可组合性。

## 学习目标

完成本章后，你将能够：
- 设计支持多语言的模块化分词器架构
- 实现高性能的语言检测系统
- 构建灵活的文本规范化管道
- 优化 N-gram 索引的存储与查询性能
- 理解文本处理中的设计权衡与性能优化策略

## 2.1 分词器的接口设计与扩展性

分词是将连续的文本流切分成独立词汇单元的过程。对于搜索引擎而言，分词的质量直接影响查询的准确性和召回率。

### 2.1.1 模块化分词器架构

```ocaml
module type TOKENIZER = sig
  type token = {
    surface: string;      (* 原始形式 *)
    normalized: string;   (* 规范化形式 *)
    position: int;        (* 在文档中的位置 *)
    offset: int * int;    (* 字节偏移量 *)
    attributes: (string * string) list;  (* 扩展属性 *)
  }
  
  type config
  val tokenize : config -> string -> token list
  val register_filter : (token -> token option) -> unit
  val compose : TOKENIZER -> TOKENIZER -> TOKENIZER
end
```

这种设计允许我们构建可组合的分词管道。例如，可以将基础分词器与词干提取器、同义词扩展器组合：

### 2.1.2 语言特定分词策略

不同语言需要不同的分词策略：

**空格分隔语言**（英语、法语等）：
- 基于空格和标点的简单分割
- 处理缩写、连字符词汇
- 识别 URL、邮箱等特殊模式

**无空格语言**（中文、日文等）：
- 基于词典的最大匹配算法
- 统计模型（HMM、CRF）
- 深度学习模型（BiLSTM-CRF）

**黏着语言**（土耳其语、芬兰语等）：
- 形态学分析器
- 词干提取与词缀处理
- 复合词分解

### 2.1.3 插件系统设计

为支持新语言和特殊处理需求，分词器需要灵活的插件机制：

```ocaml
module type TOKENIZER_PLUGIN = sig
  val name : string
  val version : string
  val supported_languages : string list
  val process : token list -> token list
  val priority : int  (* 决定插件执行顺序 *)
end
```

插件可以实现各种功能：
- 专有名词识别
- 数字规范化
- 表情符号处理
- 领域特定术语处理

### 2.1.4 性能与准确性权衡

**性能优化策略**：
1. **预编译模式**：将正则表达式、词典编译成高效数据结构
2. **批处理**：减少函数调用开销
3. **并行处理**：文档级并行、段落级并行
4. **缓存机制**：常见词汇的分词结果缓存

**准确性考虑**：
1. **歧义消解**：「研究生命起源」→「研究/生命/起源」vs「研究生/命/起源」
2. **新词识别**：社交媒体中的网络新词
3. **多语言混合**：中英混排文本的处理
4. **上下文相关**：同一词在不同上下文中的不同切分方式

## 2.2 语言检测模块的架构

准确的语言检测是多语言搜索引擎的基础。错误的语言识别会导致使用错误的分词器和查询处理策略。

### 2.2.1 统计方法 vs 神经网络方法

**N-gram 统计方法**：
- 字符级 N-gram 频率分布
- 贝叶斯分类器
- 优点：速度快、内存占用小
- 缺点：对短文本准确率低

**神经网络方法**：
- 字符级 CNN/RNN
- 预训练语言模型微调
- 优点：准确率高、鲁棒性强
- 缺点：计算开销大、需要 GPU

**混合策略**：
```ocaml
module type LANGUAGE_DETECTOR = sig
  type confidence = float  (* 0.0 - 1.0 *)
  type detection_result = {
    language: string;      (* ISO 639-1 代码 *)
    confidence: confidence;
    alternatives: (string * confidence) list;
  }
  
  val detect : string -> detection_result
  val detect_mixed : string -> (int * int * string) list  (* 起始位置、结束位置、语言 *)
end
```

### 2.2.2 多语言文档处理

现实中的文档经常包含多种语言：
- 技术文档中的代码片段
- 学术论文中的引用
- 社交媒体中的多语言对话

处理策略：
1. **滑动窗口检测**：检测文档不同部分的语言
2. **层次化检测**：先检测主要语言，再检测局部语言
3. **上下文增强**：利用周围文本提高检测准确性

### 2.2.3 增量语言检测

对于流式文本或大文档，需要增量检测机制：

```ocaml
module type INCREMENTAL_DETECTOR = sig
  type state
  val init : unit -> state
  val update : state -> string -> state * detection_result option
  val finalize : state -> detection_result
end
```

关键设计点：
- 最小检测单元：需要多少文本才能可靠检测
- 状态转换：语言切换的检测
- 内存管理：限制状态大小

### 2.2.4 置信度评估系统

不仅要检测语言，还要评估检测的可靠性：

**置信度因素**：
- 文本长度：越长越可靠
- 特征明显程度：独特字符集、词汇
- 模型一致性：多个模型的一致程度
- 歧义程度：相似语言间的区分难度

**应用场景**：
- 低置信度时的降级策略
- 人工审核触发条件
- 索引策略调整

## 2.3 文本规范化的设计模式

文本规范化将不同形式的相同概念统一表示，提高搜索的召回率。

### 2.3.1 Unicode 规范化策略

Unicode 提供了多种规范化形式（NFC、NFD、NFKC、NFKD）：

**选择原则**：
- **NFC**（标准等价合成）：保持视觉一致性，适合显示
- **NFKC**（兼容等价合成）：最大化匹配可能，适合搜索
- **自定义规范化**：针对特定需求的扩展

**实现考虑**：
```ocaml
module type NORMALIZER = sig
  type normalization_form = NFC | NFD | NFKC | NFKD | Custom of string
  val normalize : normalization_form -> string -> string
  val register_custom : string -> (string -> string) -> unit
end
```

### 2.3.2 大小写折叠决策

大小写处理看似简单，实则充满细节：

**基本策略**：
- 全部小写：最简单，但损失信息
- 保留原始 + 小写索引：占用更多空间
- 智能折叠：根据上下文决定

**特殊情况**：
- 缩写词：「IBM」vs「ibm」
- 专有名词：「iPhone」vs「iphone」
- 编程语言：「IOException」的驼峰命名
- 土耳其语 i/I 问题

### 2.3.3 标点符号处理

标点符号的处理影响查询的精确性：

**处理策略**：
1. **完全移除**：简单但可能损失语义
2. **转换为空格**：保留词边界
3. **选择性保留**：如 C++、.NET
4. **特殊标记**：将标点转为特殊词项

**设计模式**：
```ocaml
module type PUNCTUATION_HANDLER = sig
  type action = Remove | Replace of string | Keep | Tokenize
  val classify : char -> action
  val register_pattern : string -> action -> unit
end
```

### 2.3.4 同义词与变体处理

搜索引擎需要理解词汇的不同表现形式：

**处理层次**：
1. **字符级变体**：「café」→「cafe」
2. **拼写变体**：「color」⟷「colour」
3. **词形变化**：「running」→「run」
4. **语义同义词**：「car」⟷「automobile」

**实现策略**：
- 索引时扩展：增加索引大小，但查询快
- 查询时扩展：索引小，但查询复杂
- 混合方法：常见同义词索引时处理，长尾查询时处理

## 2.4 N-gram 索引的存储策略

N-gram 索引支持模糊匹配、拼写纠正和子串搜索，是现代搜索引擎的重要组件。

### 2.4.1 N-gram 大小选择

N 值的选择影响索引大小和查询效果：

**常见配置**：
- **Unigram (N=1)**：单字符，用于 CJK 语言基础索引
- **Bigram (N=2)**：平衡索引大小和查询效果
- **Trigram (N=3)**：西方语言的常见选择
- **Variable-N**：根据词长动态选择

**选择因素**：
- 语言特性：中文倾向于 bigram，英文倾向于 trigram
- 查询类型：子串搜索需要较小的 N
- 存储限制：N 越大，索引膨胀越严重
- 查询性能：N 越大，候选集越精确

### 2.4.2 存储格式设计

高效的存储格式需要平衡空间和查询性能：

**倒排索引结构**：
```ocaml
module type NGRAM_INDEX = sig
  type doc_id = int
  type position = int
  type posting = {
    doc_id: doc_id;
    positions: position list;
    frequency: int;
  }
  
  type index = (string, posting list) Hashtbl.t
  val build : (doc_id * string) list -> index
  val merge : index -> index -> index
end
```

**压缩技术**：
1. **差分编码**：存储文档 ID 差值
2. **可变长编码**：VInt、Simple9
3. **位图索引**：高频 N-gram 的位图表示
4. **前缀树**：共享公共前缀

### 2.4.3 查询时重建策略

有时不存储所有 N-gram，而是查询时动态生成：

**策略比较**：
- **完全存储**：查询快，空间大
- **按需生成**：空间小，查询慢
- **混合策略**：高频存储，低频生成

**实现考虑**：
```ocaml
module type NGRAM_GENERATOR = sig
  val generate : int -> string -> string list
  val generate_positional : int -> string -> (string * int) list
  val estimate_selectivity : string -> float  (* 估计 N-gram 的选择性 *)
end
```

### 2.4.4 优化技术

**索引优化**：
1. **跳表结构**：加速长posting list的遍历
2. **缓存策略**：LRU 缓存热门 N-gram
3. **分片索引**：将索引分布到多台机器
4. **布隆过滤器**：快速判断 N-gram 是否存在

**查询优化**：
1. **短路求值**：选择性最高的 N-gram 优先
2. **并行查询**：多个 N-gram 并行查找
3. **近似匹配**：编辑距离内的 N-gram 扩展
4. **上下文过滤**：利用语言模型过滤不可能的组合

## 本章小结

文本处理管道是搜索引擎的基础设施，其设计直接影响整个系统的性能和效果。关键要点：

1. **分词器设计**：模块化、可扩展的架构支持多语言和特殊需求
2. **语言检测**：混合统计和深度学习方法，提供可靠的置信度评估  
3. **文本规范化**：平衡信息保留和匹配宽容度
4. **N-gram 索引**：根据应用场景选择合适的粒度和存储策略

核心设计原则：
- **模块化**：通过清晰的接口定义实现组件解耦
- **可扩展性**：插件机制支持新语言和新功能
- **性能意识**：在准确性和效率之间找到平衡点
- **容错性**：优雅处理异常输入和边界情况

## 练习题

### 基础题

1. **分词器接口设计**
   设计一个支持词性标注的分词器接口，考虑如何与现有 TOKENIZER 接口集成。
   
   <details>
   <summary>Hint: 考虑扩展 token 类型或使用属性字段</summary>
   </details>
   
   <details>
   <summary>参考答案</summary>
   
   可以通过扩展 token 类型中的 attributes 字段来支持词性标注：
   ```ocaml
   type pos_tag = Noun | Verb | Adjective | (* ... *)
   
   module type POS_TOKENIZER = sig
     include TOKENIZER
     val tokenize_with_pos : config -> string -> (token * pos_tag) list
   end
   ```
   
   或者将词性信息存储在 attributes 中：
   ```ocaml
   let tokenize_with_pos config text =
     let tokens = tokenize config text in
     List.map (fun token ->
       let pos = detect_pos token.surface in
       { token with attributes = ("pos", string_of_pos pos) :: token.attributes }
     ) tokens
   ```
   </details>

2. **语言检测优化**
   给定一个包含 80% 英文和 20% 中文的文档，设计一个高效的检测策略，最小化处理时间。
   
   <details>
   <summary>Hint: 考虑采样策略和早期终止条件</summary>
   </details>
   
   <details>
   <summary>参考答案</summary>
   
   优化策略：
   1. 随机采样文档的多个片段（如 5 个 100 字符片段）
   2. 并行检测各片段的语言
   3. 如果 3 个以上片段检测为同一语言，提前返回结果
   4. 对剩余 20% 使用滑动窗口检测混合语言边界
   5. 缓存检测结果，相似文档可复用
   </details>

3. **Unicode 规范化场景**
   列举三个必须使用 NFKC 而不是 NFC 的搜索场景，并解释原因。
   
   <details>
   <summary>Hint: 考虑全角/半角、连字等情况</summary>
   </details>
   
   <details>
   <summary>参考答案</summary>
   
   1. **全角半角统一**：用户输入「ＡＢＣ」应该能搜到「ABC」
   2. **连字处理**：「ﬁ」（连字）应该匹配「fi」（两个字符）
   3. **上标下标**：「x²」应该能匹配「x2」的搜索
   
   这些场景下，NFKC 将视觉相似但编码不同的字符统一，提高搜索召回率。
   </details>

### 挑战题

4. **多语言分词器组合**
   设计一个自动选择合适分词器的系统，处理包含中英日韩的混合文本。考虑分词边界冲突的处理。
   
   <details>
   <summary>Hint: 考虑基于置信度的投票机制</summary>
   </details>
   
   <details>
   <summary>参考答案</summary>
   
   设计方案：
   1. **语言段识别**：使用滑动窗口识别连续的单语言段
   2. **分词器调度**：
      ```ocaml
      module MultilingualTokenizer = struct
        let tokenizers = Hashtbl.create 10
        
        let tokenize text =
          let segments = detect_language_segments text in
          List.concat_map (fun (start, end_, lang) ->
            let tokenizer = Hashtbl.find tokenizers lang in
            let substring = String.sub text start (end_ - start) in
            let tokens = tokenizer.tokenize substring in
            adjust_positions tokens start
          ) segments
      end
      ```
   3. **边界处理**：在语言切换点使用多个分词器，选择置信度最高的结果
   4. **后处理统一**：确保位置信息的连续性
   </details>

5. **增量 N-gram 索引更新**
   设计一个支持实时更新的 N-gram 索引结构，要求更新操作的时间复杂度为 O(log n)。
   
   <details>
   <summary>Hint: 考虑 LSM-tree 或 B+ tree 的变体</summary>
   </details>
   
   <details>
   <summary>参考答案</summary>
   
   使用 LSM-tree 风格的设计：
   1. **内存层**：使用平衡树（如红黑树）存储最新更新
   2. **磁盘层**：多层只读的有序索引文件
   3. **合并策略**：
      - 内存层达到阈值时，排序写入 L0 层
      - 定期合并相邻层，保持对数级层数
   4. **查询路径**：
      - 并行查询所有层
      - 使用布隆过滤器加速不存在的 N-gram
   5. **优化**：
      - 压缩存储的 posting list
      - 延迟删除，批量处理
   </details>

6. **自适应文本规范化**
   设计一个能够从查询日志中学习的文本规范化系统，自动发现新的规范化规则。
   
   <details>
   <summary>Hint: 考虑点击日志中的查询-文档对</summary>
   </details>
   
   <details>
   <summary>参考答案</summary>
   
   学习系统架构：
   1. **数据收集**：
      - 记录查询词和点击文档中的对应词
      - 统计查询改写模式
   2. **模式发现**：
      - 使用编辑距离聚类相似查询
      - 提取字符级转换规则（如 oe→ö）
      - 发现领域特定缩写（ML→Machine Learning）
   3. **规则生成**：
      ```ocaml
      type normalization_rule = {
        pattern: regex;
        replacement: string;
        confidence: float;
        frequency: int;
      }
      ```
   4. **在线学习**：
      - A/B 测试新规则的效果
      - 根据点击率和停留时间调整规则权重
   5. **人工审核**：高频低置信度规则触发人工审核
   </details>

7. **查询意图感知的 N-gram 选择**
   设计一个系统，根据查询意图动态选择最优的 N-gram 大小。例如，精确查询使用大 N，模糊查询使用小 N。
   
   <details>
   <summary>Hint: 考虑查询分类和历史效果反馈</summary>
   </details>
   
   <details>
   <summary>参考答案</summary>
   
   动态选择系统：
   1. **查询分类器**：
      - 短查询（1-2词）→ 倾向小 N
      - 长尾查询 → 倾向大 N  
      - 包含特殊字符 → 混合 N
   2. **意图识别**：
      - 导航型查询：使用大 N 提高精确度
      - 探索型查询：使用小 N 提高召回
      - 纠错型查询：使用 bigram 支持模糊匹配
   3. **自适应策略**：
      ```ocaml
      let select_ngram_size query =
        let base_size = classify_query query in
        let historical_performance = 
          get_click_through_rate query base_size in
        if historical_performance < threshold then
          try_alternative_sizes query
        else base_size
      ```
   4. **效果追踪**：记录不同 N 值的查询效果，持续优化
   </details>

8. **分布式语言检测的一致性**
   在分布式环境中，如何确保不同节点对同一文档的语言检测结果一致？设计一个协调机制。
   
   <details>
   <summary>Hint: 考虑模型版本管理和确定性算法</summary>
   </details>
   
   <details>
   <summary>参考答案</summary>
   
   一致性保证机制：
   1. **确定性设计**：
      - 使用确定性的特征提取（固定随机种子）
      - 模型推理避免浮点数累积误差
      - 统一的文本预处理流程
   2. **版本管理**：
      ```ocaml
      type model_version = {
        version: string;
        checksum: string;
        features: feature_spec;
      }
      ```
   3. **分布式协调**：
      - 中心化模型分发，确保版本一致
      - 使用内容哈希作为缓存键
      - 检测结果包含模型版本信息
   4. **冲突解决**：
      - 多数投票机制
      - 基于置信度的加权投票
      - 回退到确定性规则（如基于字符集）
   5. **监控告警**：检测不一致率，超过阈值触发模型同步
   </details>

## 常见陷阱与错误 (Gotchas)

### 分词相关
1. **边界歧义处理不当**：「研究生命」的分词边界
   - 错误：硬编码规则
   - 正确：结合上下文和词频统计

2. **特殊字符遗漏**：URL、邮箱被错误分词
   - 错误：简单的空格分割
   - 正确：预先识别特殊模式

3. **性能退化**：复杂正则导致的性能问题
   - 错误：嵌套的回溯正则
   - 正确：使用 DFA 或预编译模式

### 语言检测相关
4. **短文本检测失败**：单词级别的语言检测不准确
   - 错误：强制检测所有文本
   - 正确：设置最小长度阈值，短文本使用默认语言

5. **代码片段干扰**：技术文档中的代码被识别为英语
   - 错误：将所有 ASCII 文本视为英语
   - 正确：先识别并排除代码块

### 规范化相关
6. **过度规范化**：「C++」被规范化为「c」
   - 错误：盲目移除所有标点
   - 正确：维护特殊词汇白名单

7. **Unicode 陷阱**：组合字符的处理不一致
   - 错误：字节级别的处理
   - 正确：使用标准 Unicode 库

### N-gram 相关
8. **索引膨胀失控**：N-gram 索引比原文还大
   - 错误：索引所有可能的 N-gram
   - 正确：频率阈值过滤 + 停用词处理

9. **位置信息丢失**：N-gram 的原始位置无法还原
   - 错误：只存储 N-gram 到文档的映射
   - 正确：保留位置信息用于短语查询

10. **查询时间过长**：高频 N-gram 导致大量候选文档
    - 错误：线性扫描所有匹配文档
    - 正确：使用跳表或分段处理

## 最佳实践检查清单

### 设计阶段
- [ ] 是否定义了清晰的模块接口？
- [ ] 是否考虑了目标语言的特性？
- [ ] 是否设计了扩展机制？
- [ ] 是否有性能基准和目标？

### 实现阶段
- [ ] 分词器是否处理了所有 Unicode 类别？
- [ ] 语言检测是否有置信度阈值？
- [ ] 规范化是否保留了原始信息？
- [ ] N-gram 索引是否有大小限制？

### 测试阶段
- [ ] 是否测试了多语言混合文本？
- [ ] 是否测试了边界条件（空文本、超长文本）？
- [ ] 是否测试了特殊字符和表情符号？
- [ ] 是否进行了性能压力测试？

### 优化阶段
- [ ] 是否识别了性能瓶颈？
- [ ] 是否实现了适当的缓存策略？
- [ ] 是否支持增量更新？
- [ ] 是否有降级策略？

### 监控阶段
- [ ] 是否记录了分词失败案例？
- [ ] 是否追踪语言检测准确率？
- [ ] 是否监控索引大小增长？
- [ ] 是否有异常模式检测？

### 维护阶段
- [ ] 是否有添加新语言的流程？
- [ ] 是否能够更新规范化规则？
- [ ] 是否支持 A/B 测试？
- [ ] 是否有回滚机制？