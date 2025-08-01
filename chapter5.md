# Chapter 5: 分布式爬虫架构

构建一个能够处理数十亿网页的爬虫系统需要精心设计的分布式架构。本章将深入探讨如何设计一个高效、可扩展、礼貌的爬虫系统。我们将从并发模型的选择开始，逐步展开到 URL 调度、去重策略、以及现代 Web 应用所需的 JavaScript 渲染服务。通过 OCaml 的类型系统，我们将定义清晰的模块接口，确保各组件间的协作既高效又可靠。

## 5.1 爬虫系统架构概览

### 爬虫的核心任务与挑战

Web 爬虫系统的核心任务看似简单：下载网页、提取链接、继续爬取。然而在大规模场景下，这个简单的循环面临着诸多挑战：

**规模挑战**：
- 数十亿 URL 的管理与调度
- TB 级别的日下载量
- 毫秒级的爬取决策

**礼貌性要求**：
- 遵守 robots.txt 协议
- 控制对单一域名的访问频率
- 避免对目标服务器造成过大压力

**内容多样性**：
- 静态 HTML 页面
- JavaScript 渲染的单页应用
- API 接口与结构化数据
- 多媒体内容的处理

**质量控制**：
- 去重机制避免重复爬取
- 内容更新检测
- 垃圾页面识别与过滤

### 分布式架构的必要性

单机爬虫在面对互联网规模时很快达到瓶颈：

```ocaml
(* 单机爬虫的局限性 *)
module SingleNodeLimits = struct
  type bottleneck = 
    | NetworkBandwidth of int  (* Mbps *)
    | DiskIO of int           (* IOPS *)
    | CPUProcessing of float   (* cores *)
    | MemoryCapacity of int    (* GB *)
    
  (* 典型单机配置的理论极限 *)
  let typical_limits = {
    bandwidth = 10_000;        (* 10 Gbps *)
    disk_ops = 100_000;        (* NVMe SSD *)
    cpu_cores = 64.0;          
    memory = 512;              (* GB *)
  }
  
  (* 假设每个页面 50KB，处理需要 10ms *)
  let max_pages_per_day = 
    let bandwidth_limit = (10_000 * 86400) / (50 * 8) in
    let cpu_limit = (64.0 * 86400 * 1000) / 10.0 in
    min bandwidth_limit cpu_limit  (* ~216M pages/day *)
end
```

分布式架构通过水平扩展突破这些限制：

**并行下载**：多节点同时爬取不同的 URL 集合
**分布式存储**：爬取的内容分散存储在多个节点
**协调服务**：中心化或去中心化的任务分配
**弹性扩展**：根据负载动态增减爬虫节点

### 系统组件间的协作模式

现代分布式爬虫系统通常包含以下核心组件：

```ocaml
(* 爬虫系统的核心组件接口 *)
module type CrawlerComponents = sig
  (* URL 调度器 *)
  module Scheduler : sig
    type t
    type priority = High | Normal | Low | Custom of int
    
    val create : config:SchedulerConfig.t -> t
    val next_batch : t -> max_urls:int -> (Url.t * priority) list
    val mark_completed : t -> Url.t -> unit
    val mark_failed : t -> Url.t -> reason:string -> unit
  end
  
  (* 下载器 *)
  module Downloader : sig
    type t
    type result = 
      | Success of { content: string; 
                     headers: (string * string) list; 
                     status_code: int }
      | Failure of { reason: string; 
                     retry_after: float option }
    
    val create : config:DownloaderConfig.t -> t
    val fetch : t -> Url.t -> result Lwt.t
    val fetch_batch : t -> Url.t list -> result list Lwt.t
  end
  
  (* 解析器 *)
  module Parser : sig
    type t
    type parsed_doc = {
      urls: Url.t list;
      metadata: (string * string) list;
      content: structured_content;
    }
    
    val create : unit -> t
    val parse_html : t -> string -> parsed_doc
    val parse_javascript : t -> string -> render_first:bool -> parsed_doc
  end
  
  (* 去重器 *)
  module Deduplicator : sig
    type t
    type dedup_strategy = 
      | UrlExact 
      | UrlNormalized
      | ContentHash
      | ContentSimilarity of float  (* threshold *)
    
    val create : strategy:dedup_strategy -> capacity:int -> t
    val is_duplicate : t -> Url.t -> content:string option -> bool
    val add : t -> Url.t -> content:string option -> unit
  end
  
  (* 存储层 *)
  module Storage : sig
    type t
    type doc_id = string
    
    val create : config:StorageConfig.t -> t
    val store : t -> url:Url.t -> content:string -> 
                metadata:(string * string) list -> doc_id
    val retrieve : t -> doc_id -> (string * (string * string) list) option
  end
end
```

### 数据流与控制流设计

爬虫系统中的数据流设计直接影响系统的性能和可靠性：

**Push vs Pull 模式**：
- Push：调度器主动推送任务给爬虫节点
- Pull：爬虫节点主动从调度器拉取任务
- Hybrid：结合两者优势的混合模式

```ocaml
(* 不同的任务分配模式 *)
module TaskDistribution = struct
  (* Push 模式：调度器主导 *)
  module PushMode = struct
    type worker_state = Idle | Busy of int  (* tasks in queue *)
    
    let distribute scheduler workers =
      let idle_workers = 
        workers |> List.filter (fun w -> w.state = Idle) in
      let tasks = Scheduler.next_batch scheduler 
                    ~max_urls:(List.length idle_workers * 10) in
      List.iter2 (fun worker task_batch ->
        Worker.assign worker task_batch
      ) idle_workers (split_tasks tasks)
  end
  
  (* Pull 模式：工作节点主导 *)
  module PullMode = struct
    let worker_loop worker scheduler =
      let rec loop () =
        let batch_size = calculate_batch_size worker in
        let tasks = Scheduler.next_batch scheduler ~max_urls:batch_size in
        let results = Worker.process worker tasks in
        report_results scheduler results;
        loop ()
      in
      loop ()
  end
  
  (* Hybrid 模式：自适应分配 *)
  module HybridMode = struct
    type mode = Push | Pull
    
    let decide_mode worker_load scheduler_queue_size =
      match worker_load, scheduler_queue_size with
      | high, _ -> Pull  (* 高负载时工作节点主动拉取 *)
      | _, low -> Push   (* 队列较少时调度器主动推送 *)
      | _ -> Pull        (* 默认拉取模式 *)
  end
end
```

**背压控制机制**：

当下游处理速度跟不上上游产生速度时，需要背压机制防止系统过载：

```ocaml
(* 背压控制策略 *)
module BackpressureControl = struct
  type strategy =
    | DropOldest        (* 丢弃最旧的任务 *)
    | DropNewest        (* 丢弃最新的任务 *)
    | SlowDown          (* 降低上游生产速度 *)
    | BufferWithLimit   (* 有限缓冲区 *)
    | Adaptive of {
        threshold: float;
        scale_factor: float;
      }
  
  let apply_backpressure strategy queue_size max_size current_rate =
    match strategy with
    | SlowDown when queue_size > int_of_float (0.8 *. float max_size) ->
        { new_rate = current_rate *. 0.5; 
          action = `ReduceRate }
    | BufferWithLimit when queue_size >= max_size ->
        { new_rate = 0.0; 
          action = `PauseProduction }
    | Adaptive { threshold; scale_factor } ->
        let utilization = float queue_size /. float max_size in
        if utilization > threshold then
          { new_rate = current_rate *. scale_factor;
            action = `AdjustRate }
        else
          { new_rate = current_rate; 
            action = `Continue }
    | _ -> 
        { new_rate = current_rate; 
          action = `Continue }
end
```

**错误处理与重试策略**：

分布式环境中的错误处理需要考虑多种失败场景：

```ocaml
(* 错误处理与重试机制 *)
module ErrorHandling = struct
  type error_type =
    | NetworkTimeout
    | DNSFailure  
    | HTTPError of int  (* status code *)
    | ParseError of string
    | RateLimited of float  (* retry after seconds *)
    | Temporary of string
    | Permanent of string
  
  type retry_policy = {
    max_attempts: int;
    base_delay: float;  (* seconds *)
    max_delay: float;
    exponential_backoff: bool;
    jitter: float;  (* 0.0 to 1.0 *)
  }
  
  let default_policy = {
    max_attempts = 3;
    base_delay = 1.0;
    max_delay = 60.0;
    exponential_backoff = true;
    jitter = 0.1;
  }
  
  let should_retry error attempt policy =
    if attempt >= policy.max_attempts then false
    else match error with
    | NetworkTimeout | DNSFailure | Temporary _ -> true
    | HTTPError code -> code >= 500 || code = 429
    | RateLimited _ -> true
    | ParseError _ | Permanent _ -> false
  
  let calculate_delay error attempt policy =
    let base = match error with
      | RateLimited delay -> delay
      | _ -> 
        if policy.exponential_backoff then
          policy.base_delay *. (2.0 ** float (attempt - 1))
        else
          policy.base_delay
    in
    let delay = min base policy.max_delay in
    let jitter = Random.float (delay *. policy.jitter) in
    delay +. jitter
end
```

**监控与可观测性**：

分布式爬虫系统需要全面的监控来保证系统健康：

```ocaml
(* 监控指标定义 *)
module Metrics = struct
  type crawler_metrics = {
    (* 吞吐量指标 *)
    urls_crawled_per_second: float;
    bytes_downloaded_per_second: float;
    pages_parsed_per_second: float;
    
    (* 延迟指标 *)
    fetch_latency_p50: float;
    fetch_latency_p99: float;
    parse_latency_p50: float;
    
    (* 错误率 *)
    error_rate: float;
    timeout_rate: float;
    
    (* 资源使用 *)
    cpu_usage: float;
    memory_usage: float;
    network_usage: float;
    
    (* 队列状态 *)
    url_queue_size: int;
    processing_queue_size: int;
    
    (* 去重效率 *)
    duplicate_ratio: float;
  }
  
  let collect_metrics system =
    let current_time = Unix.time () in
    {
      urls_crawled_per_second = 
        float (system.stats.urls_crawled - system.stats.prev_urls_crawled) /.
        (current_time -. system.stats.prev_time);
      (* ... 其他指标计算 ... *)
    }
  
  let alert_thresholds = {
    error_rate_threshold = 0.05;  (* 5% *)
    latency_p99_threshold = 5.0;  (* 5 seconds *)
    queue_size_threshold = 1_000_000;
  }
end
```

## 5.2 并发模型与 OCaml Effects

### 并发爬取的设计模式

高性能爬虫系统的核心在于充分利用 I/O 并发。传统的线程模型在处理数万个并发连接时会遇到上下文切换开销大、内存占用高等问题。现代爬虫系统通常采用异步 I/O 模型：

```ocaml
(* 传统线程模型 vs 异步模型的对比 *)
module ConcurrencyModels = struct
  (* 线程池模型 *)
  module ThreadPool = struct
    type config = {
      num_threads: int;
      queue_size: int;
      thread_stack_size: int;  (* KB *)
    }
    
    (* 每个线程独立处理一个 URL *)
    let worker_thread queue =
      while true do
        let url = BlockingQueue.pop queue in
        let content = Http.blocking_get url in
        process_page content
      done
    
    (* 资源占用分析 *)
    let resource_usage config num_connections =
      let memory_per_thread = config.thread_stack_size in
      let total_memory = config.num_threads * memory_per_thread in
      let max_concurrent = min config.num_threads num_connections in
      { memory_mb = total_memory / 1024;
        max_concurrent_connections = max_concurrent;
        context_switches_per_sec = max_concurrent * 2; }
  end
  
  (* 异步 I/O 模型 *)
  module AsyncIO = struct
    type config = {
      max_concurrent: int;
      buffer_size: int;
    }
    
    (* 使用 Lwt 的异步爬取 *)
    let async_crawler urls config =
      let semaphore = Lwt_pool.create config.max_concurrent (fun () -> Lwt.return_unit) in
      urls |> Lwt_list.iter_p (fun url ->
        Lwt_pool.use semaphore (fun () ->
          let%lwt content = Http.async_get url in
          process_page_async content
        )
      )
    
    (* 资源效率显著提升 *)
    let resource_usage config num_connections =
      { memory_mb = config.buffer_size * config.max_concurrent / 1024;
        max_concurrent_connections = config.max_concurrent;
        context_switches_per_sec = 0; (* 事件驱动，无需切换 *) }
  end
end
```

### OCaml Effects 系统的应用

OCaml 5.0 引入的 Effects 系统为并发编程提供了新的抽象层次，允许我们编写直接风格（direct-style）的代码，同时保持异步执行的性能优势：

```ocaml
(* Effects 系统在爬虫中的应用 *)
module EffectsCrawler = struct
  (* 定义爬虫相关的 effects *)
  type _ Effect.t +=
    | Fetch : Url.t -> string Effect.t
    | Parse : string -> Document.t Effect.t
    | Store : Document.t -> unit Effect.t
    | Delay : float -> unit Effect.t
    | Spawn : (unit -> unit) -> unit Effect.t
  
  (* 直接风格的爬虫逻辑 *)
  let crawl_page url =
    let content = Effect.perform (Fetch url) in
    let doc = Effect.perform (Parse content) in
    Effect.perform (Store doc);
    (* 提取链接并继续爬取 *)
    doc.links |> List.iter (fun link ->
      Effect.perform (Spawn (fun () -> crawl_page link))
    )
  
  (* Effect handler 实现并发调度 *)
  let run_crawler initial_urls config =
    let module Queue = Domainslib.Chan in
    let task_queue = Queue.make () in
    let active_tasks = ref 0 in
    
    (* 主调度器 *)
    let rec scheduler () =
      Effect.Deep.match_with crawl_page initial_urls
        { retc = (fun () -> ());
          exnc = raise;
          effc = fun (type a) (eff : a Effect.t) ->
            match eff with
            | Fetch url -> Some (fun (k : (a, _) continuation) ->
                incr active_tasks;
                (* 异步获取，不阻塞调度器 *)
                Lwt.async (fun () ->
                  let%lwt content = Http.async_get url in
                  Queue.send task_queue (fun () ->
                    decr active_tasks;
                    Effect.Deep.continue k content
                  );
                  Lwt.return_unit
                );
                scheduler ()  (* 继续调度其他任务 *)
              )
            | Spawn task -> Some (fun k ->
                Queue.send task_queue task;
                Effect.Deep.continue k ()
              )
            | _ -> None
        }
    in
    
    (* 工作线程池 *)
    let worker_loop () =
      while true do
        let task = Queue.recv task_queue in
        task ()
      done
    in
    
    (* 启动工作线程 *)
    let domains = Array.init config.num_workers (fun _ ->
      Domain.spawn worker_loop
    ) in
    
    scheduler ();
    Array.iter Domain.join domains
end
```

### 协程与线程的权衡

不同并发原语在爬虫场景下有不同的适用性：

```ocaml
(* 并发原语的选择策略 *)
module ConcurrencyStrategy = struct
  type concurrency_primitive =
    | SystemThread     (* 系统线程 *)
    | LightweightThread (* Lwt/Async *)
    | Fiber           (* Effect-based *)
    | Domain          (* OCaml 5 domains *)
  
  type workload_characteristics = {
    io_bound_ratio: float;      (* 0.0 - 1.0 *)
    avg_task_duration: float;   (* milliseconds *)
    memory_per_task: int;       (* bytes *)
    cpu_intensive: bool;
  }
  
  (* 根据工作负载特征选择并发原语 *)
  let choose_primitive workload =
    match workload with
    | { io_bound_ratio; _ } when io_bound_ratio > 0.8 ->
        LightweightThread  (* I/O 密集型适合轻量级线程 *)
    | { cpu_intensive = true; _ } ->
        Domain  (* CPU 密集型任务使用 Domain *)
    | { avg_task_duration; _ } when avg_task_duration < 1.0 ->
        Fiber  (* 短任务适合 Effects *)
    | _ ->
        SystemThread  (* 默认使用系统线程 *)
  
  (* 混合并发模型 *)
  module Hybrid = struct
    (* Domains 处理 CPU 密集型任务 + Fibers 处理 I/O *)
    type task =
      | IOTask of (unit -> string)
      | CPUTask of (string -> Document.t)
    
    let create_pipeline num_domains =
      let io_queue = Queue.create () in
      let cpu_queue = Queue.create () in
      
      (* I/O Domain: 使用 Effects 处理并发下载 *)
      let io_domain = Domain.spawn (fun () ->
        let rec process_io () =
          match Queue.pop io_queue with
          | IOTask task ->
              let content = 
                Effect.Deep.match_with task ()
                  { retc = Fun.id;
                    exnc = raise;
                    effc = fun (type a) (eff : a Effect.t) ->
                      (* Handle Fetch effects *)
                      None
                  } in
              Queue.push (CPUTask (fun _ -> parse content)) cpu_queue;
              process_io ()
        in
        process_io ()
      ) in
      
      (* CPU Domains: 并行解析 *)
      let cpu_domains = Array.init num_domains (fun _ ->
        Domain.spawn (fun () ->
          let rec process_cpu () =
            match Queue.pop cpu_queue with
            | CPUTask task ->
                let doc = task "" in
                store_document doc;
                process_cpu ()
          in
          process_cpu ()
        )
      ) in
      
      { io_domain; cpu_domains }
  end
end
```

### 背压控制与流量管理

在高并发场景下，背压控制是保证系统稳定性的关键：

```ocaml
(* 流量控制机制 *)
module FlowControl = struct
  (* 令牌桶算法 *)
  module TokenBucket = struct
    type t = {
      capacity: int;
      refill_rate: float;  (* tokens per second *)
      mutable tokens: float;
      mutable last_refill: float;
      mutex: Mutex.t;
    }
    
    let create ~capacity ~refill_rate = {
      capacity;
      refill_rate;
      tokens = float capacity;
      last_refill = Unix.time ();
      mutex = Mutex.create ();
    }
    
    let try_acquire bucket n =
      Mutex.lock bucket.mutex;
      let now = Unix.time () in
      let elapsed = now -. bucket.last_refill in
      let new_tokens = min 
        (bucket.tokens +. elapsed *. bucket.refill_rate)
        (float bucket.capacity) in
      bucket.tokens <- new_tokens;
      bucket.last_refill <- now;
      
      if bucket.tokens >= float n then begin
        bucket.tokens <- bucket.tokens -. float n;
        Mutex.unlock bucket.mutex;
        true
      end else begin
        Mutex.unlock bucket.mutex;
        false
      end
    
    (* Effects-based 接口 *)
    let with_rate_limit bucket f =
      let rec wait_and_retry () =
        if try_acquire bucket 1 then
          f ()
        else begin
          Effect.perform (Delay 0.1);
          wait_and_retry ()
        end
      in
      wait_and_retry ()
  end
  
  (* 自适应并发控制 *)
  module AdaptiveConcurrency = struct
    type metrics = {
      success_rate: float;
      avg_latency: float;
      error_rate: float;
    }
    
    type controller = {
      mutable current_limit: int;
      min_limit: int;
      max_limit: int;
      mutable metrics_window: metrics list;
      adjustment_interval: float;
    }
    
    (* AIMD (Additive Increase Multiplicative Decrease) 算法 *)
    let adjust_limit controller current_metrics =
      let should_increase = 
        current_metrics.success_rate > 0.95 &&
        current_metrics.avg_latency < 1.0 in
      
      let should_decrease =
        current_metrics.error_rate > 0.05 ||
        current_metrics.avg_latency > 5.0 in
      
      match should_increase, should_decrease with
      | true, false ->
          (* 加性增加 *)
          controller.current_limit <- 
            min (controller.current_limit + 10) controller.max_limit
      | false, true ->
          (* 乘性减少 *)
          controller.current_limit <-
            max (controller.current_limit * 8 / 10) controller.min_limit
      | _ -> ()
    
    (* 基于 Effects 的并发限制 *)
    type _ Effect.t +=
      | AcquireSlot : unit Effect.t
      | ReleaseSlot : unit Effect.t
    
    let with_concurrency_limit controller f =
      Effect.perform AcquireSlot;
      try
        let result = f () in
        Effect.perform ReleaseSlot;
        result
      with e ->
        Effect.perform ReleaseSlot;
        raise e
  end
  
  (* 分层流控：域名级别 + 全局级别 *)
  module HierarchicalRateLimit = struct
    type config = {
      global_rps: float;        (* 全局每秒请求数 *)
      per_domain_rps: float;    (* 每个域名的限制 *)
      burst_capacity: int;      (* 突发容量 *)
    }
    
    type t = {
      global_bucket: TokenBucket.t;
      domain_buckets: (string, TokenBucket.t) Hashtbl.t;
      config: config;
    }
    
    let create config = {
      global_bucket = TokenBucket.create 
        ~capacity:config.burst_capacity
        ~refill_rate:config.global_rps;
      domain_buckets = Hashtbl.create 1000;
      config;
    }
    
    let acquire_permission t url =
      let domain = Url.domain url in
      let domain_bucket = 
        match Hashtbl.find_opt t.domain_buckets domain with
        | Some bucket -> bucket
        | None ->
            let bucket = TokenBucket.create
              ~capacity:10
              ~refill_rate:t.config.per_domain_rps in
            Hashtbl.add t.domain_buckets domain bucket;
            bucket
      in
      (* 需要同时获得全局和域名级别的许可 *)
      TokenBucket.try_acquire t.global_bucket 1 &&
      TokenBucket.try_acquire domain_bucket 1
  end
end
```

### 并发爬虫的性能优化

```ocaml
(* 性能优化技术 *)
module PerformanceOptimization = struct
  (* 连接池管理 *)
  module ConnectionPool = struct
    type conn = {
      domain: string;
      socket: Lwt_unix.file_descr;
      last_used: float;
      requests_count: int;
    }
    
    type t = {
      connections: (string, conn Queue.t) Hashtbl.t;
      max_per_domain: int;
      max_idle_time: float;
      max_requests_per_conn: int;
    }
    
    let get_connection pool domain =
      match Hashtbl.find_opt pool.connections domain with
      | None -> None
      | Some queue ->
          try
            let conn = Queue.pop queue in
            if Unix.time () -. conn.last_used > pool.max_idle_time ||
               conn.requests_count >= pool.max_requests_per_conn then
              None  (* 连接过期或达到请求上限 *)
            else
              Some conn
          with Queue.Empty -> None
    
    (* HTTP/2 多路复用 *)
    module Http2Multiplexing = struct
      type stream_id = int
      type h2_connection = {
        domain: string;
        mutable active_streams: int;
        max_concurrent_streams: int;
        pending_requests: (stream_id * Url.t) Queue.t;
      }
      
      let can_create_stream conn =
        conn.active_streams < conn.max_concurrent_streams
      
      let multiplex_requests conn requests =
        requests |> List.filter_map (fun req ->
          if can_create_stream conn then begin
            conn.active_streams <- conn.active_streams + 1;
            Some (create_stream conn req)
          end else begin
            Queue.push req conn.pending_requests;
            None
          end
        )
    end
  end
  
  (* 零拷贝优化 *)
  module ZeroCopy = struct
    (* 使用 memory-mapped I/O 避免拷贝 *)
    let download_to_mmap url =
      let fd = Http.download_to_fd url in
      let size = (Unix.fstat fd).st_size in
      let mapped = Bigarray.Array1.map_file fd 
        Bigarray.char Bigarray.c_layout false size in
      { data = mapped; size }
    
    (* 直接从网络缓冲区解析 *)
    let parse_streaming socket =
      let buffer = Bytes.create 4096 in
      let parser = Parser.create_incremental () in
      let rec read_and_parse () =
        match Lwt_unix.read socket buffer 0 4096 with
        | 0 -> Parser.finalize parser
        | n ->
            Parser.feed parser buffer 0 n;
            let partial_results = Parser.get_results parser in
            process_partial_results partial_results;
            read_and_parse ()
      in
      read_and_parse ()
  end
  
  (* CPU 亲和性优化 *)
  module CPUAffinity = struct
    external set_affinity : int -> int -> unit = "caml_set_cpu_affinity"
    
    let optimize_for_numa ~num_crawlers ~num_parsers =
      (* 将爬虫线程绑定到网卡所在的 NUMA 节点 *)
      let network_numa_node = detect_network_numa_node () in
      for i = 0 to num_crawlers - 1 do
        let cpu = network_numa_node * 8 + i in
        set_affinity (crawler_thread_id i) cpu
      done;
      
      (* 解析线程分布到其他 NUMA 节点 *)
      for i = 0 to num_parsers - 1 do
        let numa_node = (network_numa_node + 1 + i) mod num_numa_nodes in
        let cpu = numa_node * 8 + (i mod 8) in
        set_affinity (parser_thread_id i) cpu
      done
  end
end
```

## 5.3 URL 调度器的设计模式

URL 调度器是爬虫系统的大脑，负责决定何时爬取哪个 URL。一个优秀的调度器需要平衡多个目标：最大化吞吐量、保证礼貌性、优先爬取重要页面、支持断点续爬。

### URL 优先级管理

在面对数十亿 URL 时，合理的优先级策略决定了爬虫的效率：

```ocaml
(* URL 优先级管理系统 *)
module PriorityScheduler = struct
  (* 多维度的优先级计算 *)
  type priority_factors = {
    page_rank: float;          (* 0.0 - 1.0 *)
    update_frequency: float;   (* 页面更新频率 *)
    depth_from_seed: int;      (* 距离种子页面的深度 *)
    domain_authority: float;   (* 域名权威度 *)
    content_quality: float;    (* 内容质量评分 *)
    user_interest: float;      (* 用户兴趣匹配度 *)
    freshness_requirement: float; (* 新鲜度要求 *)
  }
  
  (* 优先级计算策略 *)
  module PriorityStrategy = struct
    type strategy = 
      | Linear of (string * float) list  (* factor_name * weight *)
      | NonLinear of (priority_factors -> float)
      | Adaptive of {
          mutable weights: float array;
          learning_rate: float;
        }
    
    let default_linear_weights = [
      ("page_rank", 0.3);
      ("update_frequency", 0.2);
      ("domain_authority", 0.2);
      ("content_quality", 0.15);
      ("freshness_requirement", 0.15);
    ]
    
    let calculate_priority strategy factors =
      match strategy with
      | Linear weights ->
          List.fold_left (fun acc (factor, weight) ->
            let value = match factor with
              | "page_rank" -> factors.page_rank
              | "update_frequency" -> factors.update_frequency
              | "domain_authority" -> factors.domain_authority
              | "content_quality" -> factors.content_quality
              | "freshness_requirement" -> factors.freshness_requirement
              | _ -> 0.0
            in
            acc +. value *. weight
          ) 0.0 weights
      
      | NonLinear f -> f factors
      
      | Adaptive { weights; _ } ->
          (* 使用学习到的权重 *)
          let features = [|
            factors.page_rank;
            factors.update_frequency;
            1.0 /. float (factors.depth_from_seed + 1);
            factors.domain_authority;
            factors.content_quality;
          |] in
          Array.fold_left2 (+.) 0.0 features weights
  end
  
  (* 优先级队列实现 *)
  module PriorityQueue = struct
    (* 使用多层优先级队列避免饥饿 *)
    type 'a multilevel_queue = {
      high_priority: 'a Heap.t;
      normal_priority: 'a Heap.t;
      low_priority: 'a Heap.t;
      mutable selection_counter: int;
      selection_ratio: int * int * int;  (* high:normal:low *)
    }
    
    let create ?(ratio=(5, 3, 2)) () = {
      high_priority = Heap.create (fun a b -> compare b a);
      normal_priority = Heap.create (fun a b -> compare b a);
      low_priority = Heap.create (fun a b -> compare b a);
      selection_counter = 0;
      selection_ratio = ratio;
    }
    
    let add queue priority url =
      let (high, normal, low) = queue.selection_ratio in
      if priority > 0.7 then
        Heap.add queue.high_priority (priority, url)
      else if priority > 0.3 then
        Heap.add queue.normal_priority (priority, url)
      else
        Heap.add queue.low_priority (priority, url)
    
    (* 防止低优先级饥饿的调度算法 *)
    let take queue =
      let (high_r, normal_r, low_r) = queue.selection_ratio in
      let total_ratio = high_r + normal_r + low_r in
      let slot = queue.selection_counter mod total_ratio in
      queue.selection_counter <- queue.selection_counter + 1;
      
      (* 根据比例选择队列 *)
      let selected_queue = 
        if slot < high_r then queue.high_priority
        else if slot < high_r + normal_r then queue.normal_priority
        else queue.low_priority
      in
      
      (* 如果选中队列为空，尝试其他队列 *)
      match Heap.take selected_queue with
      | Some item -> Some item
      | None ->
          (* 按优先级顺序尝试其他队列 *)
          match Heap.take queue.high_priority with
          | Some item -> Some item
          | None ->
              match Heap.take queue.normal_priority with
              | Some item -> Some item
              | None -> Heap.take queue.low_priority
  end
end
```

### 礼貌性爬取策略

礼貌性是爬虫系统的基本准则，需要在系统设计中深度集成：

```ocaml
(* 礼貌性爬取控制 *)
module PolitenessControl = struct
  (* robots.txt 解析与缓存 *)
  module RobotsCache = struct
    type rule = {
      user_agent: string;
      disallow: string list;
      allow: string list;
      crawl_delay: float option;
      sitemap: string list;
    }
    
    type robots_info = {
      rules: rule list;
      last_fetched: float;
      expire_time: float;
    }
    
    type t = {
      cache: (string, robots_info) Hashtbl.t;
      default_expire: float;  (* seconds *)
      mutex: Mutex.t;
    }
    
    let parse_robots_txt content =
      let lines = String.split_on_char '\n' content in
      let rec parse_lines lines current_rule rules =
        match lines with
        | [] -> 
            if current_rule.user_agent <> "" then
              current_rule :: rules
            else rules
        | line :: rest ->
            let line = String.trim line in
            if String.starts_with ~prefix:"User-agent:" line then
              let agent = String.trim (String.sub line 11 (String.length line - 11)) in
              let new_rule = { 
                user_agent = agent; 
                disallow = []; 
                allow = [];
                crawl_delay = None;
                sitemap = [];
              } in
              if current_rule.user_agent <> "" then
                parse_lines rest new_rule (current_rule :: rules)
              else
                parse_lines rest new_rule rules
            else if String.starts_with ~prefix:"Disallow:" line then
              let path = String.trim (String.sub line 9 (String.length line - 9)) in
              let updated = { current_rule with disallow = path :: current_rule.disallow } in
              parse_lines rest updated rules
            else if String.starts_with ~prefix:"Allow:" line then
              let path = String.trim (String.sub line 6 (String.length line - 6)) in
              let updated = { current_rule with allow = path :: current_rule.allow } in
              parse_lines rest updated rules
            else if String.starts_with ~prefix:"Crawl-delay:" line then
              let delay = String.trim (String.sub line 12 (String.length line - 12)) in
              let updated = { current_rule with crawl_delay = Some (float_of_string delay) } in
              parse_lines rest updated rules
            else
              parse_lines rest current_rule rules
      in
      parse_lines lines { user_agent = ""; disallow = []; allow = []; 
                          crawl_delay = None; sitemap = [] } []
    
    let is_allowed cache domain path user_agent =
      Mutex.lock cache.mutex;
      let result = 
        match Hashtbl.find_opt cache.cache domain with
        | None -> true  (* 如果没有 robots.txt，默认允许 *)
        | Some info ->
            if Unix.time () > info.expire_time then
              true  (* 过期了，暂时允许，后台更新 *)
            else
              (* 查找匹配的规则 *)
              let applicable_rules = 
                info.rules |> List.filter (fun rule ->
                  rule.user_agent = "*" || 
                  String.lowercase_ascii rule.user_agent = String.lowercase_ascii user_agent
                ) in
              (* 检查是否允许 *)
              List.for_all (fun rule ->
                (* Allow 规则优先于 Disallow *)
                let allowed = List.exists (fun pattern ->
                  String.starts_with ~prefix:pattern path
                ) rule.allow in
                if allowed then true
                else
                  not (List.exists (fun pattern ->
                    String.starts_with ~prefix:pattern path
                  ) rule.disallow)
              ) applicable_rules
      in
      Mutex.unlock cache.mutex;
      result
  end
  
  (* 域名级别的爬取速率控制 *)
  module DomainThrottling = struct
    type domain_state = {
      mutable last_access: float;
      mutable min_delay: float;  (* seconds *)
      mutable adaptive_delay: float;
      mutable recent_response_times: float list;
      mutable error_count: int;
      mutable success_count: int;
    }
    
    type t = {
      domains: (string, domain_state) Hashtbl.t;
      default_delay: float;
      min_delay: float;
      max_delay: float;
      mutex: Mutex.t;
    }
    
    let create ?(default_delay=1.0) ?(min_delay=0.1) ?(max_delay=60.0) () = {
      domains = Hashtbl.create 10000;
      default_delay;
      min_delay;
      max_delay;
      mutex = Mutex.create ();
    }
    
    (* 自适应延迟调整 *)
    let adjust_delay state response_time success =
      (* 维护最近的响应时间窗口 *)
      state.recent_response_times <- 
        response_time :: (List.take 19 state.recent_response_times);
      
      if success then begin
        state.success_count <- state.success_count + 1;
        state.error_count <- 0;
        
        (* 如果响应快且成功率高，可以适当减少延迟 *)
        let avg_response_time = 
          List.fold_left (+.) 0.0 state.recent_response_times /.
          float (List.length state.recent_response_times) in
        
        if avg_response_time < 0.5 && state.success_count > 10 then
          state.adaptive_delay <- state.adaptive_delay *. 0.9
        else if avg_response_time > 2.0 then
          state.adaptive_delay <- state.adaptive_delay *. 1.1
      end else begin
        state.error_count <- state.error_count + 1;
        (* 错误时指数退避 *)
        state.adaptive_delay <- 
          min (state.adaptive_delay *. 2.0) 60.0
      end;
      
      (* 确保在合理范围内 *)
      state.adaptive_delay <- 
        max state.min_delay (min state.adaptive_delay state.max_delay)
    
    let wait_if_needed t domain =
      Mutex.lock t.mutex;
      let state = 
        match Hashtbl.find_opt t.domains domain with
        | Some s -> s
        | None ->
            let s = {
              last_access = 0.0;
              min_delay = t.default_delay;
              adaptive_delay = t.default_delay;
              recent_response_times = [];
              error_count = 0;
              success_count = 0;
            } in
            Hashtbl.add t.domains domain s;
            s
      in
      
      let now = Unix.time () in
      let time_since_last = now -. state.last_access in
      let required_delay = max state.min_delay state.adaptive_delay in
      
      let wait_time = 
        if time_since_last < required_delay then
          required_delay -. time_since_last
        else 0.0
      in
      
      state.last_access <- now +. wait_time;
      Mutex.unlock t.mutex;
      
      if wait_time > 0.0 then
        Unix.sleepf wait_time
  end
  
  (* IP 地址级别的连接管理 *)
  module IPConnectionLimit = struct
    type t = {
      ip_connections: (string, int) Hashtbl.t;
      max_per_ip: int;
      domain_to_ip: (string, string) Hashtbl.t;
      mutex: Mutex.t;
    }
    
    let try_acquire t domain =
      Mutex.lock t.mutex;
      let ip = 
        match Hashtbl.find_opt t.domain_to_ip domain with
        | Some ip -> ip
        | None ->
            (* 实际实现中需要 DNS 解析 *)
            let ip = resolve_domain domain in
            Hashtbl.add t.domain_to_ip domain ip;
            ip
      in
      
      let current = 
        match Hashtbl.find_opt t.ip_connections ip with
        | Some n -> n
        | None -> 0
      in
      
      let acquired = 
        if current < t.max_per_ip then begin
          Hashtbl.replace t.ip_connections ip (current + 1);
          true
        end else
          false
      in
      
      Mutex.unlock t.mutex;
      acquired
    
    let release t domain =
      Mutex.lock t.mutex;
      match Hashtbl.find_opt t.domain_to_ip domain with
      | None -> ()
      | Some ip ->
          match Hashtbl.find_opt t.ip_connections ip with
          | None -> ()
          | Some n -> 
              if n > 1 then
                Hashtbl.replace t.ip_connections ip (n - 1)
              else
                Hashtbl.remove t.ip_connections ip
      ;
      Mutex.unlock t.mutex
  end
end
```

### 分布式队列架构

当爬虫规模扩大到多个节点时，需要分布式的 URL 队列：

```ocaml
(* 分布式 URL 队列设计 *)
module DistributedQueue = struct
  (* 一致性哈希用于 URL 分片 *)
  module ConsistentHashing = struct
    type node = {
      id: string;
      host: string;
      port: int;
      virtual_nodes: int;
    }
    
    type t = {
      ring: (int32 * node) list ref;
      hash_func: string -> int32;
    }
    
    let default_hash s =
      Digest.string s |> Digest.to_hex |> 
      (fun hex -> Int32.of_string ("0x" ^ String.sub hex 0 8))
    
    let create ?(hash_func=default_hash) nodes =
      let ring = ref [] in
      List.iter (fun node ->
        for i = 0 to node.virtual_nodes - 1 do
          let virtual_id = Printf.sprintf "%s:%d" node.id i in
          let hash = hash_func virtual_id in
          ring := (hash, node) :: !ring
        done
      ) nodes;
      ring := List.sort (fun (h1, _) (h2, _) -> Int32.compare h1 h2) !ring;
      { ring; hash_func }
    
    let find_node t key =
      let hash = t.hash_func key in
      let rec find_first_larger = function
        | [] -> List.hd !(t.ring) |> snd  (* wrap around *)
        | (h, node) :: rest ->
            if Int32.compare h hash >= 0 then node
            else find_first_larger rest
      in
      find_first_larger !(t.ring)
  end
  
  (* 分片队列管理器 *)
  module ShardedQueueManager = struct
    type shard = {
      id: int;
      queue: PriorityScheduler.PriorityQueue.multilevel_queue;
      pending_count: int ref;
      processing_count: int ref;
      completed_count: int ref;
      last_activity: float ref;
    }
    
    type t = {
      shards: shard array;
      consistent_hash: ConsistentHashing.t;
      rebalance_threshold: float;  (* 负载不均衡阈值 *)
    }
    
    (* 动态负载均衡 *)
    let rebalance_if_needed t =
      let loads = Array.map (fun shard ->
        float !(shard.pending_count)
      ) t.shards in
      
      let avg_load = 
        Array.fold_left (+.) 0.0 loads /. float (Array.length loads) in
      let max_load = Array.fold_left max 0.0 loads in
      let min_load = Array.fold_left min Float.max_float loads in
      
      (* 如果负载严重不均，触发重平衡 *)
      if max_load > avg_load *. t.rebalance_threshold &&
         max_load -. min_load > avg_load *. 0.5 then begin
        (* 找出负载最高和最低的分片 *)
        let max_shard_idx = 
          Array.mapi (fun i load -> (i, load)) loads |>
          Array.to_list |>
          List.sort (fun (_, l1) (_, l2) -> compare l2 l1) |>
          List.hd |> fst
        in
        let min_shard_idx =
          Array.mapi (fun i load -> (i, load)) loads |>
          Array.to_list |>
          List.sort (fun (_, l1) (_, l2) -> compare l1 l2) |>
          List.hd |> fst
        in
        
        (* 迁移部分 URL *)
        let migration_count = 
          int_of_float ((max_load -. avg_load) *. 0.5) in
        (* 实际迁移逻辑... *)
        ()
      end
  end
  
  (* 分布式事务支持 *)
  module DistributedTransaction = struct
    type transaction_id = string
    
    type operation =
      | Enqueue of Url.t * float  (* URL and priority *)
      | Dequeue of int  (* number of URLs *)
      | UpdatePriority of Url.t * float
    
    type transaction = {
      id: transaction_id;
      operations: operation list;
      participants: string list;  (* node IDs *)
      mutable state: [ `Preparing | `Prepared | `Committing | `Committed | `Aborted ];
      created_at: float;
      timeout: float;
    }
    
    (* 两阶段提交协议 *)
    let execute_2pc coordinator transaction =
      (* Phase 1: Prepare *)
      let prepare_results = 
        transaction.participants |> List.map (fun node_id ->
          try
            let response = RPC.call node_id 
              (`Prepare (transaction.id, transaction.operations)) in
            (node_id, response = `Prepared)
          with _ -> (node_id, false)
        ) in
      
      let all_prepared = List.for_all snd prepare_results in
      
      (* Phase 2: Commit or Abort *)
      if all_prepared then begin
        transaction.state <- `Committing;
        transaction.participants |> List.iter (fun node_id ->
          ignore (RPC.call node_id (`Commit transaction.id))
        );
        transaction.state <- `Committed;
        `Success
      end else begin
        transaction.state <- `Aborted;
        transaction.participants |> List.iter (fun node_id ->
          ignore (RPC.call node_id (`Abort transaction.id))
        );
        `Aborted
      end
  end
end
```

### 故障恢复与断点续爬

爬虫系统需要能够从各种故障中恢复，并支持断点续爬：

```ocaml
(* 故障恢复机制 *)
module FaultRecovery = struct
  (* 爬取状态持久化 *)
  module CrawlState = struct
    type url_state = 
      | Pending of { 
          enqueued_at: float;
          priority: float;
          retry_count: int;
        }
      | Processing of {
          started_at: float;
          worker_id: string;
          attempt: int;
        }
      | Completed of {
          finished_at: float;
          success: bool;
          content_hash: string option;
        }
      | Failed of {
          failed_at: float;
          reason: string;
          retry_after: float option;
        }
    
    type checkpoint = {
      timestamp: float;
      total_urls: int;
      completed_urls: int;
      failed_urls: int;
      processing_urls: int;
      state_snapshot: (Url.t * url_state) list;
    }
    
    (* 增量检查点 *)
    module IncrementalCheckpoint = struct
      type t = {
        base_checkpoint: checkpoint;
        mutable deltas: (float * (Url.t * url_state)) list;
        mutable delta_count: int;
        compact_threshold: int;
      }
      
      let add_delta t url state =
        t.deltas <- (Unix.time (), (url, state)) :: t.deltas;
        t.delta_count <- t.delta_count + 1;
        
        (* 如果增量过多，创建新的基础检查点 *)
        if t.delta_count >= t.compact_threshold then
          compact_checkpoint t
      
      let replay_from_checkpoint t =
        (* 先恢复基础状态 *)
        let state_table = Hashtbl.create 10000 in
        List.iter (fun (url, state) ->
          Hashtbl.add state_table url state
        ) t.base_checkpoint.state_snapshot;
        
        (* 应用增量更新 *)
        List.iter (fun (_, (url, state)) ->
          Hashtbl.replace state_table url state
        ) (List.rev t.deltas);
        
        state_table
    end
  end
  
  (* 工作节点故障检测与恢复 *)
  module WorkerFailover = struct
    type worker_status = 
      | Active of { last_heartbeat: float; current_load: int }
      | Suspicious of { last_seen: float; missed_heartbeats: int }
      | Failed of { detected_at: float; assigned_urls: Url.t list }
    
    type t = {
      workers: (string, worker_status) Hashtbl.t;
      heartbeat_interval: float;
      failure_threshold: int;  (* 错过心跳次数 *)
      mutable monitor_thread: Thread.t option;
    }
    
    let monitor_workers t =
      let rec check_loop () =
        Unix.sleep (int_of_float t.heartbeat_interval);
        let now = Unix.time () in
        
        Hashtbl.iter (fun worker_id status ->
          match status with
          | Active { last_heartbeat; current_load } ->
              if now -. last_heartbeat > t.heartbeat_interval *. 2.0 then
                Hashtbl.replace t.workers worker_id 
                  (Suspicious { last_seen = last_heartbeat; missed_heartbeats = 1 })
          
          | Suspicious { last_seen; missed_heartbeats } ->
              if missed_heartbeats >= t.failure_threshold then begin
                (* 标记为失败并触发恢复 *)
                let assigned_urls = get_assigned_urls worker_id in
                Hashtbl.replace t.workers worker_id 
                  (Failed { detected_at = now; assigned_urls });
                trigger_failover worker_id assigned_urls
              end else
                Hashtbl.replace t.workers worker_id
                  (Suspicious { last_seen; 
                               missed_heartbeats = missed_heartbeats + 1 })
          
          | Failed _ -> ()  (* 已经在处理中 *)
        ) t.workers;
        
        check_loop ()
      in
      t.monitor_thread <- Some (Thread.create check_loop ())
    
    let trigger_failover failed_worker_id urls =
      (* 将失败节点的 URL 重新分配给健康节点 *)
      let healthy_workers = 
        Hashtbl.fold (fun id status acc ->
          match status with
          | Active _ -> id :: acc
          | _ -> acc
        ) t.workers []
      in
      
      (* 使用一致性哈希重新分配 *)
      List.iter (fun url ->
        let new_worker = select_least_loaded healthy_workers in
        reassign_url url failed_worker_id new_worker
      ) urls
  end
  
  (* WAL (Write-Ahead Logging) 保证持久性 *)
  module WriteAheadLog = struct
    type log_entry =
      | UrlEnqueued of { url: Url.t; priority: float; timestamp: float }
      | UrlDequeued of { url: Url.t; worker: string; timestamp: float }
      | UrlCompleted of { url: Url.t; success: bool; timestamp: float }
      | CheckpointCreated of { id: string; timestamp: float }
    
    type t = {
      log_dir: string;
      current_file: out_channel ref;
      mutable current_size: int;
      max_file_size: int;
      mutable sequence_number: int64;
    }
    
    let append_entry t entry =
      let serialized = Marshal.to_string (t.sequence_number, entry) [] in
      output_string !(t.current_file) serialized;
      flush !(t.current_file);
      
      t.sequence_number <- Int64.succ t.sequence_number;
      t.current_size <- t.current_size + String.length serialized;
      
      (* 轮转日志文件 *)
      if t.current_size >= t.max_file_size then
        rotate_log_file t
    
    let replay_log t after_checkpoint =
      let files = get_log_files_after t.log_dir after_checkpoint in
      let entries = ref [] in
      
      List.iter (fun file ->
        let ic = open_in file in
        try
          while true do
            let (seq, entry) = Marshal.from_channel ic in
            if seq > after_checkpoint then
              entries := entry :: !entries
          done
        with End_of_file ->
          close_in ic
      ) files;
      
      (* 按顺序重放日志条目 *)
      List.rev !entries |> List.iter (fun entry ->
        match entry with
        | UrlEnqueued { url; priority; _ } ->
            restore_url_to_queue url priority
        | UrlDequeued { url; worker; _ } ->
            mark_url_processing url worker
        | UrlCompleted { url; success; _ } ->
            mark_url_completed url success
        | CheckpointCreated _ -> ()
      )
  end
end
```

## 5.4 去重策略与数据结构选择

### URL 去重的挑战

在大规模爬虫系统中，去重是一个看似简单实则充满挑战的问题。一个典型的爬虫系统可能需要处理数十亿个 URL，每个 URL 平均长度 50-100 字符，简单的哈希表存储就需要数百 GB 的内存。更复杂的是，URL 去重不仅要考虑精确匹配，还要处理 URL 规范化、参数变化、内容去重等多层次的问题。

```ocaml
(* URL 去重的多层次挑战 *)
module DuplicationChallenges = struct
  (* URL 变体问题 *)
  type url_variant_source =
    | CaseVariation         (* http://example.com vs HTTP://EXAMPLE.COM *)
    | TrailingSlash        (* /path vs /path/ *)
    | ParameterOrder       (* ?a=1&b=2 vs ?b=2&a=1 *)
    | FragmentIdentifier   (* /page vs /page#section *)
    | DefaultPorts         (* http://site.com:80 vs http://site.com *)
    | PercentEncoding      (* /hello world vs /hello%20world *)
    | WWWPrefix           (* www.site.com vs site.com *)
    | SessionParameters    (* ?sessionid=123&page=1 *)
    | UTMTracking         (* ?utm_source=google&utm_medium=cpc *)
  
  (* 内容变体问题 *)
  type content_variant_source =
    | DynamicTimestamps    (* 页面包含当前时间 *)
    | PersonalizedContent  (* 基于用户的个性化内容 *)
    | RandomAdvertisements (* 随机广告位 *)
    | CommentSections     (* 动态评论区 *)
    | ServerHeaders       (* 变化的服务器响应头 *)
    | MinorTemplateChanges (* 模板小幅调整 *)
  
  (* 规模挑战 *)
  type scale_challenge = {
    total_urls: int64;              (* 数十亿 *)
    memory_constraint: int;         (* GB *)
    false_positive_tolerance: float; (* 可接受的误判率 *)
    lookup_latency_ms: float;       (* 毫秒级要求 *)
    update_throughput: int;         (* URLs/second *)
  }
  
  (* 分布式挑战 *)
  type distributed_challenge =
    | ConsistencyRequirement   (* 多节点间的一致性 *)
    | NetworkLatency          (* 跨节点查询延迟 *)
    | PartialFailures        (* 部分节点失效 *)
    | DataSkew              (* 热点 URL 导致的倾斜 *)
    | ReplicationOverhead   (* 复制开销 *)
end
```

### URL 规范化策略

在进行去重之前，URL 规范化是关键的预处理步骤：

```ocaml
(* URL 规范化实现 *)
module UrlNormalization = struct
  type normalization_rule =
    | LowercaseSchemeHost    (* 协议和主机名小写 *)
    | RemoveDefaultPort      (* 移除默认端口 *)
    | RemoveFragment         (* 移除锚点 *)
    | SortQueryParameters    (* 查询参数排序 *)
    | RemoveEmptyParameters  (* 移除空参数 *)
    | DecodeUnreserved      (* 解码非保留字符 *)
    | RemoveTrailingSlash   (* 移除末尾斜杠 *)
    | NormalizePathSegments (* 规范化路径 ../ ./ *)
    | RemoveWWW            (* 移除 www 前缀 *)
    | RemoveSessionParams   (* 移除会话参数 *)
    | CustomRule of (Url.t -> Url.t)
  
  type normalization_profile = {
    rules: normalization_rule list;
    parameter_blacklist: string list;  (* 需要移除的参数名 *)
    preserve_parameters: string list;   (* 必须保留的参数 *)
  }
  
  (* 标准规范化配置 *)
  let standard_profile = {
    rules = [
      LowercaseSchemeHost;
      RemoveDefaultPort;
      RemoveFragment;
      SortQueryParameters;
      RemoveEmptyParameters;
      DecodeUnreserved;
      NormalizePathSegments;
    ];
    parameter_blacklist = [
      "utm_source"; "utm_medium"; "utm_campaign";
      "fbclid"; "gclid"; "sessionid"; "sid";
      "_ga"; "callback"; "rand"; "timestamp";
    ];
    preserve_parameters = [
      "id"; "page"; "query"; "search"; "category";
    ];
  }
  
  (* 规范化实现 *)
  let normalize profile url =
    let open Url in
    let url = parse url in
    
    (* 应用规则链 *)
    List.fold_left (fun u rule ->
      match rule with
      | LowercaseSchemeHost ->
          { u with 
            scheme = String.lowercase_ascii u.scheme;
            host = String.lowercase_ascii u.host }
      
      | RemoveDefaultPort ->
          let default_port = match u.scheme with
            | "http" -> Some 80
            | "https" -> Some 443
            | "ftp" -> Some 21
            | _ -> None
          in
          if u.port = default_port then
            { u with port = None }
          else u
      
      | SortQueryParameters ->
          let params = parse_query u.query in
          let filtered = params |> List.filter (fun (k, _) ->
            not (List.mem k profile.parameter_blacklist) ||
            List.mem k profile.preserve_parameters
          ) in
          let sorted = List.sort (fun (k1, _) (k2, _) -> 
            String.compare k1 k2) filtered in
          { u with query = build_query sorted }
      
      | NormalizePathSegments ->
          let segments = String.split_on_char '/' u.path in
          let normalized = normalize_path_segments segments in
          { u with path = String.concat "/" normalized }
      
      | _ -> u
    ) url profile.rules
    |> to_string
  
  (* 相似 URL 检测 *)
  module SimilarityDetection = struct
    type similarity_method =
      | ExactAfterNormalization
      | IgnoreQueryString
      | PathPrefixMatch of int  (* 前 N 段路径 *)
      | DomainAndPath          (* 忽略所有参数 *)
      | ContentHash            (* 基于内容的哈希 *)
    
    let are_similar method1 url1 url2 =
      match method1 with
      | ExactAfterNormalization ->
          normalize standard_profile url1 = 
          normalize standard_profile url2
      
      | IgnoreQueryString ->
          let u1 = Url.parse url1 in
          let u2 = Url.parse url2 in
          u1.scheme = u2.scheme && 
          u1.host = u2.host && 
          u1.path = u2.path
      
      | PathPrefixMatch n ->
          let segments1 = Url.parse url1 |> fun u -> 
            String.split_on_char '/' u.path |> List.take n in
          let segments2 = Url.parse url2 |> fun u ->
            String.split_on_char '/' u.path |> List.take n in
          segments1 = segments2
      
      | _ -> false
  end
end
```

### 布隆过滤器的应用

布隆过滤器是解决大规模 URL 去重的经典方案，提供了空间效率和查询速度的良好平衡：

```ocaml
(* 布隆过滤器实现与优化 *)
module BloomFilter = struct
  type t = {
    bits: Bytes.t;
    size: int;           (* 位数组大小 *)
    num_hashes: int;     (* 哈希函数数量 *)
    mutable count: int;  (* 已插入元素数量 *)
  }
  
  (* 最优参数计算 *)
  module OptimalParameters = struct
    let calculate ~expected_elements ~false_positive_rate =
      (* m = -n * ln(p) / (ln(2)^2) *)
      let m = -. (float expected_elements) *. 
              (log false_positive_rate) /. 
              (log 2.0 ** 2.0) in
      let m = int_of_float (ceil m) in
      
      (* k = (m/n) * ln(2) *)
      let k = (float m /. float expected_elements) *. log 2.0 in
      let k = int_of_float (round k) in
      
      { size = m; num_hashes = max 1 k }
    
    (* 实际误判率计算 *)
    let actual_false_positive_rate t =
      (* p = (1 - e^(-k*n/m))^k *)
      let ratio = float t.count /. float t.size in
      (1.0 -. exp(-. float t.num_hashes *. ratio)) ** 
      float t.num_hashes
  end
  
  (* 创建布隆过滤器 *)
  let create ~expected_elements ~false_positive_rate =
    let params = OptimalParameters.calculate 
      ~expected_elements ~false_positive_rate in
    {
      bits = Bytes.create ((params.size + 7) / 8);
      size = params.size;
      num_hashes = params.num_hashes;
      count = 0;
    }
  
  (* 哈希函数族 *)
  module HashFamily = struct
    (* 使用双重哈希生成多个哈希值 *)
    let double_hashing seed1 seed2 data k m =
      let h1 = Hashtbl.hash_param seed1 100 data in
      let h2 = Hashtbl.hash_param seed2 100 data in
      Array.init k (fun i ->
        abs ((h1 + i * h2) mod m)
      )
    
    (* MurmurHash3 实现用于更好的分布 *)
    external murmur3_32: string -> int32 -> int32 = "caml_murmur3_32"
    
    let murmur_hashes data k m =
      Array.init k (fun i ->
        let h = murmur3_32 data (Int32.of_int i) in
        Int32.to_int (Int32.rem h (Int32.of_int m)) |> abs
      )
  end
  
  (* 基本操作 *)
  let add t data =
    let positions = HashFamily.murmur_hashes data t.num_hashes t.size in
    Array.iter (fun pos ->
      let byte_idx = pos / 8 in
      let bit_idx = pos mod 8 in
      let mask = 1 lsl bit_idx in
      let old_byte = Bytes.get t.bits byte_idx |> Char.code in
      Bytes.set t.bits byte_idx (Char.chr (old_byte lor mask))
    ) positions;
    t.count <- t.count + 1
  
  let might_contain t data =
    let positions = HashFamily.murmur_hashes data t.num_hashes t.size in
    Array.for_all (fun pos ->
      let byte_idx = pos / 8 in
      let bit_idx = pos mod 8 in
      let mask = 1 lsl bit_idx in
      let byte = Bytes.get t.bits byte_idx |> Char.code in
      (byte land mask) <> 0
    ) positions
  
  (* Counting Bloom Filter 支持删除 *)
  module Counting = struct
    type t = {
      counters: int array;
      size: int;
      num_hashes: int;
      mutable count: int;
      max_count: int;  (* 计数器最大值，防止溢出 *)
    }
    
    let create ~expected_elements ~false_positive_rate =
      let params = OptimalParameters.calculate 
        ~expected_elements ~false_positive_rate in
      {
        counters = Array.make params.size 0;
        size = params.size;
        num_hashes = params.num_hashes;
        count = 0;
        max_count = 15;  (* 4-bit 计数器 *)
      }
    
    let add t data =
      let positions = HashFamily.murmur_hashes data t.num_hashes t.size in
      Array.iter (fun pos ->
        if t.counters.(pos) < t.max_count then
          t.counters.(pos) <- t.counters.(pos) + 1
      ) positions;
      t.count <- t.count + 1
    
    let remove t data =
      let positions = HashFamily.murmur_hashes data t.num_hashes t.size in
      let can_remove = Array.for_all (fun pos ->
        t.counters.(pos) > 0
      ) positions in
      
      if can_remove then begin
        Array.iter (fun pos ->
          t.counters.(pos) <- t.counters.(pos) - 1
        ) positions;
        t.count <- t.count - 1;
        true
      end else
        false
  end
  
  (* 可扩展布隆过滤器 *)
  module Scalable = struct
    type t = {
      filters: BloomFilter.t list ref;
      growth_factor: float;
      target_fpr: float;
      mutable current_capacity: int;
    }
    
    let create ~initial_capacity ~false_positive_rate ~growth_factor =
      let first = BloomFilter.create 
        ~expected_elements:initial_capacity
        ~false_positive_rate in
      {
        filters = ref [first];
        growth_factor;
        target_fpr = false_positive_rate;
        current_capacity = initial_capacity;
      }
    
    let add t data =
      (* 检查是否需要新的过滤器 *)
      let current = List.hd !(t.filters) in
      if float current.count >= 
         float current.size *. 0.9 then begin
        (* 创建新的更大的过滤器 *)
        t.current_capacity <- 
          int_of_float (float t.current_capacity *. t.growth_factor);
        let new_filter = BloomFilter.create
          ~expected_elements:t.current_capacity
          ~false_positive_rate:(t.target_fpr *. 0.5) in
        t.filters := new_filter :: !(t.filters)
      end;
      
      BloomFilter.add (List.hd !(t.filters)) data
    
    let might_contain t data =
      List.exists (fun f -> BloomFilter.might_contain f data) !(t.filters)
  end
end
```

### 分布式去重架构

当单机内存无法容纳所有去重数据时，需要分布式去重方案：

```ocaml
(* 分布式去重系统 *)
module DistributedDeduplication = struct
  (* 分片策略 *)
  module ShardingStrategy = struct
    type strategy =
      | HashMod of int              (* 简单取模 *)
      | ConsistentHash              (* 一致性哈希 *)
      | RangePartition of string list (* 范围分区 *)
      | GeoSharding                 (* 基于地理位置 *)
      | DomainSharding              (* 基于域名 *)
    
    type shard_info = {
      id: int;
      node: string;
      capacity: int64;
      current_load: int64;
    }
    
    let compute_shard strategy url total_shards =
      match strategy with
      | HashMod n ->
          (Hashtbl.hash url) mod n
      
      | ConsistentHash ->
          let hash = Digest.string url in
          ConsistentHashing.find_node hash
      
      | DomainSharding ->
          let domain = Url.parse url |> fun u -> u.host in
          (Hashtbl.hash domain) mod total_shards
      
      | _ -> 0
  end
  
  (* 分布式布隆过滤器 *)
  module DistributedBloom = struct
    type node = {
      id: string;
      address: string;
      local_filter: BloomFilter.t;
      capacity: int;
      mutable load: int;
    }
    
    type t = {
      nodes: node array;
      sharding: ShardingStrategy.strategy;
      replication_factor: int;
      read_quorum: int;
      write_quorum: int;
    }
    
    (* 带复制的写入 *)
    let add t url =
      let primary_shard = 
        ShardingStrategy.compute_shard t.sharding url (Array.length t.nodes) in
      
      (* 写入主分片和副本 *)
      let replicas = get_replicas primary_shard t.replication_factor in
      let write_results = replicas |> List.map (fun shard_id ->
        try
          let node = t.nodes.(shard_id) in
          RPC.call node.address (`Add url);
          true
        with _ -> false
      ) in
      
      (* 检查是否满足写入仲裁 *)
      let success_count = 
        List.filter (fun x -> x) write_results |> List.length in
      success_count >= t.write_quorum
    
    (* 带仲裁的查询 *)
    let might_contain t url =
      let primary_shard = 
        ShardingStrategy.compute_shard t.sharding url (Array.length t.nodes) in
      
      let replicas = get_replicas primary_shard t.replication_factor in
      let query_results = replicas |> List.filter_map (fun shard_id ->
        try
          let node = t.nodes.(shard_id) in
          Some (RPC.call node.address (`Contains url))
        with _ -> None
      ) in
      
      (* 需要读仲裁数量的一致结果 *)
      let positive_count = 
        List.filter (fun x -> x) query_results |> List.length in
      positive_count >= t.read_quorum
  end
  
  (* HyperLogLog 用于基数估计 *)
  module HyperLogLog = struct
    type t = {
      registers: int array;
      b: int;  (* 使用 b 位作为桶索引，m = 2^b 个桶 *)
      alpha: float;  (* 修正常数 *)
    }
    
    let create ?(error_rate=0.01) () =
      (* 标准误差率 1.04/√m *)
      let m = int_of_float (ceil (1.04 /. error_rate) ** 2.0) in
      let b = int_of_float (ceil (log (float m) /. log 2.0)) in
      let m = 1 lsl b in  (* 确保 m 是 2 的幂 *)
      
      let alpha = match m with
        | m when m >= 128 -> 0.7213 /. (1.0 +. 1.079 /. float m)
        | m when m >= 64 -> 0.709
        | m when m >= 32 -> 0.697
        | 16 -> 0.673
        | _ -> 0.5
      in
      
      { registers = Array.make m 0; b; alpha }
    
    let add t data =
      let hash = Hashtbl.hash data in
      let j = hash land ((1 lsl t.b) - 1) in  (* 前 b 位作为桶索引 *)
      let w = hash lsr t.b in  (* 剩余位 *)
      
      (* 计算第一个 1 出现的位置 *)
      let rho = 
        if w = 0 then 32 - t.b + 1
        else 
          let rec leading_zeros x acc =
            if x land 1 = 1 then acc + 1
            else leading_zeros (x lsr 1) (acc + 1)
          in
          leading_zeros w 0
      in
      
      (* 更新寄存器 *)
      t.registers.(j) <- max t.registers.(j) rho
    
    let cardinality t =
      let m = float (Array.length t.registers) in
      let raw_estimate = t.alpha *. m *. m /.
        (Array.fold_left (fun acc reg ->
          acc +. (2.0 ** (-. float reg))
        ) 0.0 t.registers) in
      
      (* 小范围修正 *)
      if raw_estimate <= 2.5 *. m then
        let zeros = Array.fold_left (fun acc reg ->
          if reg = 0 then acc + 1 else acc
        ) 0 t.registers in
        if zeros <> 0 then
          m *. log (m /. float zeros)
        else
          raw_estimate
      (* 大范围修正 *)
      else if raw_estimate <= (1.0 /. 30.0) *. (2.0 ** 32.0) then
        raw_estimate
      else
        -. (2.0 ** 32.0) *. log (1.0 -. raw_estimate /. (2.0 ** 32.0))
    
    (* 合并多个 HyperLogLog *)
    let merge t1 t2 =
      if Array.length t1.registers <> Array.length t2.registers then
        failwith "Cannot merge HyperLogLog with different sizes";
      
      let merged = create ~error_rate:0.01 () in
      for i = 0 to Array.length t1.registers - 1 do
        merged.registers.(i) <- max t1.registers.(i) t2.registers.(i)
      done;
      merged
  end
  
  (* 两阶段去重：本地 + 全局 *)
  module TwoPhaseDedup = struct
    type local_cache = {
      recent_urls: (string, float) Hashtbl.t;  (* URL -> timestamp *)
      local_bloom: BloomFilter.t;
      sync_interval: float;
      mutable last_sync: float;
    }
    
    type global_dedup = {
      distributed_bloom: DistributedBloom.t;
      hyperloglog: HyperLogLog.t;  (* 用于统计 *)
    }
    
    let check_duplicate local global url =
      (* 第一阶段：本地缓存检查 *)
      if Hashtbl.mem local.recent_urls url then
        `Duplicate
      else if BloomFilter.might_contain local.local_bloom url then
        (* 第二阶段：全局检查 *)
        if DistributedBloom.might_contain global.distributed_bloom url then
          `Duplicate
        else begin
          (* 全局不存在，更新本地 *)
          Hashtbl.add local.recent_urls url (Unix.time ());
          BloomFilter.add local.local_bloom url;
          `New
        end
      else begin
        (* 本地不存在，添加到本地 *)
        Hashtbl.add local.recent_urls url (Unix.time ());
        BloomFilter.add local.local_bloom url;
        `New
      end
    
    (* 定期同步本地去重信息到全局 *)
    let sync_to_global local global =
      let now = Unix.time () in
      if now -. local.last_sync > local.sync_interval then begin
        (* 批量更新全局布隆过滤器 *)
        Hashtbl.iter (fun url timestamp ->
          if timestamp > local.last_sync then
            DistributedBloom.add global.distributed_bloom url
        ) local.recent_urls;
        
        (* 清理过期的本地缓存 *)
        Hashtbl.filter_map_inplace (fun _ timestamp ->
          if now -. timestamp < 3600.0 then Some timestamp else None
        ) local.recent_urls;
        
        local.last_sync <- now
      end
  end
end
```

### 内容指纹与相似度检测

除了 URL 去重，内容去重同样重要，特别是对于动态生成的页面：

```ocaml
(* 内容指纹与相似度检测 *)
module ContentFingerprinting = struct
  (* SimHash 算法实现 *)
  module SimHash = struct
    type feature_weight = string * float
    
    let hash_bits = 64
    
    (* 计算 SimHash *)
    let compute features =
      let v = Array.make hash_bits 0.0 in
      
      List.iter (fun (feature, weight) ->
        let hash = Hashtbl.hash feature in
        for i = 0 to hash_bits - 1 do
          if (hash lsr i) land 1 = 1 then
            v.(i) <- v.(i) +. weight
          else
            v.(i) <- v.(i) -. weight
        done
      ) features;
      
      (* 生成最终的哈希值 *)
      let hash = ref Int64.zero in
      for i = 0 to hash_bits - 1 do
        if v.(i) > 0.0 then
          hash := Int64.logor !hash (Int64.shift_left Int64.one i)
      done;
      !hash
    
    (* 计算汉明距离 *)
    let hamming_distance h1 h2 =
      let xor = Int64.logxor h1 h2 in
      let rec count_bits n acc =
        if n = Int64.zero then acc
        else count_bits (Int64.shift_right_logical n 1) 
                       (acc + Int64.to_int (Int64.logand n Int64.one))
      in
      count_bits xor 0
    
    (* 相似度阈值判断 *)
    let are_similar ?(threshold=3) h1 h2 =
      hamming_distance h1 h2 <= threshold
    
    (* 特征提取 *)
    module FeatureExtraction = struct
      (* 基于 shingle 的特征提取 *)
      let shingles n text =
        let words = String.split_on_char ' ' text in
        let rec extract acc = function
          | [] -> acc
          | words when List.length words < n -> acc
          | words ->
              let shingle = List.take n words |> String.concat " " in
              extract ((shingle, 1.0) :: acc) (List.tl words)
        in
        extract [] words
      
      (* 基于关键词的特征提取 *)
      let keywords text =
        let words = tokenize text in
        let freq = Hashtbl.create 100 in
        
        (* 统计词频 *)
        List.iter (fun word ->
          let count = try Hashtbl.find freq word with Not_found -> 0 in
          Hashtbl.replace freq word (count + 1)
        ) words;
        
        (* TF-IDF 权重 *)
        Hashtbl.fold (fun word count acc ->
          let tf = float count /. float (List.length words) in
          let idf = get_idf word in  (* 预计算的 IDF *)
          (word, tf *. idf) :: acc
        ) freq []
        |> List.sort (fun (_, w1) (_, w2) -> compare w2 w1)
        |> List.take 100  (* 前 100 个关键词 *)
    end
  end
  
  (* MinHash 用于集合相似度 *)
  module MinHash = struct
    type t = {
      num_hashes: int;
      signatures: int array;
    }
    
    (* 生成哈希函数族 *)
    let create_hash_functions n =
      Array.init n (fun i ->
        let a = Random.int 1000000 + 1 in
        let b = Random.int 1000000 in
        let p = 1000000007 in  (* 大素数 *)
        fun x -> (a * x + b) mod p
      )
    
    let compute num_hashes elements =
      let hash_funcs = create_hash_functions num_hashes in
      let signatures = Array.make num_hashes max_int in
      
      List.iter (fun elem ->
        let elem_hash = Hashtbl.hash elem in
        Array.iteri (fun i hash_func ->
          let h = hash_func elem_hash in
          signatures.(i) <- min signatures.(i) h
        ) hash_funcs
      ) elements;
      
      { num_hashes; signatures }
    
    (* Jaccard 相似度估计 *)
    let similarity mh1 mh2 =
      if mh1.num_hashes <> mh2.num_hashes then
        failwith "MinHash signatures must have same length";
      
      let matches = ref 0 in
      for i = 0 to mh1.num_hashes - 1 do
        if mh1.signatures.(i) = mh2.signatures.(i) then
          incr matches
      done;
      
      float !matches /. float mh1.num_hashes
    
    (* LSH (Locality Sensitive Hashing) 用于快速查找 *)
    module LSH = struct
      type t = {
        bands: int;
        rows: int;  (* rows * bands = num_hashes *)
        buckets: (int, MinHash.t list) Hashtbl.t array;
      }
      
      let create ~num_hashes ~similarity_threshold =
        (* 选择 bands 和 rows 使得相似文档大概率在同一桶 *)
        let bands = int_of_float (sqrt (float num_hashes)) in
        let rows = num_hashes / bands in
        {
          bands;
          rows;
          buckets = Array.init bands (fun _ -> Hashtbl.create 1000);
        }
      
      let add lsh minhash =
        for band = 0 to lsh.bands - 1 do
          (* 计算该 band 的哈希值 *)
          let band_hash = ref 0 in
          for row = 0 to lsh.rows - 1 do
            let idx = band * lsh.rows + row in
            band_hash := !band_hash * 31 + minhash.signatures.(idx)
          done;
          
          (* 添加到对应的桶 *)
          let bucket = lsh.buckets.(band) in
          let current = 
            try Hashtbl.find bucket !band_hash 
            with Not_found -> [] in
          Hashtbl.replace bucket !band_hash (minhash :: current)
        done
      
      let find_similar lsh minhash threshold =
        let candidates = ref [] in
        
        for band = 0 to lsh.bands - 1 do
          let band_hash = ref 0 in
          for row = 0 to lsh.rows - 1 do
            let idx = band * lsh.rows + row in
            band_hash := !band_hash * 31 + minhash.signatures.(idx)
          done;
          
          let bucket = lsh.buckets.(band) in
          match Hashtbl.find_opt bucket !band_hash with
          | None -> ()
          | Some items ->
              List.iter (fun candidate ->
                if MinHash.similarity minhash candidate >= threshold then
                  candidates := candidate :: !candidates
              ) items
        done;
        
        !candidates |> List.sort_uniq compare
    end
  end
  
  (* 基于 DOM 树的结构化指纹 *)
  module StructuralFingerprint = struct
    type dom_feature =
      | TagSequence of string list     (* 标签序列 *)
      | TreeDepth of int              (* DOM 树深度 *)
      | TextDensity of float          (* 文本密度 *)
      | LinkDensity of float          (* 链接密度 *)
      | BlockStructure of string      (* 块级元素结构 *)
    
    let extract_features dom =
      let rec traverse node depth acc =
        match node with
        | Element (tag, attrs, children) ->
            let child_features = List.fold_left (fun acc child ->
              traverse child (depth + 1) acc
            ) acc children in
            (TagSequence [tag]) :: child_features
        | Text content ->
            (TextDensity (float (String.length content))) :: acc
        | _ -> acc
      in
      traverse dom 0 []
    
    (* 模板检测 *)
    let detect_template pages =
      (* 提取所有页面的公共结构 *)
      let all_features = List.map extract_features pages in
      
      (* 找出频繁出现的特征模式 *)
      let pattern_freq = Hashtbl.create 1000 in
      List.iter (fun features ->
        List.iter (fun feature ->
          let count = 
            try Hashtbl.find pattern_freq feature 
            with Not_found -> 0 in
          Hashtbl.replace pattern_freq feature (count + 1)
        ) features
      ) all_features;
      
      (* 返回出现在 80% 以上页面的模式 *)
      let threshold = float (List.length pages) *. 0.8 in
      Hashtbl.fold (fun pattern count acc ->
        if float count >= threshold then
          pattern :: acc
        else acc
      ) pattern_freq []
  end
end
```

## 5.5 渲染服务的接口设计

### JavaScript 渲染需求

现代 Web 应用大量依赖 JavaScript 动态生成内容，传统的 HTTP 客户端只能获取初始 HTML，无法执行 JavaScript 代码。这给爬虫系统带来了新的挑战：如何高效地渲染 JavaScript 驱动的页面，同时保持系统的可扩展性和稳定性。

```ocaml
(* JavaScript 渲染需求分析 *)
module RenderingRequirements = struct
  (* 需要渲染的页面类型 *)
  type page_type =
    | StaticHTML              (* 纯静态页面 *)
    | SPAApplication          (* 单页应用 *)
    | LazyLoadedContent       (* 懒加载内容 *)
    | InfiniteScroll         (* 无限滚动 *)
    | AJAXDrivenContent      (* AJAX 加载的内容 *)
    | WebSocketUpdates       (* WebSocket 实时更新 *)
    | CanvasRendered         (* Canvas 绘制的内容 *)
    | WebAssemblyApp         (* WebAssembly 应用 *)
  
  (* 渲染策略选择 *)
  type rendering_strategy =
    | NoRendering            (* 不需要渲染 *)
    | BasicJSExecution       (* 基础 JS 执行 *)
    | FullBrowserEmulation   (* 完整浏览器模拟 *)
    | SelectiveRendering of { (* 选择性渲染 *)
        wait_conditions: wait_condition list;
        timeout: float;
        resource_filters: resource_filter list;
      }
  
  and wait_condition =
    | DOMContentLoaded
    | NetworkIdle of float    (* 网络空闲时间 *)
    | ElementPresent of string (* CSS 选择器 *)
    | CustomScript of string   (* 自定义 JS 条件 *)
    | TimeDelay of float      (* 固定延迟 *)
  
  and resource_filter =
    | BlockImages
    | BlockStylesheets
    | BlockFonts
    | BlockMedia
    | AllowOnlyDomain of string list
    | BlockDomain of string list
  
  (* 渲染成本评估 *)
  module CostEstimation = struct
    type rendering_cost = {
      cpu_time: float;        (* CPU 时间（秒） *)
      memory_usage: int;      (* 内存使用（MB） *)
      network_traffic: int;   (* 网络流量（KB） *)
      rendering_time: float;  (* 总渲染时间 *)
    }
    
    let estimate_cost page_type strategy =
      match page_type, strategy with
      | StaticHTML, NoRendering ->
          { cpu_time = 0.01; memory_usage = 10; 
            network_traffic = 50; rendering_time = 0.1 }
      
      | SPAApplication, FullBrowserEmulation ->
          { cpu_time = 2.0; memory_usage = 300;
            network_traffic = 2000; rendering_time = 5.0 }
      
      | LazyLoadedContent, SelectiveRendering _ ->
          { cpu_time = 1.0; memory_usage = 150;
            network_traffic = 500; rendering_time = 2.0 }
      
      | _ ->
          { cpu_time = 0.5; memory_usage = 100;
            network_traffic = 200; rendering_time = 1.0 }
    
    (* 动态策略选择 *)
    let choose_strategy url historical_data =
      let domain = Url.parse url |> fun u -> u.host in
      
      (* 基于历史数据的启发式判断 *)
      match Hashtbl.find_opt historical_data domain with
      | Some { requires_js = false; _ } -> NoRendering
      | Some { spa_detected = true; _ } -> FullBrowserEmulation
      | Some { ajax_requests > 0; _ } -> 
          SelectiveRendering {
            wait_conditions = [NetworkIdle 0.5];
            timeout = 3.0;
            resource_filters = [BlockImages; BlockMedia];
          }
      | None ->
          (* 首次访问，使用保守策略 *)
          BasicJSExecution
  end
end
```

### 无头浏览器集成

无头浏览器是实现 JavaScript 渲染的核心组件。设计良好的集成接口可以支持多种浏览器引擎，提供统一的抽象层：

```ocaml
(* 无头浏览器抽象接口 *)
module type BrowserEngine = sig
  type t
  type page
  type context
  
  (* 浏览器生命周期管理 *)
  val create : config:browser_config -> t Lwt.t
  val close : t -> unit Lwt.t
  
  (* 上下文管理（隔离的浏览器环境） *)
  val create_context : t -> ?options:context_options -> unit -> context Lwt.t
  val close_context : context -> unit Lwt.t
  
  (* 页面操作 *)
  val new_page : context -> page Lwt.t
  val goto : page -> url:string -> ?options:navigation_options -> unit -> response Lwt.t
  val content : page -> string Lwt.t
  val screenshot : page -> ?options:screenshot_options -> unit -> bytes Lwt.t
  
  (* JavaScript 执行 *)
  val evaluate : page -> script:string -> Yojson.Safe.t Lwt.t
  val wait_for : page -> condition:wait_condition -> ?timeout:float -> unit -> unit Lwt.t
  
  (* 网络控制 *)
  val intercept_requests : page -> (request -> request_action) -> unit
  val get_network_logs : page -> network_event list
end

(* Chromium 引擎实现 *)
module ChromiumEngine : BrowserEngine = struct
  type t = {
    process: Lwt_process.process_none;
    ws_endpoint: string;
    mutable contexts: context list;
  }
  
  and context = {
    id: string;
    browser: t;
    mutable pages: page list;
    options: context_options;
  }
  
  and page = {
    id: string;
    context: context;
    mutable url: string option;
    network_manager: NetworkManager.t;
    execution_context: ExecutionContext.t;
  }
  
  (* CDP (Chrome DevTools Protocol) 通信 *)
  module CDP = struct
    type method_name = string
    type params = Yojson.Safe.t
    type session_id = string
    
    type command = {
      id: int;
      method_: method_name;
      params: params;
      session_id: session_id option;
    }
    
    type response = 
      | Success of { id: int; result: Yojson.Safe.t }
      | Error of { id: int; error: string }
      | Event of { method_: string; params: Yojson.Safe.t }
    
    let send websocket command =
      let json = command_to_json command in
      Websocket.send websocket (Yojson.Safe.to_string json)
    
    let receive websocket =
      let%lwt message = Websocket.receive websocket in
      parse_response (Yojson.Safe.from_string message)
  end
  
  (* 启动浏览器进程 *)
  let create ~config =
    let args = [
      "--headless";
      "--disable-gpu";
      "--no-sandbox";  (* 容器环境需要 *)
      sprintf "--remote-debugging-port=%d" config.debug_port;
      sprintf "--user-data-dir=%s" config.user_data_dir;
    ] @ config.extra_args in
    
    let%lwt process = 
      Lwt_process.open_process_none 
        ("chromium", Array.of_list ("chromium" :: args)) in
    
    (* 等待浏览器启动 *)
    let%lwt ws_endpoint = wait_for_ws_endpoint config.debug_port in
    
    Lwt.return { process; ws_endpoint; contexts = [] }
  
  (* 页面导航与等待策略 *)
  let goto page ~url ~options () =
    (* 设置导航超时 *)
    let timeout = Option.value options.timeout ~default:30.0 in
    
    (* 监听网络事件 *)
    let%lwt () = CDP.send page.websocket {
      method_ = "Network.enable";
      params = `Assoc [];
      session_id = Some page.session_id;
    } in
    
    (* 开始导航 *)
    let%lwt navigation_id = CDP.send page.websocket {
      method_ = "Page.navigate";
      params = `Assoc ["url", `String url];
      session_id = Some page.session_id;
    } in
    
    (* 等待加载完成 *)
    let wait_strategy = Option.value options.wait_until 
      ~default:`DOMContentLoaded in
    
    match wait_strategy with
    | `Load ->
        wait_for_event page "Page.loadEventFired" timeout
    | `DOMContentLoaded ->
        wait_for_event page "Page.domContentLoadedEventFired" timeout
    | `NetworkIdle n ->
        wait_for_network_idle page n timeout
    | `NetworkAlmostIdle n ->
        wait_for_network_almost_idle page n timeout
  
  (* 智能等待机制 *)
  module SmartWait = struct
    type wait_result = 
      | Completed
      | Timeout
      | Error of string
    
    (* 组合多个等待条件 *)
    let wait_for_any page conditions timeout =
      let condition_promises = List.map (fun cond ->
        match cond with
        | ElementPresent selector ->
            wait_for_selector page selector timeout
        | ElementVisible selector ->
            wait_for_selector_visible page selector timeout
        | CustomFunction js_func ->
            poll_js_condition page js_func 100 timeout
        | NetworkIdle duration ->
            wait_for_network_idle page duration timeout
        | RequestFinished pattern ->
            wait_for_request page pattern timeout
      ) conditions in
      
      (* 返回第一个完成的条件 *)
      Lwt.pick condition_promises
    
    (* 自适应等待 *)
    let adaptive_wait page hints =
      let conditions = match hints with
        | `SPA -> [
            NetworkIdle 0.5;
            CustomFunction "() => window.__APP_READY__ === true";
          ]
        | `LazyLoad -> [
            ElementVisible ".content";
            NetworkIdle 1.0;
          ]
        | `InfiniteScroll -> [
            CustomFunction "() => document.body.scrollHeight > window.innerHeight";
            NetworkIdle 0.5;
          ]
        | _ -> [NetworkIdle 0.5]
      in
      
      wait_for_any page conditions 10.0
  end
end
```

### 渲染池管理

高效的渲染池管理是保证系统性能的关键。需要考虑浏览器实例的创建开销、内存占用、以及并发限制：

```ocaml
(* 渲染池管理 *)
module RenderingPool = struct
  type pool_config = {
    min_size: int;              (* 最小池大小 *)
    max_size: int;              (* 最大池大小 *)
    idle_timeout: float;        (* 空闲超时时间 *)
    max_uses_per_instance: int; (* 每个实例最大使用次数 *)
    memory_limit_mb: int;       (* 内存限制 *)
    scale_up_threshold: float;  (* 扩容阈值 *)
    scale_down_threshold: float; (* 缩容阈值 *)
  }
  
  type instance_state =
    | Idle of { since: float }
    | Busy of { 
        task_id: string; 
        started_at: float;
        client: string;
      }
    | Maintenance           (* 维护中 *)
    | Terminated           (* 已终止 *)
  
  type browser_instance = {
    id: string;
    engine: ChromiumEngine.t;
    mutable state: instance_state;
    mutable use_count: int;
    mutable total_render_time: float;
    mutable memory_usage: int;
    created_at: float;
  }
  
  type t = {
    config: pool_config;
    instances: (string, browser_instance) Hashtbl.t;
    available: browser_instance Queue.t;
    mutable stats: pool_stats;
    monitor: monitor_thread;
  }
  
  and pool_stats = {
    mutable total_requests: int;
    mutable cache_hits: int;
    mutable avg_wait_time: float;
    mutable avg_render_time: float;
    mutable instance_restarts: int;
  }
  
  (* 获取渲染实例 *)
  let acquire pool ~priority =
    let start_time = Unix.time () in
    
    (* 尝试从可用队列获取 *)
    let rec try_acquire retry_count =
      match Queue.take_opt pool.available with
      | Some instance when instance.state = Idle _ ->
          instance.state <- Busy { 
            task_id = generate_task_id (); 
            started_at = Unix.time ();
            client = get_client_id ();
          };
          Lwt.return instance
      
      | None when Hashtbl.length pool.instances < pool.config.max_size ->
          (* 创建新实例 *)
          let%lwt new_instance = create_instance pool in
          Hashtbl.add pool.instances new_instance.id new_instance;
          new_instance.state <- Busy { 
            task_id = generate_task_id (); 
            started_at = Unix.time ();
            client = get_client_id ();
          };
          Lwt.return new_instance
      
      | None ->
          (* 等待可用实例 *)
          if retry_count > 0 then begin
            let%lwt () = Lwt_unix.sleep 0.1 in
            try_acquire (retry_count - 1)
          end else
            failwith "No available rendering instances"
      
      | _ -> try_acquire retry_count
    in
    
    let%lwt instance = try_acquire 50 in (* 最多等待 5 秒 *)
    
    (* 更新统计 *)
    let wait_time = Unix.time () -. start_time in
    update_stats pool (`WaitTime wait_time);
    
    Lwt.return instance
  
  (* 归还实例 *)
  let release pool instance ~render_time ~success =
    instance.use_count <- instance.use_count + 1;
    instance.total_render_time <- instance.total_render_time +. render_time;
    
    (* 检查是否需要回收 *)
    if should_recycle instance pool.config then begin
      recycle_instance pool instance
    end else begin
      (* 清理页面状态 *)
      let%lwt () = cleanup_instance instance in
      instance.state <- Idle { since = Unix.time () };
      Queue.push instance pool.available;
      Lwt.return_unit
    end
  
  (* 动态扩缩容 *)
  module AutoScaling = struct
    type scaling_decision = 
      | ScaleUp of int    (* 增加实例数 *)
      | ScaleDown of int  (* 减少实例数 *)
      | NoChange
    
    let analyze_metrics pool =
      let active_count = count_active_instances pool in
      let total_count = Hashtbl.length pool.instances in
      let utilization = float active_count /. float total_count in
      
      let queue_length = Queue.length pool.available in
      let avg_wait = pool.stats.avg_wait_time in
      
      if utilization > pool.config.scale_up_threshold && 
         total_count < pool.config.max_size then
        ScaleUp (min 3 (pool.config.max_size - total_count))
      
      else if utilization < pool.config.scale_down_threshold &&
              total_count > pool.config.min_size &&
              queue_length > 2 then
        ScaleDown (min 2 (total_count - pool.config.min_size))
      
      else NoChange
    
    let execute_scaling pool decision =
      match decision with
      | ScaleUp n ->
          Lwt_list.iter_p (fun _ ->
            let%lwt instance = create_instance pool in
            Hashtbl.add pool.instances instance.id instance;
            Queue.push instance pool.available;
            Lwt.return_unit
          ) (List.init n (fun _ -> ()))
      
      | ScaleDown n ->
          (* 选择空闲时间最长的实例 *)
          let idle_instances = 
            Hashtbl.fold (fun _ inst acc ->
              match inst.state with
              | Idle { since } -> (inst, since) :: acc
              | _ -> acc
            ) pool.instances []
            |> List.sort (fun (_, t1) (_, t2) -> compare t1 t2)
            |> List.take n
            |> List.map fst
          in
          
          Lwt_list.iter_p (fun inst ->
            terminate_instance pool inst
          ) idle_instances
      
      | NoChange -> Lwt.return_unit
  end
  
  (* 健康检查与自愈 *)
  module HealthCheck = struct
    type health_status =
      | Healthy
      | Degraded of string
      | Unhealthy of string
    
    let check_instance instance =
      try%lwt
        (* 检查进程是否存活 *)
        let%lwt alive = process_is_alive instance.engine.process in
        if not alive then
          Lwt.return (Unhealthy "Process dead")
        else
          (* 检查内存使用 *)
          let%lwt memory = get_memory_usage instance.engine.process in
          if memory > 500 * 1024 * 1024 then  (* 500MB *)
            Lwt.return (Degraded "High memory usage")
          else
            (* 执行简单的 JS 测试 *)
            let%lwt result = ChromiumEngine.evaluate 
              instance.engine.current_page 
              ~script:"1 + 1" in
            if result = `Int 2 then
              Lwt.return Healthy
            else
              Lwt.return (Unhealthy "JS execution failed")
      with
      | exn -> 
          Lwt.return (Unhealthy (Printexc.to_string exn))
    
    let heal_instance pool instance status =
      match status with
      | Healthy -> Lwt.return_unit
      
      | Degraded reason ->
          (* 软重启：关闭所有页面，保持进程 *)
          Log.warn "Instance %s degraded: %s" instance.id reason;
          cleanup_instance instance
      
      | Unhealthy reason ->
          (* 硬重启：终止进程并重建 *)
          Log.error "Instance %s unhealthy: %s" instance.id reason;
          let%lwt () = terminate_instance pool instance in
          let%lwt new_instance = create_instance pool in
          Hashtbl.replace pool.instances instance.id new_instance;
          Lwt.return_unit
  end
  
  (* 渲染缓存 *)
  module RenderCache = struct
    type cache_key = {
      url: string;
      wait_condition: string;
      viewport: (int * int) option;
      cookies_hash: string option;
    }
    
    type cache_entry = {
      key: cache_key;
      content: string;
      screenshot: bytes option;
      rendered_at: float;
      render_time: float;
      size_bytes: int;
    }
    
    type t = {
      entries: (cache_key, cache_entry) Hashtbl.t;
      lru: cache_key Dllist.t;
      max_size_bytes: int;
      mutable current_size_bytes: int;
      ttl: float;
    }
    
    let make_key url options =
      {
        url;
        wait_condition = serialize_wait_condition options.wait_condition;
        viewport = options.viewport;
        cookies_hash = options.cookies |> Option.map hash_cookies;
      }
    
    let get cache key =
      match Hashtbl.find_opt cache.entries key with
      | None -> None
      | Some entry ->
          let age = Unix.time () -. entry.rendered_at in
          if age > cache.ttl then begin
            (* 过期，删除 *)
            evict_entry cache key;
            None
          end else begin
            (* 更新 LRU *)
            Dllist.remove cache.lru (Dllist.find cache.lru key);
            Dllist.add_last cache.lru key;
            Some entry
          end
    
    let put cache key content ~screenshot ~render_time =
      let size = String.length content + 
                 Option.value (Option.map Bytes.length screenshot) ~default:0 in
      
      (* 确保有足够空间 *)
      while cache.current_size_bytes + size > cache.max_size_bytes do
        match Dllist.take_first cache.lru with
        | None -> break
        | Some old_key -> evict_entry cache old_key
      done;
      
      let entry = {
        key;
        content;
        screenshot;
        rendered_at = Unix.time ();
        render_time;
        size_bytes = size;
      } in
      
      Hashtbl.replace cache.entries key entry;
      Dllist.add_last cache.lru key;
      cache.current_size_bytes <- cache.current_size_bytes + size
  end
end
```

### 资源优化策略

渲染服务的资源消耗很大，需要多层次的优化策略：

```ocaml
(* 资源优化策略 *)
module ResourceOptimization = struct
  (* 预渲染配置 *)
  type prerender_config = {
    block_resources: resource_type list;
    disable_javascript: bool;
    disable_images: bool;
    disable_css: bool;
    use_mobile_viewport: bool;
    network_conditions: network_preset option;
  }
  
  and resource_type = 
    | Script | Stylesheet | Image | Font 
    | Media | Websocket | XHR | Other of string
  
  and network_preset =
    | Fast3G
    | Slow3G  
    | Offline
    | Custom of {
        download_throughput: int;  (* bytes/sec *)
        upload_throughput: int;
        latency: int;  (* ms *)
      }
  
  (* 请求拦截器 *)
  module RequestInterceptor = struct
    type intercept_decision =
      | Allow
      | Block
      | Modify of request -> request
      | Respond of response
    
    let create_interceptor config =
      fun request ->
        let resource_type = infer_resource_type request in
        
        (* 基于配置的拦截规则 *)
        match resource_type with
        | Script when List.mem Script config.block_resources -> Block
        | Image when config.disable_images -> Block
        | Stylesheet when config.disable_css -> Block
        | Font when List.mem Font config.block_resources -> Block
        | Media -> Block  (* 总是阻止音视频 *)
        | _ ->
            (* 域名过滤 *)
            let domain = extract_domain request.url in
            if is_third_party_tracker domain then Block
            else if is_advertising_domain domain then Block
            else Allow
    
    (* 广告和跟踪器黑名单 *)
    let third_party_trackers = [
      "google-analytics.com";
      "googletagmanager.com";
      "facebook.com/tr";
      "doubleclick.net";
      "scorecardresearch.com";
      "quantserve.com";
      "outbrain.com";
      "taboola.com";
    ]
    
    let is_third_party_tracker domain =
      List.exists (fun tracker ->
        String.contains domain tracker
      ) third_party_trackers
  end
  
  (* 并行预取优化 *)
  module ParallelPrefetch = struct
    type prefetch_hint = {
      urls: string list;
      priority: [ `High | `Low ];
      resource_type: resource_type;
    }
    
    let analyze_page_links page =
      let%lwt links = ChromiumEngine.evaluate page
        ~script:{|
          Array.from(document.querySelectorAll('a[href]'))
            .map(a => ({
              href: a.href,
              visible: a.offsetParent !== null,
              text: a.textContent.trim(),
              area: a.offsetWidth * a.offsetHeight
            }))
            .filter(link => link.visible && link.area > 100)
            .sort((a, b) => b.area - a.area)
            .slice(0, 20)
            .map(link => link.href)
        |} in
      
      parse_url_list links
    
    (* 预测性渲染 *)
    let predictive_render pool urls =
      (* 基于历史数据预测可能需要渲染的页面 *)
      let predictions = urls |> List.filter_map (fun url ->
        let domain = extract_domain url in
        match DomainProfile.lookup domain with
        | Some { requires_js = true; avg_render_time; _ } ->
            Some (url, avg_render_time)
        | _ -> None
      ) |> List.sort (fun (_, t1) (_, t2) -> compare t1 t2) in
      
      (* 预渲染前 N 个最可能需要的页面 *)
      let to_prerender = List.take 5 predictions in
      
      Lwt_list.iter_p (fun (url, _) ->
        Lwt.async (fun () ->
          try%lwt
            let%lwt instance = RenderingPool.acquire pool ~priority:`Low in
            let%lwt _ = render_page instance url default_options in
            RenderingPool.release pool instance
          with _ -> Lwt.return_unit
        );
        Lwt.return_unit
      ) to_prerender
  end
  
  (* 内存管理优化 *)
  module MemoryManagement = struct
    type memory_pressure = Low | Medium | High | Critical
    
    let get_system_memory_pressure () =
      let%lwt available = get_available_memory () in
      let%lwt total = get_total_memory () in
      let usage_ratio = 1.0 -. (float available /. float total) in
      
      if usage_ratio < 0.6 then Low
      else if usage_ratio < 0.75 then Medium
      else if usage_ratio < 0.85 then High
      else Critical
    
    (* 根据内存压力调整策略 *)
    let adapt_strategy pressure config =
      match pressure with
      | Low -> config  (* 无需调整 *)
      
      | Medium ->
          { config with
            block_resources = Image :: Font :: config.block_resources;
            use_mobile_viewport = true;
          }
      
      | High ->
          { config with
            disable_images = true;
            disable_css = true;
            block_resources = Script :: Stylesheet :: Image :: Font :: [];
          }
      
      | Critical ->
          { config with
            disable_javascript = true;
            disable_images = true;
            disable_css = true;
            network_conditions = Some Offline;  (* 禁止新的网络请求 *)
          }
    
    (* 定期清理 *)
    let cleanup_routine pool =
      let rec loop () =
        let%lwt () = Lwt_unix.sleep 60.0 in  (* 每分钟检查 *)
        
        (* 清理空闲实例 *)
        let now = Unix.time () in
        let idle_instances = 
          Hashtbl.fold (fun _ inst acc ->
            match inst.state with
            | Idle { since } when now -. since > 300.0 -> (* 5分钟 *)
                inst :: acc
            | _ -> acc
          ) pool.instances []
        in
        
        let%lwt () = Lwt_list.iter_p (fun inst ->
          (* 执行垃圾回收 *)
          let%lwt () = ChromiumEngine.evaluate inst.engine.current_page
            ~script:"if (window.gc) window.gc();" in
          
          (* 清理缓存 *)
          let%lwt () = CDP.send inst.engine.websocket {
            method_ = "Network.clearBrowserCache";
            params = `Assoc [];
          } in
          
          Lwt.return_unit
        ) idle_instances in
        
        loop ()
      in
      Lwt.async loop
  end
  
  (* 批量渲染优化 *)
  module BatchRendering = struct
    type batch_request = {
      urls: (string * render_options) list;
      priority: priority;
      callback: batch_result -> unit;
    }
    
    and batch_result = {
      results: (string * render_result) list;
      total_time: float;
      parallel_factor: int;
    }
    
    (* 智能批处理调度 *)
    let schedule_batch pool requests =
      (* 按域名分组 *)
      let by_domain = group_by_domain requests in
      
      (* 计算最优并行度 *)
      let available_instances = Queue.length pool.available in
      let optimal_parallel = min available_instances (List.length by_domain) in
      
      (* 创建渲染任务 *)
      let tasks = List.map (fun (domain, urls) ->
        let shared_context = create_shared_context domain in
        
        Lwt.async (fun () ->
          let%lwt instance = RenderingPool.acquire pool ~priority:`Normal in
          
          (* 批量渲染同域名页面，复用连接和缓存 *)
          let%lwt results = Lwt_list.map_s (fun (url, options) ->
            let start = Unix.time () in
            try%lwt
              let%lwt content = render_with_context 
                instance shared_context url options in
              let render_time = Unix.time () -. start in
              Lwt.return (url, Success { content; render_time })
            with exn ->
              Lwt.return (url, Failure (Printexc.to_string exn))
          ) urls in
          
          let%lwt () = RenderingPool.release pool instance in
          Lwt.return results
        )
      ) by_domain in
      
      tasks
  end
end
```

## 5.6 爬虫系统的扩展性设计

### 水平扩展架构

构建可扩展的爬虫系统需要从架构层面考虑如何支持从单机到数千节点的平滑扩展。关键在于设计无状态的爬虫节点、中心化的协调服务、以及高效的任务分发机制。

```ocaml
(* 可扩展爬虫架构 *)
module ScalableCrawlerArchitecture = struct
  (* 系统组件角色定义 *)
  type node_role =
    | Coordinator      (* 协调节点：任务调度、状态管理 *)
    | Worker          (* 工作节点：执行爬取任务 *)
    | Storage         (* 存储节点：持久化爬取结果 *)
    | Monitor         (* 监控节点：系统监控和告警 *)
    | Gateway         (* 网关节点：对外接口 *)
  
  (* 节点能力描述 *)
  type node_capability = {
    role: node_role;
    capacity: capacity_spec;
    location: geo_location;
    network_bandwidth: int;  (* Mbps *)
    specialization: specialization list;
  }
  
  and capacity_spec = {
    cpu_cores: int;
    memory_gb: int;
    disk_gb: int;
    concurrent_connections: int;
  }
  
  and specialization =
    | JavaScriptRendering   (* 支持 JS 渲染 *)
    | HighBandwidth        (* 高带宽节点 *)
    | GeoSpecific of string (* 特定地理位置 *)
    | DomainExpert of string (* 特定域名专家 *)
  
  (* 服务发现机制 *)
  module ServiceDiscovery = struct
    type service_registry = {
      etcd_client: Etcd.client;
      namespace: string;
      ttl: int;  (* 服务注册 TTL *)
    }
    
    (* 节点注册 *)
    let register_node registry node_id capability =
      let key = Printf.sprintf "%s/nodes/%s" registry.namespace node_id in
      let value = capability_to_json capability in
      
      (* 带 TTL 的注册，需要定期续期 *)
      let%lwt lease = Etcd.lease_grant registry.etcd_client registry.ttl in
      let%lwt () = Etcd.put registry.etcd_client 
        ~key ~value ~lease:lease.id in
      
      (* 启动心跳线程 *)
      let rec heartbeat () =
        let%lwt () = Lwt_unix.sleep (float registry.ttl /. 2.0) in
        let%lwt () = Etcd.lease_keep_alive registry.etcd_client lease.id in
        heartbeat ()
      in
      Lwt.async heartbeat;
      
      Lwt.return lease
    
    (* 发现特定角色的节点 *)
    let discover_nodes registry role =
      let prefix = Printf.sprintf "%s/nodes/" registry.namespace in
      let%lwt all_nodes = Etcd.get_prefix registry.etcd_client prefix in
      
      all_nodes 
      |> List.filter_map (fun (key, value) ->
        try
          let capability = json_to_capability (Yojson.Safe.from_string value) in
          if capability.role = role then
            Some (extract_node_id key, capability)
          else None
        with _ -> None
      ) |> Lwt.return
    
    (* 监听节点变化 *)
    let watch_nodes registry role callback =
      let prefix = Printf.sprintf "%s/nodes/" registry.namespace in
      
      Etcd.watch registry.etcd_client prefix (fun event ->
        match event with
        | `Put (key, value) ->
            let node_id = extract_node_id key in
            let capability = json_to_capability (Yojson.Safe.from_string value) in
            if capability.role = role then
              callback (`NodeAdded (node_id, capability))
        
        | `Delete key ->
            let node_id = extract_node_id key in
            callback (`NodeRemoved node_id)
      )
  end
  
  (* 任务分发策略 *)
  module TaskDistribution = struct
    type distribution_strategy =
      | RoundRobin           (* 轮询分发 *)
      | LeastConnections     (* 最少连接数 *)
      | WeightedRandom       (* 加权随机 *)
      | ConsistentHashing    (* 一致性哈希 *)
      | CapabilityBased      (* 基于能力的分发 *)
      | GeographyAware       (* 地理位置感知 *)
    
    type task_router = {
      strategy: distribution_strategy;
      nodes: (string, node_info) Hashtbl.t;
      mutable stats: router_stats;
    }
    
    and node_info = {
      capability: node_capability;
      mutable current_load: int;
      mutable total_assigned: int;
      mutable success_rate: float;
      mutable avg_latency: float;
    }
    
    and router_stats = {
      mutable total_routed: int;
      mutable routing_errors: int;
      mutable rebalance_count: int;
    }
    
    (* 选择最佳节点 *)
    let select_node router task =
      match router.strategy with
      | RoundRobin ->
          let nodes = Hashtbl.to_seq router.nodes |> List.of_seq in
          let idx = router.stats.total_routed mod (List.length nodes) in
          List.nth nodes idx |> fst
      
      | LeastConnections ->
          Hashtbl.fold (fun id info (best_id, best_load) ->
            if info.current_load < best_load then
              (id, info.current_load)
            else
              (best_id, best_load)
          ) router.nodes ("", max_int) |> fst
      
      | WeightedRandom ->
          (* 根据节点能力和当前负载计算权重 *)
          let weights = Hashtbl.fold (fun id info acc ->
            let capacity_score = 
              float info.capability.capacity.cpu_cores *.
              float info.capability.capacity.concurrent_connections in
            let load_factor = 1.0 /. (1.0 +. float info.current_load) in
            let weight = capacity_score *. load_factor *. info.success_rate in
            (id, weight) :: acc
          ) router.nodes [] in
          
          weighted_random_choice weights
      
      | ConsistentHashing ->
          let task_hash = hash_task task in
          ConsistentHash.find_node router.consistent_hash task_hash
      
      | CapabilityBased ->
          (* 匹配任务需求和节点能力 *)
          select_by_capability router task
      
      | GeographyAware ->
          (* 选择地理位置最近的节点 *)
          select_by_geography router task
    
    (* 任务亲和性 *)
    module TaskAffinity = struct
      type affinity_rule =
        | DomainAffinity      (* 同域名任务分配到同一节点 *)
        | SessionAffinity     (* 会话保持 *)
        | DataLocalityAffinity (* 数据本地性 *)
      
      let apply_affinity router task rule =
        match rule with
        | DomainAffinity ->
            let domain = extract_domain task.url in
            let node_id = 
              match Hashtbl.find_opt router.domain_mapping domain with
              | Some id when Hashtbl.mem router.nodes id -> id
              | _ ->
                  let id = select_node router task in
                  Hashtbl.add router.domain_mapping domain id;
                  id
            in
            node_id
        
        | SessionAffinity ->
            match task.session_id with
            | Some sid ->
                (match Hashtbl.find_opt router.session_mapping sid with
                | Some id when Hashtbl.mem router.nodes id -> id
                | _ -> select_node router task)
            | None -> select_node router task
        
        | DataLocalityAffinity ->
            (* 选择数据所在的节点或最近的节点 *)
            select_by_data_locality router task
    end
  end
  
  (* 弹性伸缩控制器 *)
  module ElasticScaling = struct
    type scaling_policy = {
      min_nodes: int;
      max_nodes: int;
      target_utilization: float;  (* 0.0 - 1.0 *)
      scale_up_threshold: float;
      scale_down_threshold: float;
      cool_down_period: float;    (* 秒 *)
    }
    
    type scaling_metrics = {
      cpu_usage: float;
      memory_usage: float;
      queue_length: int;
      avg_task_latency: float;
      error_rate: float;
    }
    
    type controller = {
      policy: scaling_policy;
      mutable last_scale_time: float;
      mutable current_nodes: int;
      metrics_window: scaling_metrics list ref;
    }
    
    (* 计算是否需要伸缩 *)
    let compute_scaling_decision controller current_metrics =
      let now = Unix.time () in
      
      (* 检查冷却期 *)
      if now -. controller.last_scale_time < controller.policy.cool_down_period then
        `NoChange
      else
        (* 计算综合利用率 *)
        let utilization = 
          (current_metrics.cpu_usage *. 0.4 +.
           current_metrics.memory_usage *. 0.3 +.
           (float current_metrics.queue_length /. 1000.0) *. 0.2 +.
           (current_metrics.avg_task_latency /. 10.0) *. 0.1) in
        
        if utilization > controller.policy.scale_up_threshold &&
           controller.current_nodes < controller.policy.max_nodes then
          `ScaleUp (calculate_scale_up_count controller utilization)
        
        else if utilization < controller.policy.scale_down_threshold &&
                controller.current_nodes > controller.policy.min_nodes then
          `ScaleDown (calculate_scale_down_count controller utilization)
        
        else
          `NoChange
    
    (* 执行伸缩操作 *)
    let execute_scaling controller decision =
      match decision with
      | `ScaleUp count ->
          Log.info "Scaling up by %d nodes" count;
          controller.last_scale_time <- Unix.time ();
          controller.current_nodes <- controller.current_nodes + count;
          
          (* 通过容器编排系统增加节点 *)
          Kubernetes.scale_deployment "crawler-worker" 
            controller.current_nodes
      
      | `ScaleDown count ->
          Log.info "Scaling down by %d nodes" count;
          controller.last_scale_time <- Unix.time ();
          controller.current_nodes <- controller.current_nodes - count;
          
          (* 优雅关闭：先停止分配新任务，等待现有任务完成 *)
          graceful_shutdown_nodes count
      
      | `NoChange -> Lwt.return_unit
    
    (* 预测性伸缩 *)
    module PredictiveScaling = struct
      type prediction_model = {
        historical_data: (float * scaling_metrics) list;
        seasonality_period: float;  (* 秒 *)
        prediction_horizon: float;  (* 预测未来多久 *)
      }
      
      (* 基于历史数据预测负载 *)
      let predict_load model current_time =
        (* 简单的季节性模型 *)
        let period_ago = current_time -. model.seasonality_period in
        
        let similar_time_data = 
          model.historical_data |> List.filter (fun (timestamp, _) ->
            abs_float (timestamp -. period_ago) < 300.0  (* 5分钟窗口 *)
          ) in
        
        match similar_time_data with
        | [] -> None
        | data ->
            let avg_metrics = average_metrics (List.map snd data) in
            Some avg_metrics
      
      (* 提前触发伸缩 *)
      let proactive_scale controller model =
        let current_time = Unix.time () in
        let future_time = current_time +. model.prediction_horizon in
        
        match predict_load model future_time with
        | None -> Lwt.return_unit
        | Some predicted_metrics ->
            let decision = compute_scaling_decision controller predicted_metrics in
            if decision <> `NoChange then begin
              Log.info "Predictive scaling triggered";
              execute_scaling controller decision
            end else
              Lwt.return_unit
    end
  end
end
```

### 负载均衡策略

高效的负载均衡是保证爬虫系统性能和可靠性的关键：

```ocaml
(* 负载均衡策略实现 *)
module LoadBalancing = struct
  (* 负载指标 *)
  type load_metrics = {
    active_connections: int;
    cpu_usage: float;
    memory_usage: float;
    network_bandwidth_usage: float;
    queue_depth: int;
    response_time_ms: float;
    error_rate: float;
    custom_metrics: (string * float) list;
  }
  
  (* 负载均衡算法 *)
  module Algorithms = struct
    (* 加权轮询 *)
    module WeightedRoundRobin = struct
      type t = {
        nodes: (string * int * int ref) array;  (* id, weight, current *)
        mutable current_index: int;
        mutable current_weight: int;
        total_weight: int;
      }
      
      let create nodes_with_weights =
        let nodes = Array.of_list (List.map (fun (id, w) -> 
          (id, w, ref 0)) nodes_with_weights) in
        let total = List.fold_left (fun acc (_, w) -> acc + w) 0 
          nodes_with_weights in
        { nodes; current_index = 0; current_weight = 0; total_weight = total }
      
      let next_node t =
        let rec find_next () =
          t.current_index <- t.current_index mod (Array.length t.nodes);
          let (id, weight, current) = t.nodes.(t.current_index) in
          
          if t.current_weight <= 0 then
            t.current_weight <- t.total_weight;
          
          current := !current + weight;
          t.current_weight <- t.current_weight - weight;
          t.current_index <- t.current_index + 1;
          
          if !current >= weight then begin
            current := !current - t.total_weight;
            id
          end else
            find_next ()
        in
        find_next ()
    end
    
    (* 最少连接数 *)
    module LeastConnections = struct
      type t = {
        connections: (string, int ref) Hashtbl.t;
        mutex: Mutex.t;
      }
      
      let select t =
        Mutex.lock t.mutex;
        let node = 
          Hashtbl.fold (fun id count_ref (best_id, best_count) ->
            if !count_ref < best_count then
              (id, !count_ref)
            else
              (best_id, best_count)
          ) t.connections ("", max_int) |> fst
        in
        (match Hashtbl.find_opt t.connections node with
        | Some count_ref -> incr count_ref
        | None -> ());
        Mutex.unlock t.mutex;
        node
      
      let release t node_id =
        Mutex.lock t.mutex;
        (match Hashtbl.find_opt t.connections node_id with
        | Some count_ref when !count_ref > 0 -> decr count_ref
        | _ -> ());
        Mutex.unlock t.mutex
    end
    
    (* 一致性哈希 *)
    module ConsistentHash = struct
      type t = {
        ring: (int32 * string) array;
        hash_func: string -> int32;
        virtual_nodes: int;
      }
      
      let create nodes ?(virtual_nodes=150) ?(hash_func=default_hash) () =
        let points = List.fold_left (fun acc node ->
          let rec add_virtual acc i =
            if i >= virtual_nodes then acc
            else
              let key = Printf.sprintf "%s#%d" node i in
              let hash = hash_func key in
              (hash, node) :: add_virtual acc (i + 1)
          in
          add_virtual acc 0
        ) [] nodes in
        
        let ring = Array.of_list (List.sort (fun (h1, _) (h2, _) -> 
          Int32.compare h1 h2) points) in
        { ring; hash_func; virtual_nodes }
      
      let find_node t key =
        let hash = t.hash_func key in
        
        (* 二分查找 *)
        let rec binary_search low high =
          if low > high then
            snd t.ring.(0)  (* 环绕到第一个节点 *)
          else
            let mid = (low + high) / 2 in
            let (mid_hash, mid_node) = t.ring.(mid) in
            
            if Int32.compare hash mid_hash <= 0 then
              if mid = 0 || Int32.compare hash (fst t.ring.(mid - 1)) > 0 then
                mid_node
              else
                binary_search low (mid - 1)
            else
              binary_search (mid + 1) high
        in
        
        binary_search 0 (Array.length t.ring - 1)
    end
    
    (* 基于响应时间的动态负载均衡 *)
    module ResponseTimeBalancer = struct
      type node_stats = {
        mutable total_requests: int;
        mutable total_response_time: float;
        mutable recent_response_times: float Queue.t;
        mutable weight: float;
      }
      
      type t = {
        nodes: (string, node_stats) Hashtbl.t;
        window_size: int;
        mutable total_weight: float;
      }
      
      let update_stats t node_id response_time =
        match Hashtbl.find_opt t.nodes node_id with
        | None -> ()
        | Some stats ->
            stats.total_requests <- stats.total_requests + 1;
            stats.total_response_time <- stats.total_response_time +. response_time;
            
            Queue.add response_time stats.recent_response_times;
            if Queue.length stats.recent_response_times > t.window_size then
              ignore (Queue.take stats.recent_response_times);
            
            (* 更新权重：响应时间越短，权重越高 *)
            let avg_recent = 
              let sum = Queue.fold (+.) 0.0 stats.recent_response_times in
              sum /. float (Queue.length stats.recent_response_times) in
            
            let old_weight = stats.weight in
            stats.weight <- 1.0 /. (1.0 +. avg_recent /. 1000.0);
            t.total_weight <- t.total_weight -. old_weight +. stats.weight
      
      let select t =
        if t.total_weight <= 0.0 then
          (* 退化到随机选择 *)
          let nodes = Hashtbl.to_seq_keys t.nodes |> List.of_seq in
          List.nth nodes (Random.int (List.length nodes))
        else
          (* 加权随机选择 *)
          let r = Random.float t.total_weight in
          let rec find_node acc = function
            | [] -> failwith "No node found"
            | (id, stats) :: rest ->
                let acc' = acc +. stats.weight in
                if acc' >= r then id
                else find_node acc' rest
          in
          find_node 0.0 (Hashtbl.to_seq t.nodes |> List.of_seq)
    end
  end
  
  (* 健康检查 *)
  module HealthCheck = struct
    type check_type =
      | HTTPCheck of { path: string; expected_status: int }
      | TCPCheck of { port: int }
      | ScriptCheck of { script: string }
      | GRPCCheck of { service: string; method: string }
    
    type health_status =
      | Healthy
      | Unhealthy of { reason: string; since: float }
      | Degraded of { reason: string; weight_penalty: float }
    
    type health_checker = {
      check_type: check_type;
      interval: float;
      timeout: float;
      healthy_threshold: int;
      unhealthy_threshold: int;
      mutable consecutive_failures: int;
      mutable consecutive_successes: int;
      mutable status: health_status;
    }
    
    let perform_check checker node_address =
      match checker.check_type with
      | HTTPCheck { path; expected_status } ->
          let url = Printf.sprintf "http://%s%s" node_address path in
          (try%lwt
            let%lwt (resp, _) = Cohttp_lwt_unix.Client.get 
              (Uri.of_string url) in
            let status = Cohttp.Code.code_of_status 
              (Cohttp.Response.status resp) in
            Lwt.return (status = expected_status)
          with _ -> Lwt.return false)
      
      | TCPCheck { port } ->
          (try%lwt
            let%lwt socket = Lwt_unix.socket Unix.PF_INET Unix.SOCK_STREAM 0 in
            let addr = Unix.ADDR_INET (Unix.inet_addr_of_string 
              (extract_ip node_address), port) in
            let%lwt () = Lwt_unix.connect socket addr in
            let%lwt () = Lwt_unix.close socket in
            Lwt.return true
          with _ -> Lwt.return false)
      
      | ScriptCheck { script } ->
          let%lwt exit_code = run_health_script script node_address in
          Lwt.return (exit_code = 0)
      
      | GRPCCheck { service; method } ->
          check_grpc_health node_address service method
    
    let update_status checker success =
      if success then begin
        checker.consecutive_failures <- 0;
        checker.consecutive_successes <- checker.consecutive_successes + 1;
        
        if checker.consecutive_successes >= checker.healthy_threshold then
          checker.status <- Healthy
      end else begin
        checker.consecutive_successes <- 0;
        checker.consecutive_failures <- checker.consecutive_failures + 1;
        
        if checker.consecutive_failures >= checker.unhealthy_threshold then
          checker.status <- Unhealthy { 
            reason = "Health check failed"; 
            since = Unix.time () 
          }
      end
  end
  
  (* 会话保持 *)
  module SessionAffinity = struct
    type affinity_method =
      | SourceIP
      | Cookie of string
      | Header of string
      | QueryParam of string
    
    type session_table = {
      method_: affinity_method;
      sessions: (string, string * float) Hashtbl.t;  (* key -> (node_id, last_used) *)
      ttl: float;
      mutex: Mutex.t;
    }
    
    let extract_session_key t request =
      match t.method_ with
      | SourceIP -> request.client_ip
      | Cookie name -> 
          extract_cookie request.headers name |> Option.value ~default:""
      | Header name ->
          extract_header request.headers name |> Option.value ~default:""
      | QueryParam name ->
          extract_query_param request.url name |> Option.value ~default:""
    
    let get_affinity_node t request =
      let key = extract_session_key t request in
      if key = "" then None
      else begin
        Mutex.lock t.mutex;
        let result = 
          match Hashtbl.find_opt t.sessions key with
          | Some (node_id, last_used) ->
              let now = Unix.time () in
              if now -. last_used < t.ttl then begin
                Hashtbl.replace t.sessions key (node_id, now);
                Some node_id
              end else begin
                Hashtbl.remove t.sessions key;
                None
              end
          | None -> None
        in
        Mutex.unlock t.mutex;
        result
      end
    
    let set_affinity_node t request node_id =
      let key = extract_session_key t request in
      if key <> "" then begin
        Mutex.lock t.mutex;
        Hashtbl.replace t.sessions key (node_id, Unix.time ());
        Mutex.unlock t.mutex
      end
  end
end
```

### 监控与调试

全面的监控和调试能力是维护大规模爬虫系统的基础：

```ocaml
(* 监控与调试系统 *)
module MonitoringAndDebugging = struct
  (* 指标收集 *)
  module Metrics = struct
    type metric_type =
      | Counter of { value: int64 ref; labels: (string * string) list }
      | Gauge of { value: float ref; labels: (string * string) list }
      | Histogram of { buckets: float array; values: int array; sum: float ref }
      | Summary of { quantiles: float array; window: float; values: float Queue.t }
    
    type metric_registry = {
      metrics: (string, metric_type) Hashtbl.t;
      exporters: exporter list;
    }
    
    and exporter =
      | PrometheusExporter of { port: int; path: string }
      | StatsDExporter of { host: string; port: int }
      | OpenTelemetryExporter of { endpoint: string }
    
    (* 爬虫特定指标 *)
    let register_crawler_metrics registry =
      (* 吞吐量指标 *)
      register_counter registry "crawler_urls_total" 
        ~help:"Total number of URLs crawled"
        ~labels:["status"; "domain"];
      
      register_histogram registry "crawler_response_time_seconds"
        ~help:"Response time distribution"
        ~buckets:[|0.1; 0.5; 1.0; 2.0; 5.0; 10.0|];
      
      register_gauge registry "crawler_active_connections"
        ~help:"Current number of active connections";
      
      (* 错误率指标 *)
      register_counter registry "crawler_errors_total"
        ~help:"Total number of errors"
        ~labels:["error_type"; "domain"];
      
      (* 资源使用指标 *)
      register_gauge registry "crawler_memory_usage_bytes"
        ~help:"Memory usage in bytes";
      
      register_gauge registry "crawler_cpu_usage_percent"
        ~help:"CPU usage percentage";
      
      (* 队列指标 *)
      register_gauge registry "crawler_queue_depth"
        ~help:"Number of URLs in queue"
        ~labels:["priority"];
      
      (* 去重效率指标 *)
      register_counter registry "crawler_duplicates_found"
        ~help:"Number of duplicate URLs detected";
      
      register_gauge registry "crawler_bloom_filter_saturation"
        ~help:"Bloom filter saturation percentage"
    
    (* 自定义指标收集器 *)
    module CustomCollector = struct
      type collector = unit -> metric_sample list
      
      and metric_sample = {
        name: string;
        value: metric_value;
        timestamp: float;
        labels: (string * string) list;
      }
      
      and metric_value = 
        | CounterValue of int64
        | GaugeValue of float
        | HistogramValue of float
      
      (* 域名级别的统计 *)
      let domain_stats_collector crawler_state =
        Hashtbl.fold (fun domain stats acc ->
          let samples = [
            { name = "crawler_domain_urls_total";
              value = CounterValue stats.total_urls;
              timestamp = Unix.time ();
              labels = [("domain", domain)] };
            
            { name = "crawler_domain_success_rate";
              value = GaugeValue stats.success_rate;
              timestamp = Unix.time ();
              labels = [("domain", domain)] };
            
            { name = "crawler_domain_avg_response_time";
              value = GaugeValue stats.avg_response_time;
              timestamp = Unix.time ();
              labels = [("domain", domain)] };
          ] in
          samples @ acc
        ) crawler_state.domain_stats []
    end
  end
  
  (* 分布式追踪 *)
  module DistributedTracing = struct
    type span_context = {
      trace_id: string;
      span_id: string;
      parent_span_id: string option;
      baggage: (string * string) list;
    }
    
    type span = {
      context: span_context;
      operation_name: string;
      start_time: float;
      mutable end_time: float option;
      mutable tags: (string * tag_value) list;
      mutable logs: log_entry list;
      mutable status: span_status;
    }
    
    and tag_value = String of string | Int of int | Float of float | Bool of bool
    and span_status = Ok | Error of string
    
    and log_entry = {
      timestamp: float;
      fields: (string * string) list;
    }
    
    (* 爬虫操作的追踪 *)
    let trace_crawl_operation tracer url =
      let span = Tracer.start_span tracer "crawl_url" in
      
      Span.set_tag span "url" (String url);
      Span.set_tag span "domain" (String (extract_domain url));
      
      (* 追踪各个阶段 *)
      let%lwt download_span = Tracer.start_child_span span "download" in
      let%lwt content = 
        try%lwt
          let%lwt result = download_with_span download_span url in
          Span.finish download_span;
          Lwt.return result
        with exn ->
          Span.set_error download_span (Printexc.to_string exn);
          Span.finish download_span;
          raise exn
      in
      
      let%lwt parse_span = Tracer.start_child_span span "parse" in
      let%lwt parsed = parse_with_span parse_span content in
      Span.finish parse_span;
      
      let%lwt store_span = Tracer.start_child_span span "store" in
      let%lwt () = store_with_span store_span parsed in
      Span.finish store_span;
      
      Span.finish span;
      Lwt.return parsed
    
    (* 跨服务追踪传播 *)
    let inject_trace_context span headers =
      let context = Span.get_context span in
      headers @ [
        ("X-Trace-Id", context.trace_id);
        ("X-Span-Id", context.span_id);
        ("X-Parent-Span-Id", 
         Option.value context.parent_span_id ~default:"");
      ]
    
    let extract_trace_context headers =
      let trace_id = List.assoc_opt "X-Trace-Id" headers in
      let span_id = List.assoc_opt "X-Span-Id" headers in
      let parent_span_id = List.assoc_opt "X-Parent-Span-Id" headers in
      
      match trace_id, span_id with
      | Some tid, Some sid ->
          Some { 
            trace_id = tid; 
            span_id = generate_span_id (); 
            parent_span_id = Some sid;
            baggage = [] 
          }
      | _ -> None
  end
  
  (* 日志聚合 *)
  module LogAggregation = struct
    type log_level = Debug | Info | Warn | Error | Fatal
    
    type structured_log = {
      timestamp: float;
      level: log_level;
      message: string;
      fields: (string * Yojson.Safe.t) list;
      trace_context: DistributedTracing.span_context option;
    }
    
    type log_sink =
      | ConsoleSink of { format: log_format }
      | FileSink of { path: string; rotation: rotation_policy }
      | RemoteSink of { endpoint: string; batch_size: int }
      | ElasticsearchSink of { 
          nodes: string list; 
          index_pattern: string;
          bulk_size: int;
        }
    
    and log_format = Text | JSON | Logfmt
    
    and rotation_policy =
      | SizeBasedRotation of int  (* bytes *)
      | TimeBasedRotation of float  (* seconds *)
      | NoRotation
    
    (* 爬虫专用日志记录器 *)
    let create_crawler_logger sinks =
      let logger = Logger.create sinks in
      
      (* 添加默认字段 *)
      Logger.with_fields logger [
        ("service", `String "crawler");
        ("node_id", `String (get_node_id ()));
        ("version", `String (get_version ()));
      ]
    
    (* 结构化日志示例 *)
    let log_crawl_result logger url result =
      match result with
      | Ok response ->
          Logger.info logger "URL crawled successfully" [
            ("url", `String url);
            ("status_code", `Int response.status_code);
            ("content_length", `Int response.content_length);
            ("response_time_ms", `Float response.time_ms);
          ]
      
      | Error err ->
          Logger.error logger "Failed to crawl URL" [
            ("url", `String url);
            ("error", `String (error_to_string err));
            ("error_type", `String (classify_error err));
          ]
  end
  
  (* 调试工具 *)
  module DebugTools = struct
    (* 请求记录器 *)
    module RequestRecorder = struct
      type recording = {
        id: string;
        start_time: float;
        mutable requests: recorded_request list;
        filters: request_filter list;
        max_requests: int;
      }
      
      and recorded_request = {
        timestamp: float;
        url: string;
        method_: string;
        headers: (string * string) list;
        body: string option;
        response: recorded_response option;
      }
      
      and recorded_response = {
        status_code: int;
        headers: (string * string) list;
        body: string;
        time_ms: float;
      }
      
      and request_filter =
        | DomainFilter of string
        | URLPatternFilter of Re.re
        | StatusCodeFilter of int list
        | HeaderFilter of string * string
      
      let record_request recording request response =
        if should_record recording.filters request then begin
          let recorded = {
            timestamp = Unix.time ();
            url = request.url;
            method_ = request.method_;
            headers = request.headers;
            body = request.body;
            response = Some {
              status_code = response.status_code;
              headers = response.headers;
              body = response.body;
              time_ms = response.time_ms;
            };
          } in
          
          recording.requests <- recorded :: recording.requests;
          
          (* 限制记录数量 *)
          if List.length recording.requests > recording.max_requests then
            recording.requests <- List.take recording.max_requests 
              recording.requests
        end
      
      let export_har recording =
        let har = {
          log = {
            version = "1.2";
            creator = { name = "Crawler"; version = "1.0" };
            entries = List.map request_to_har_entry recording.requests;
          }
        } in
        har_to_json har
    end
    
    (* 性能分析 *)
    module Profiler = struct
      type profile_data = {
        samples: profile_sample list;
        duration: float;
        sample_rate: int;
      }
      
      and profile_sample = {
        timestamp: float;
        stack_trace: string list;
        cpu_time: float;
        allocations: int;
      }
      
      let profile_crawler_operation ~duration ~sample_rate f =
        let profiler = Profiler.create ~sample_rate in
        Profiler.start profiler;
        
        let result = 
          try
            let r = f () in
            Profiler.stop profiler;
            Ok r
          with exn ->
            Profiler.stop profiler;
            Error exn
        in
        
        let profile_data = Profiler.get_data profiler in
        (result, profile_data)
      
      (* 火焰图生成 *)
      let generate_flamegraph profile_data =
        let stacks = profile_data.samples |> List.map (fun sample ->
          (sample.stack_trace, sample.cpu_time)
        ) in
        
        FlameGraph.generate stacks
          ~title:"Crawler CPU Profile"
          ~width:1200
          ~height:600
    end
    
    (* 诊断端点 *)
    module DiagnosticEndpoints = struct
      let setup_debug_server port =
        let app = App.create () in
        
        (* 健康检查 *)
        App.get app "/health" (fun _req ->
          let status = check_system_health () in
          Response.json (health_to_json status)
        );
        
        (* 性能指标 *)
        App.get app "/metrics" (fun _req ->
          let metrics = collect_all_metrics () in
          Response.text (metrics_to_prometheus metrics)
        );
        
        (* 运行时信息 *)
        App.get app "/debug/runtime" (fun _req ->
          let info = {
            uptime = Unix.time () -. start_time;
            goroutines = get_thread_count ();
            memory = Gc.stat ();
            version = get_version ();
          } in
          Response.json (runtime_info_to_json info)
        );
        
        (* pprof 兼容端点 *)
        App.get app "/debug/pprof/heap" (fun _req ->
          let profile = get_heap_profile () in
          Response.binary profile ~content_type:"application/octet-stream"
        );
        
        App.get app "/debug/pprof/cpu" (fun req ->
          let duration = get_query_param req "seconds" 
            |> Option.value ~default:"30" 
            |> int_of_string in
          let profile = get_cpu_profile ~duration in
          Response.binary profile ~content_type:"application/octet-stream"
        );
        
        (* 实时日志流 *)
        App.get app "/debug/logs" (fun req ->
          let level = get_query_param req "level" 
            |> Option.value ~default:"info" in
          Response.stream (stream_logs ~level)
        );
        
        App.listen app port
    end
  end
end
```

### 反爬虫应对策略

面对日益复杂的反爬虫机制，需要设计灵活的应对策略：

```ocaml
(* 反爬虫应对策略 *)
module AntiScrapingStrategies = struct
  (* 检测到的反爬虫机制类型 *)
  type anti_scraping_mechanism =
    | RateLimiting of { requests_per_second: float; burst: int }
    | UserAgentBlocking of { blocked_patterns: string list }
    | IPBlocking of { blocked_ips: string list; block_duration: float }
    | CaptchaChallenge of { captcha_type: captcha_type }
    | JavaScriptChallenge of { script_hash: string }
    | FingerprintDetection of { required_features: string list }
    | BehaviorAnalysis of { suspicious_patterns: behavior_pattern list }
    | CloudflareProtection
    | HoneypotTraps of { trap_urls: string list }
  
  and captcha_type = 
    | ImageCaptcha | ReCaptchaV2 | ReCaptchaV3 | HCaptcha | CustomCaptcha
  
  and behavior_pattern =
    | TooFastNavigation
    | NoMouseMovement
    | LinearScrolling
    | MissingReferrer
    | SuspiciousTimingPattern
  
  (* 反检测策略 *)
  module EvasionTechniques = struct
    (* 请求头伪装 *)
    module HeaderSpoofing = struct
      type header_profile = {
        user_agents: string list;
        accept_languages: string list;
        accept_encodings: string list;
        referer_patterns: (string * string) list;  (* URL pattern -> referer *)
      }
      
      let real_browser_profiles = [
        { (* Chrome on Windows *)
          user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36";
          ];
          accept_languages = ["en-US,en;q=0.9"; "en-GB,en;q=0.9"];
          accept_encodings = ["gzip, deflate, br"];
          referer_patterns = [];
        };
        { (* Firefox on macOS *)
          user_agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0";
          ];
          accept_languages = ["en-US,en;q=0.5"];
          accept_encodings = ["gzip, deflate, br"];
          referer_patterns = [];
        };
      ]
      
      let generate_headers profile url previous_url =
        let ua = List.nth profile.user_agents 
          (Random.int (List.length profile.user_agents)) in
        let lang = List.nth profile.accept_languages
          (Random.int (List.length profile.accept_languages)) in
        
        let base_headers = [
          ("User-Agent", ua);
          ("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8");
          ("Accept-Language", lang);
          ("Accept-Encoding", List.hd profile.accept_encodings);
          ("DNT", "1");
          ("Connection", "keep-alive");
          ("Upgrade-Insecure-Requests", "1");
        ] in
        
        (* 添加 referer *)
        let headers_with_referer = 
          match previous_url with
          | Some ref_url -> ("Referer", ref_url) :: base_headers
          | None -> base_headers
        in
        
        (* 随机化顺序，模拟真实浏览器 *)
        shuffle_list headers_with_referer
    end
    
    (* 浏览器指纹伪装 *)
    module FingerprintSpoofing = struct
      type browser_fingerprint = {
        screen_resolution: int * int;
        color_depth: int;
        timezone_offset: int;
        plugins: plugin_info list;
        fonts: string list;
        canvas_hash: string;
        webgl_vendor: string;
        audio_context_hash: string;
      }
      
      and plugin_info = {
        name: string;
        filename: string;
        description: string;
      }
      
      let generate_realistic_fingerprint () =
        let resolutions = [
          (1920, 1080); (1366, 768); (1440, 900); 
          (1536, 864); (1280, 720); (2560, 1440)
        ] in
        
        {
          screen_resolution = List.nth resolutions 
            (Random.int (List.length resolutions));
          color_depth = 24;  (* 最常见 *)
          timezone_offset = -420;  (* PST *)
          plugins = [
            { name = "Chrome PDF Plugin";
              filename = "internal-pdf-viewer";
              description = "Portable Document Format" };
            { name = "Chrome PDF Viewer";
              filename = "mhjfbmdgcfjbbpaeojofohoefgiehjai";
              description = "" };
          ];
          fonts = standard_font_list ();
          canvas_hash = generate_canvas_fingerprint ();
          webgl_vendor = "Intel Inc.";
          audio_context_hash = generate_audio_fingerprint ();
        }
      
      let inject_fingerprint page fingerprint =
        let script = Printf.sprintf {|
          // Override screen properties
          Object.defineProperty(screen, 'width', { get: () => %d });
          Object.defineProperty(screen, 'height', { get: () => %d });
          Object.defineProperty(screen, 'colorDepth', { get: () => %d });
          
          // Override timezone
          Date.prototype.getTimezoneOffset = function() { return %d; };
          
          // Override plugins
          Object.defineProperty(navigator, 'plugins', {
            get: () => {
              return %s;
            }
          });
          
          // Override WebGL
          const getParameter = WebGLRenderingContext.prototype.getParameter;
          WebGLRenderingContext.prototype.getParameter = function(parameter) {
            if (parameter === 37445) return '%s';  // VENDOR
            if (parameter === 37446) return 'Generic Renderer';  // RENDERER
            return getParameter.apply(this, arguments);
          };
        |} 
          (fst fingerprint.screen_resolution)
          (snd fingerprint.screen_resolution)
          fingerprint.color_depth
          fingerprint.timezone_offset
          (plugins_to_js fingerprint.plugins)
          fingerprint.webgl_vendor
        in
        
        Page.evaluate_on_new_document page script
    end
    
    (* 行为模拟 *)
    module BehaviorSimulation = struct
      type human_behavior = {
        mouse_movements: mouse_path list;
        scroll_patterns: scroll_event list;
        typing_speed: typing_pattern;
        reading_time: float -> float;  (* content_length -> time *)
        navigation_delays: float * float;  (* min, max *)
      }
      
      and mouse_path = {
        points: (float * float) list;
        duration: float;
      }
      
      and scroll_event = {
        delta_y: float;
        duration: float;
        easing: easing_function;
      }
      
      and typing_pattern = {
        base_cpm: int;  (* characters per minute *)
        variance: float;
        pause_probability: float;
      }
      
      and easing_function = Linear | EaseIn | EaseOut | EaseInOut
      
      (* 生成真实的鼠标移动路径 *)
      let generate_mouse_path start_pos end_pos =
        let distance = sqrt ((fst end_pos -. fst start_pos) ** 2.0 +.
                           (snd end_pos -. snd start_pos) ** 2.0) in
        let duration = 0.5 +. Random.float 0.5 in  (* 0.5-1秒 *)
        let num_points = int_of_float (distance /. 5.0) |> max 10 in
        
        (* 使用贝塞尔曲线生成平滑路径 *)
        let control1 = (
          fst start_pos +. Random.float 100.0 -. 50.0,
          snd start_pos +. Random.float 100.0 -. 50.0
        ) in
        let control2 = (
          fst end_pos +. Random.float 100.0 -. 50.0,
          snd end_pos +. Random.float 100.0 -. 50.0
        ) in
        
        let points = List.init num_points (fun i ->
          let t = float i /. float (num_points - 1) in
          bezier_point start_pos control1 control2 end_pos t
        ) in
        
        { points; duration }
      
      (* 模拟人类阅读行为 *)
      let simulate_reading page content =
        let content_length = String.length content in
        let words = content_length / 5 in  (* 平均单词长度 *)
        let reading_speed = 200 + Random.int 100 in  (* 200-300 WPM *)
        let reading_time = float words /. float reading_speed *. 60.0 in
        
        (* 添加随机的滚动行为 *)
        let%lwt () = Lwt_unix.sleep (0.5 +. Random.float 1.0) in
        
        let rec scroll_randomly remaining_time =
          if remaining_time > 0.0 then
            let pause = 1.0 +. Random.float 3.0 in
            let%lwt () = smooth_scroll page 
              (100 + Random.int 300) ~duration:0.5 in
            let%lwt () = Lwt_unix.sleep pause in
            scroll_randomly (remaining_time -. pause -. 0.5)
          else
            Lwt.return_unit
        in
        
        scroll_randomly reading_time
      
      (* 模拟表单填写 *)
      let simulate_typing page selector text =
        let%lwt () = Page.click page selector in
        
        (* 逐字符输入，模拟真实打字速度 *)
        let chars = String.to_list text in
        Lwt_list.iter_s (fun char ->
          let base_delay = 60.0 /. 250.0 in  (* 250 CPM *)
          let variance = base_delay *. 0.3 in
          let delay = base_delay +. (Random.float variance -. variance /. 2.0) in
          
          (* 偶尔停顿 *)
          let%lwt () = 
            if Random.float 1.0 < 0.1 then
              Lwt_unix.sleep (0.5 +. Random.float 1.0)
            else
              Lwt.return_unit
          in
          
          let%lwt () = Page.type_char page (String.make 1 char) in
          Lwt_unix.sleep delay
        ) chars
    end
    
    (* 验证码处理 *)
    module CaptchaHandling = struct
      type solver_service =
        | TwoCaptcha of { api_key: string }
        | AntiCaptcha of { api_key: string }
        | DeathByCaptcha of { username: string; password: string }
        | ManualSolver of { webhook_url: string }
      
      type solve_result = 
        | Solved of string
        | Failed of string
        | NeedRetry
      
      let solve_captcha service captcha_type page =
        match captcha_type with
        | ImageCaptcha ->
            (* 截图并发送到解决服务 *)
            let%lwt screenshot = Page.screenshot page 
              ~selector:".captcha-image" in
            send_to_solver service screenshot
        
        | ReCaptchaV2 ->
            (* 获取 sitekey *)
            let%lwt sitekey = Page.evaluate page
              ~script:"document.querySelector('.g-recaptcha').getAttribute('data-sitekey')" in
            solve_recaptcha_v2 service sitekey (Page.url page)
        
        | ReCaptchaV3 ->
            (* V3 需要更复杂的处理 *)
            solve_recaptcha_v3 service page
        
        | _ -> Lwt.return (Failed "Unsupported captcha type")
      
      (* ReCaptcha V2 自动解决 *)
      let auto_solve_recaptcha_v2 page solver_service =
        (* 等待 ReCaptcha 加载 *)
        let%lwt () = Page.wait_for_selector page ".g-recaptcha" in
        
        (* 获取必要信息 *)
        let%lwt sitekey = Page.evaluate page {|
          document.querySelector('.g-recaptcha').dataset.sitekey
        |} in
        
        (* 请求解决服务 *)
        let%lwt token = match solver_service with
          | TwoCaptcha { api_key } ->
              TwoCaptcha.solve_recaptcha ~api_key ~sitekey 
                ~pageurl:(Page.url page)
          | _ -> failwith "Solver not implemented"
        in
        
        (* 注入解决方案 *)
        let%lwt () = Page.evaluate page 
          (Printf.sprintf {|
            document.getElementById('g-recaptcha-response').innerHTML = '%s';
            if (typeof ___grecaptcha_cfg !== 'undefined') {
              Object.entries(___grecaptcha_cfg.clients).forEach(([key, client]) => {
                if (client.callback) {
                  client.callback('%s');
                }
              });
            }
          |} token token)
        in
        
        Lwt.return (Solved token)
    end
    
    (* 代理管理 *)
    module ProxyManagement = struct
      type proxy_type = HTTP | HTTPS | SOCKS5
      
      type proxy = {
        host: string;
        port: int;
        proxy_type: proxy_type;
        credentials: (string * string) option;
        country: string option;
        city: string option;
        mutable last_used: float;
        mutable success_count: int;
        mutable failure_count: int;
        mutable banned: bool;
        mutable response_times: float list;
      }
      
      type proxy_pool = {
        proxies: proxy list ref;
        rotation_strategy: rotation_strategy;
        health_check_interval: float;
        ban_threshold: float;  (* failure rate *)
        mutable current_index: int;
      }
      
      and rotation_strategy =
        | Sequential
        | Random
        | LeastUsed
        | GeoTargeted of string  (* country code *)
        | PerformanceBased
      
      let select_proxy pool ~target_domain =
        let available = List.filter (fun p -> not p.banned) !(pool.proxies) in
        
        match pool.rotation_strategy with
        | Sequential ->
            let proxy = List.nth available 
              (pool.current_index mod List.length available) in
            pool.current_index <- pool.current_index + 1;
            proxy
        
        | Random ->
            List.nth available (Random.int (List.length available))
        
        | LeastUsed ->
            List.sort (fun p1 p2 -> 
              compare p1.last_used p2.last_used) available |> List.hd
        
        | GeoTargeted country ->
            available |> List.filter (fun p -> 
              p.country = Some country) |> List.hd
        
        | PerformanceBased ->
            (* 选择平均响应时间最短的代理 *)
            available |> List.map (fun p ->
              let avg_time = 
                if p.response_times = [] then Float.max_float
                else 
                  List.fold_left (+.) 0.0 p.response_times /.
                  float (List.length p.response_times) in
              (p, avg_time)
            ) |> List.sort (fun (_, t1) (_, t2) -> compare t1 t2)
            |> List.hd |> fst
      
      (* 代理健康检查 *)
      let health_check_proxy proxy test_url =
        try%lwt
          let start_time = Unix.time () in
          let%lwt response = Http.get test_url ~proxy in
          let response_time = Unix.time () -. start_time in
          
          proxy.response_times <- response_time :: 
            (List.take 100 proxy.response_times);
          proxy.success_count <- proxy.success_count + 1;
          
          Lwt.return true
        with _ ->
          proxy.failure_count <- proxy.failure_count + 1;
          
          (* 检查是否应该禁用 *)
          let failure_rate = 
            float proxy.failure_count /. 
            float (proxy.success_count + proxy.failure_count) in
          
          if failure_rate > pool.ban_threshold then
            proxy.banned <- true;
          
          Lwt.return false
    end
  end
  
  (* 检测与适应 *)
  module Detection = struct
    type scraping_detection = {
      url: string;
      detected_mechanisms: anti_scraping_mechanism list;
      confidence: float;
      suggested_strategy: evasion_strategy;
    }
    
    and evasion_strategy = {
      use_browser_automation: bool;
      enable_proxy_rotation: bool;
      request_delay: float;
      custom_headers: (string * string) list;
      solve_challenges: bool;
    }
    
    (* 检测反爬虫机制 *)
    let detect_anti_scraping response =
      let mechanisms = ref [] in
      
      (* 检测速率限制 *)
      if List.mem_assoc "X-RateLimit-Limit" response.headers then
        mechanisms := RateLimiting {
          requests_per_second = extract_rate_limit response.headers;
          burst = extract_burst_limit response.headers;
        } :: !mechanisms;
      
      (* 检测 Cloudflare *)
      if List.mem_assoc "CF-RAY" response.headers ||
         String.contains response.body "Checking your browser" then
        mechanisms := CloudflareProtection :: !mechanisms;
      
      (* 检测验证码 *)
      if String.contains response.body "g-recaptcha" ||
         String.contains response.body "h-captcha" then
        mechanisms := CaptchaChallenge {
          captcha_type = detect_captcha_type response.body
        } :: !mechanisms;
      
      (* 检测 JavaScript 挑战 *)
      if String.contains response.body "document.cookie" &&
         String.contains response.body "setTimeout" then
        mechanisms := JavaScriptChallenge {
          script_hash = hash_js_challenge response.body
        } :: !mechanisms;
      
      !mechanisms
    
    (* 自适应策略生成 *)
    let generate_evasion_strategy mechanisms =
      let strategy = {
        use_browser_automation = false;
        enable_proxy_rotation = false;
        request_delay = 1.0;
        custom_headers = [];
        solve_challenges = false;
      } in
      
      List.fold_left (fun strat mech ->
        match mech with
        | RateLimiting { requests_per_second; _ } ->
            { strat with request_delay = 1.0 /. requests_per_second *. 1.5 }
        
        | CloudflareProtection | JavaScriptChallenge _ ->
            { strat with 
              use_browser_automation = true;
              solve_challenges = true }
        
        | IPBlocking _ ->
            { strat with enable_proxy_rotation = true }
        
        | CaptchaChallenge _ ->
            { strat with 
              use_browser_automation = true;
              solve_challenges = true }
        
        | UserAgentBlocking _ ->
            { strat with 
              custom_headers = 
                HeaderSpoofing.generate_headers 
                  (List.hd HeaderSpoofing.real_browser_profiles) "" None }
        
        | _ -> strat
      ) strategy mechanisms
  end
end
```

## 本章小结

本章深入探讨了分布式爬虫系统的设计与实现。我们从并发模型开始，介绍了 OCaml Effects 系统在爬虫中的应用，展示了如何构建高效的异步爬取框架。URL 调度器的设计展示了如何在礼貌性和效率之间取得平衡，包括优先级管理、分布式队列架构和故障恢复机制。

去重策略部分详细讨论了从 URL 规范化到内容指纹的多层次去重方案，特别是布隆过滤器、HyperLogLog 等概率数据结构在大规模场景下的应用。渲染服务的设计解决了现代 JavaScript 驱动网页的爬取挑战，通过渲染池管理和资源优化策略实现了高效的浏览器自动化。

最后，我们探讨了爬虫系统的扩展性设计，包括水平扩展架构、负载均衡策略、全面的监控调试工具，以及应对反爬虫机制的策略。这些设计模式和技术选择为构建生产级爬虫系统提供了坚实的基础。

关键要点：
- 使用 Effects 系统实现高效的并发爬取，避免回调地狱
- 分布式 URL 队列需要考虑一致性、容错和负载均衡
- 去重不仅是技术问题，更是空间和准确性的权衡
- JavaScript 渲染需要在资源消耗和爬取完整性之间平衡
- 反爬虫对抗是持续的攻防游戏，需要灵活的策略组合

## 练习题

### 基础题

1. **并发控制设计**
   设计一个自适应的并发控制器，能够根据目标服务器的响应时间和错误率动态调整并发数。
   
   **Hint**: 考虑使用 AIMD (Additive Increase Multiplicative Decrease) 算法，类似 TCP 拥塞控制。

   <details>
   <summary>参考答案</summary>
   
   实现一个基于响应时间和错误率的反馈控制系统。当响应时间低于阈值且错误率低时，线性增加并发数；当检测到错误或响应时间过高时，将并发数减半。使用滑动窗口统计最近 N 个请求的指标，避免单次异常导致的剧烈波动。
   </details>

2. **URL 优先级计算**
   给定页面的 PageRank、更新频率、域名权威度等因素，设计一个综合的优先级计算公式。
   
   **Hint**: 不同因素可能有不同的尺度，需要归一化处理。

   <details>
   <summary>参考答案</summary>
   
   Priority = α × normalize(PageRank) + β × normalize(UpdateFrequency) + γ × normalize(DomainAuthority) + δ × (1/depth)
   
   其中 α + β + γ + δ = 1，normalize 函数将值映射到 [0,1] 区间。可以使用对数变换处理 PageRank 的长尾分布。
   </details>

3. **布隆过滤器参数选择**
   预计爬取 10 亿个 URL，要求误判率不超过 0.1%，计算最优的位数组大小和哈希函数数量。
   
   **Hint**: 使用布隆过滤器的参数计算公式。

   <details>
   <summary>参考答案</summary>
   
   m = -n × ln(p) / (ln(2)²) ≈ 14.4 × 10⁹ bits ≈ 1.7 GB
   k = (m/n) × ln(2) ≈ 10 个哈希函数
   
   其中 n = 10⁹，p = 0.001
   </details>

4. **渲染池大小估算**
   假设每个浏览器实例占用 300MB 内存，平均渲染时间 3 秒，QPS 需求 100，计算所需的渲染池大小。
   
   **Hint**: 使用 Little's Law。

   <details>
   <summary>参考答案</summary>
   
   根据 Little's Law: L = λ × W
   其中 λ = 100 QPS，W = 3 秒
   因此需要 L = 300 个并发渲染
   
   考虑到故障和负载波动，建议配置 350-400 个实例
   总内存需求：400 × 300MB = 120GB
   </details>

### 挑战题

5. **分布式去重系统设计**
   设计一个支持 100 台爬虫节点的分布式去重系统，要求查询延迟 < 10ms，支持每秒 100 万次查询。
   
   **Hint**: 考虑使用分片、缓存、批量查询等优化技术。

   <details>
   <summary>参考答案</summary>
   
   架构设计：
   - 使用一致性哈希将 URL 空间分片到多个去重节点
   - 每个爬虫节点维护本地 LRU 缓存，减少远程查询
   - 批量查询接口，减少网络往返
   - 使用布隆过滤器作为第一层过滤，减少精确查询
   - 去重节点使用 Redis 集群存储已见 URL
   - 异步复制实现最终一致性，容忍短时间重复
   </details>

6. **反爬虫检测算法**
   设计一个算法，通过分析 HTTP 响应特征自动检测网站使用的反爬虫机制。
   
   **Hint**: 考虑响应头、响应体特征、JavaScript 代码模式等。

   <details>
   <summary>参考答案</summary>
   
   检测流程：
   1. 分析响应头：X-RateLimit-*, CF-RAY, Server 等
   2. 扫描响应体关键词：captcha, challenge, robot, blocked
   3. 识别 JavaScript 挑战：setTimeout + document.cookie 模式
   4. 检测蜜罐链接：display:none 或 visibility:hidden 的链接
   5. 分析重定向链：多次 302/303 重定向
   6. 使用机器学习分类器，基于上述特征判断反爬类型
   </details>

7. **爬虫调度优化问题**
   给定 N 个域名，每个域名有不同的爬取延迟要求和优先级，M 个爬虫节点，设计一个调度算法最大化吞吐量。
   
   **Hint**: 这是一个带约束的优化问题，考虑贪心或动态规划方法。

   <details>
   <summary>参考答案</summary>
   
   使用优先队列 + 域名分组的方法：
   1. 将 URL 按域名分组，每个域名维护独立队列
   2. 使用最小堆维护每个域名的下次可爬取时间
   3. 爬虫节点空闲时，从堆顶选择可爬取的最高优先级域名
   4. 考虑地理位置，将域名分配给网络延迟最小的节点
   5. 动态调整：根据实际爬取时间更新延迟估计
   6. 使用 ε-贪心策略平衡开发和探索
   </details>

8. **增量爬取策略**
   设计一个增量爬取系统，能够智能地决定何时重新爬取已知页面，最小化带宽使用同时保证内容新鲜度。
   
   **Hint**: 考虑页面变化模式、重要性、历史更新频率等因素。

   <details>
   <summary>参考答案</summary>
   
   多层次更新策略：
   1. 基于历史的更新频率估计：使用泊松过程建模页面更新
   2. 自适应调度：根据实际变化调整爬取间隔
   3. 重要性加权：PageRank × 更新频率 × 用户兴趣
   4. 使用 HTTP 条件请求：If-Modified-Since, ETag
   5. 内容变化检测：结构化 diff，忽略时间戳等噪声
   6. 分层爬取：高优先级页面使用 RSS/Sitemap 推送
   7. 预测模型：基于时间序列预测最佳爬取时机
   </details>

## 常见陷阱与错误 (Gotchas)

1. **内存泄漏问题**
   - 错误：长时间运行的爬虫进程内存持续增长
   - 原因：DOM 对象、事件监听器未正确清理
   - 解决：定期重启浏览器实例，使用弱引用管理缓存

2. **死锁风险**
   - 错误：多个爬虫节点相互等待，系统停止响应
   - 原因：分布式锁获取顺序不一致
   - 解决：使用超时机制，按固定顺序获取锁

3. **时区问题**
   - 错误：爬取的时间戳不一致或错误
   - 原因：服务器和爬虫时区不同
   - 解决：统一使用 UTC 时间，在展示层转换

4. **字符编码陷阱**
   - 错误：非 UTF-8 页面出现乱码
   - 原因：未正确检测和转换字符编码
   - 解决：使用 chardet 检测编码，优先信任 HTTP 头

5. **Cookie 处理错误**
   - 错误：登录状态丢失，被识别为爬虫
   - 原因：Cookie 域名、路径、过期时间处理不当
   - 解决：使用专门的 Cookie Jar，注意 SameSite 属性

6. **重定向循环**
   - 错误：爬虫陷入无限重定向
   - 原因：未记录重定向历史
   - 解决：限制重定向次数，检测循环

7. **JavaScript 执行超时**
   - 错误：等待 JavaScript 加载超时
   - 原因：网络慢或 JS 死循环
   - 解决：设置合理超时，使用多种等待条件

8. **代理失效**
   - 错误：使用失效代理导致大量失败
   - 原因：未及时检测和剔除坏代理
   - 解决：定期健康检查，使用代理池

## 最佳实践检查清单

### 架构设计
- [ ] 爬虫节点无状态设计，支持随时扩缩容
- [ ] 使用消息队列解耦组件，提高系统弹性
- [ ] 实现优雅关闭，确保任务不丢失
- [ ] 设计容错机制，单点故障不影响整体

### 性能优化
- [ ] 使用连接池复用 TCP 连接
- [ ] 启用 HTTP/2 多路复用
- [ ] 实现本地 DNS 缓存
- [ ] 批量处理减少网络往返

### 礼貌爬取
- [ ] 遵守 robots.txt 规则
- [ ] 实现域名级别的速率限制
- [ ] 设置合理的 User-Agent
- [ ] 支持 crawl-delay 指令

### 数据质量
- [ ] URL 规范化确保去重准确性
- [ ] 内容完整性校验（Content-Length）
- [ ] 处理各种字符编码
- [ ] 保存原始响应用于调试

### 监控告警
- [ ] 实时监控爬取速率和成功率
- [ ] 设置错误率告警阈值
- [ ] 记录详细日志便于问题排查
- [ ] 定期生成爬取报告

### 安全合规
- [ ] 不爬取个人隐私信息
- [ ] 遵守网站服务条款
- [ ] 实现访问控制和审计日志
- [ ] 定期安全扫描和渗透测试

### 扩展性考虑
- [ ] 模块化设计便于功能扩展
- [ ] 预留接口支持新的内容类型
- [ ] 配置外部化，避免硬编码
- [ ] 支持灰度发布和 A/B 测试
