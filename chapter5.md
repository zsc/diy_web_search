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

### 5.4 去重策略与数据结构选择
- URL 去重的挑战
- 布隆过滤器的应用
- 分布式去重架构
- 内容指纹与相似度检测

### 5.5 渲染服务的接口设计
- JavaScript 渲染需求
- 无头浏览器集成
- 渲染池管理
- 资源优化策略

### 5.6 爬虫系统的扩展性设计
- 水平扩展架构
- 负载均衡策略
- 监控与调试
- 反爬虫应对策略
