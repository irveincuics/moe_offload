# SSD MoE 持续优化总记录

本文是 `ssd_off` 分支的唯一持续优化主记录。从 2026-10-09 起，已经完成的优化、失败方案、动机数据、实现位置、验证结果和后续优化都追加到本文。专题文档和机器可读结果仍然保留，但结论必须同步回本文，方便按时间复盘，避免把不同版本的结果混用。

状态定义：`已验证` 表示真实 30B BF16 模型的正常路径数值检查已经通过；`功能通过` 表示只完成构建或小型测试；`进行中` 表示尚不能作为论文结果；`否决` 表示本机数据不支持继续采用。

## 固定环境与产物

- 分支：`ssd_off`。工作区包含未提交的 M0 和后续优化，不自动 commit 或 push。
- 容器：复用已经运行的 `moe-llama-dev`，源码挂载点为 `/workspace/llama.cpp`。
- 设备：NVIDIA GB10，统一物理内存；驱动 580.173.02；容器 CUDA 12.9.86；编译目标 121a。
- 原模型：容器内 `/ssd/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-BF16.gguf`。
- 重组权重：容器内 `/ssd/test/ssd-pack-20261008/weights/Qwen3-30B-A3B-BF16.experts.gguf`；宿主机对应 `/home/cuichengshang/moe_ssd/test/ssd-pack-20261008/weights/Qwen3-30B-A3B-BF16.experts.gguf`。
- 原模型和重组文件保持不变。所有新构建、日志和实验结果使用 `/ssd/test/` 下的新目录，不能覆盖已经冻结的实验。

## 优化演进总览

| 阶段 | 状态 | 主要改变 | 仍存在的问题 |
|---|---|---|---|
| V1：mmap 按需专家 | 已有实现 | 专家权重不在模型加载时全部预读，router 选择后再复制 | 依赖 page fault；无应用级容量控制 |
| V2.0：有界 SSD cache | 已有实现 | direct I/O、LRU、容量和统计 | 每个投影单独读取；非对齐读取需要 bounce copy |
| V2.2：异步需求流水 | 已有实现 | 多 worker 并行读取，等待和 copy submission 流水化 | 不是跨层预测；仍对每个投影执行 backend copy |
| M0：路径与数值观测 | 功能通过 | trace、权重校验、CUDA copy/consumer 事件、正常路径 logits 比较 | 诊断模式同步开销大，不是性能结果 |
| P1：整专家权重重组 | 已验证 | gate/up/down 共用一次对齐读取和一个 cache entry | host cache 到 CUDA 临时张量的 payload copy 仍存在 |
| P2：GB10 地址表零复制 | 已验证 | MMF kernel 直接读取 SSD cache 中的所选专家；只更新地址表 | 仅适用于支持 pageable host page table 且命中 MMF 的路径 |

## V1 至 M0：基础 SSD 卸载与可观测性

V1 将 Qwen3 MoE routed expert 保留在 GGUF 文件中，router 给出 Top-K 后只复制选中范围。V2 引入受容量约束的应用级 cache、direct I/O、LRU 淘汰、并行读取 worker，以及同步需求和异步需求两种等待方式。M0 增加了独立实验驱动和 trace，能够检查读取、cache 生命周期、权重字节、CUDA copy、真实消费者位置和正常执行路径 logits。

这些阶段建立了后续优化所需的强基线，但没有实现预测预取、CPU 专家计算或零复制。GB10 使用统一物理内存，因此“CUDA buffer”和“host buffer”不能直接解释成两份独立物理内存；是否存在多余复制必须根据代码路径、设备能力和实测共同判断。

详细历史说明保留在 `llama.cpp/docs/ssd-moe-offload.md` 和 `llama.cpp/docs/ssd-moe-m0.md`。后续论文引用性能数字时，以本主记录及其关联的冻结实验为准。

## P1：整专家权重重组与 direct I/O 直写

### 动机

原始 GGUF 将 gate、up、down 分别存放。普通 SSD cache 按投影读取，非对齐 direct I/O 先读入对齐 bounce buffer，再 `memcpy` 到 cache entry。计算前，每个投影还需要复制到 CUDA 临时权重张量。

### 实现

`scripts/pack-ssd-moe.py` 将每个 `(layer, expert)` 重组为一条 4096 字节对齐记录：

```text
[gate][gate 保护区][up][up 保护区][down][down 保护区][对齐填充]
```

三个投影共享一次读取、一个分配、future、LRU 身份和复制生命周期。对齐 direct I/O 直接写入 cache entry，删除 `bounce buffer -> cache entry` 的 payload memcpy。原权重类型和字节不变。

### 真实模型验收

- 重组文件 58,007,564,288 字节，约 54.02 GiB；6,144 条记录逐字节通过。
- resident、普通 SSD、重组 SSD 的完整保存 logits 最大绝对误差为 0，greedy token 完全一致。
- 15,204 次源权重检查和 15,204 次 CUDA copy readback 检查通过。
- direct read 请求由 465,270 降至 155,055，减少 66.67%。
- 三轮 12 请求累计 `host_bounce_bytes` 由 1.331 TiB 降至 0。
- `backend_copy` 均为 1,746,749.49 MiB，没有下降。
- 平均 TPOT 从 408.74 降至 288.62 ms/token，降低 29.39%；平均 TTFT 从 12.60 降至 11.36 秒，降低 9.82%。

冻结结果位于 `/ssd/test/ssd-pack-20261008/`。详细实验协议和限制见[权重重组验证报告](/home/cuichengshang/moe_offload/docs/ssd-pack-validation-20261008.zh.md)。

### 阶段结论

P1 已经消除 host 侧 direct-I/O bounce payload copy，并将三次投影读取合并为一次整专家读取。它没有消除 `host cache -> CUDA 临时权重张量` 的 payload copy，因此 P2 必须以 `backend_copy` 是否实际下降为核心验收指标。

## P2：GB10 地址表零复制

### 2026-10-09 动机微基准

微基准固定 64 MiB copy 和 3 MiB BF16 专家矩阵，结果保存在 `/ssd/test/ssd-pack-20261008/moe-copy-bench-v3.txt`：

| 路径 | 本机结果 |
|---|---:|
| pageable `cudaMemcpyAsync` | 58.99 GiB/s |
| pinned `cudaMemcpyAsync` | 57.97 GiB/s |
| mapped-host kernel copy | 107.99 GiB/s |
| pageable-host kernel copy | 102.88 GiB/s |
| device BF16 GEMV | 0.0104 ms/expert |
| mapped-host BF16 GEMV | 0.0179 ms/expert |
| pageable-host BF16 GEMV | 0.0103 ms/expert |
| copy + device BF16 GEMV | 0.0721 ms/expert |

设备属性为 `can_map=1`、`pageable_access=1`、`pageable_host_pt=1`，pageable 指针由 GPU 通过 host page table 直接访问。单纯把 cache 改成 pinned 内存没有收益，因此该方案标记为否决。pageable 直接 GEMV 在这个受控形状上与 device GEMV 接近，并显著低于 copy 后计算，为 P2 原型提供了本机依据。这仍然只是微基准，不能替代真实 MMF kernel 和完整模型结果。

### 实现

P2 仅在以下条件全部满足时启用：设置 `GGML_MOE_ZERO_COPY=1`；使用已重组的专家文件；后端为支持 pageable host page table 的 CUDA 设备；当前 `MUL_MAT_ID` 能走非量化 MMF kernel。

scheduler 不再为每个所选专家调用 payload `cudaMemcpyAsync`，而是为当前投影生成 128 项专家地址表，其中所选项指向 SSD cache slice。CUDA backend 将约 1 KiB 地址表更新到设备，MMF kernel 按 expert ID 取对应源地址并直接读取权重。cache entry 的 `copy_backend` 生命周期保护被复用于零复制消费者，淘汰前必须等待对应 CUDA backend 完成。

新增统计：

- `backend_copy_saved`：本轮因零复制跳过的专家 payload 字节。
- `zero_copy_binds`：成功绑定地址表的投影次数。
- `zero_copy_control_bytes`：地址表控制数据复制量。
- `zero_copy_fallbacks`：请求零复制但设备或 kernel 条件不满足的次数。

主要改动位于 `ggml/src/ggml-backend.cpp`、`ggml/src/ggml-cuda/ggml-cuda.cu`、`ggml/src/ggml-cuda/common.cuh`、`ggml/src/ggml-cuda/mmf.cu` 和 `ggml/src/ggml-cuda/mmf.cuh`。该路径默认关闭，不影响原重组路径；设备能力、算子形状或 kernel 条件不满足时，保留原 payload copy 作为回退。

### 真实模型验收

冻结实验位于 `/ssd/test/ssd-zerocopy-20261009/`，独立构建位于其 `build/` 子目录。源码摘要、二进制摘要、输入、运行顺序、日志和机器可读报告都保存在该目录，未覆盖 P1 的冻结结果。

正确性与生命周期结果：

- 小请求 smoke 中 resident、packed、zerocopy 保存的 607,744 个 logits 逐元素一致，最大绝对误差为 0，greedy token 一致。
- 四类正式输入分别比较 packed 和 zerocopy，共 8 组、18,536,192 个 logits；全部有限且与同版本 resident 基线逐元素一致，最大绝对误差为 0，greedy token 全部一致。
- 深度审计检查 4,357 次整专家读取、15,204 次源权重校验、15,204 次零复制地址绑定和 720 个 CUDA 消费节点，生命周期审计通过。
- 深度审计期间发生 3,448 次 cache 淘汰，峰值 cache 分配为 8,585,845,875 字节；未发现 GPU 消费完成前的条目复用。
- 正确性、审计和性能运行中 `zero_copy_fallbacks=0`，`host_bounce_bytes=0`。

三轮性能实验采用交错顺序，每轮包含相同的 4 类请求、每个请求新建 context，关闭 trace、权重 readback 和 logits 写盘：

| 指标 | P1 重组路径 | P2 零复制路径 | 变化 |
|---|---:|---:|---:|
| 平均 TTFT | 11,451.76 ms | 10,159.04 ms | 降低 11.29% |
| 平均 TPOT | 295.94 ms/token | 244.57 ms/token | 降低 17.36% |
| 解码速度比 | 1.000× | 1.210× | 提升 21.01% |
| 每轮四请求端到端总时间 | 62,628.46 ms | 54,519.43 ms | 降低 12.95% |
| payload `backend_copy` | 1,746,749.49 MiB | 0 MiB | 降低 100% |
| 地址表控制复制 | 0 MiB | 32.0625 MiB | 新增控制开销 |
| 零复制回退 | 不适用 | 0 | 无回退 |

三轮平均 TPOT 分别为：P1 的 295.74、293.31、298.78 ms/token；P2 的 244.82、245.65、243.23 ms/token。P2 每轮跳过 568.60 GiB 专家 payload copy，仅提交 10.6875 MiB 地址表；三轮显式复制的控制数据相对被消除的 payload 为约 1/54,480。这里的“零复制”特指专家权重 payload，不表示整个推理过程没有激活、地址表或框架内部控制复制。

机器可读总报告为 `/ssd/test/ssd-zerocopy-20261009/validation-report.json`，trace 审计为同目录下的 `trace-audit.json`。本轮结果证明在 GB10 统一物理内存上，让 MMF 直接读取应用级 pageable cache 比先复制到 CUDA 临时权重张量更快；它不能直接外推到离散 GPU，也不能替代毕业论文所需的长文本、不同 cache 容量和多模型实验。

### 阶段结论

P1 消除了 direct-I/O bounce payload copy，P2 进一步消除了 `host cache -> CUDA 临时专家张量` 的 payload copy。当前专家数据路径只保留 SSD 到应用 cache 的必要读取，以及每次投影约 1 KiB 的专家地址表更新。P2 已达到本阶段功能、数值、生命周期和短负载性能验收，可作为后续预取、缓存策略和 CPU/GPU 协作实验的新强基线。

## 统一验收规则

每项后续优化都必须追加以下内容：日期、假设、改动文件和开关、冻结输入、强基线、数值阈值、读取与复制字节、TTFT、TPOT、失败和 fallback、适用条件、原始结果目录。性能运行不得打开 trace、权重 readback 或 logits 写盘；正确性运行不得复用旧版本 logits 代替同版本 resident 基线。

任何“减少复制”方案至少同时报告：payload copy 字节、控制 copy 字节、额外内存、kernel 时间和端到端 TPOT。若 copy 字节减少但 kernel 直接访问变慢，必须将净结果报告为负结果，不能只展示流量下降。

## 后续候选

1. 扩展 P2 到长 prompt、长生成、不同 cache 容量和不同并发度，分别确定 prefill 与 decode 的收益边界。
2. 在 P2 可用后重做跨层预取实验，避免把旧复制路径的开销误归因于预取策略。
3. 评估带容量约束的 GPU 可执行专家槽，与 P2 按相同总物理预算比较。
4. 在固定数据路径后研究专家预测、分阶段预约与安全取消。
5. 最后再评估 CPU/GPU 协作；CPU 方案必须与最佳 GPU-only 数据路径比较。

## 后续追加模板

```text
### YYYY-MM-DD：优化名称
状态：进行中 / 功能通过 / 已验证 / 否决
假设：
实现与开关：
冻结实验：
正确性：
读取与复制：
TTFT / TPOT：
失败与限制：
结论和下一步：
```
