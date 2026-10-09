# SSD MoE 专家权重重组

本功能是 `ssd_off` 分支上的可选路径，并保留现有 M0 观测能力。当前仅支持 Linux 下的单文件 Qwen3 MoE GGUF。原始模型仍然必需，用于加载元数据、非专家权重以及保持不变的计算图。

## 重组文件位置

本次已经生成并验证的重组文件为：

- 宿主机：`/home/cuichengshang/moe_ssd/test/ssd-pack-20261008/weights/Qwen3-30B-A3B-BF16.experts.gguf`
- `moe-llama-dev` 容器：`/ssd/test/ssd-pack-20261008/weights/Qwen3-30B-A3B-BF16.experts.gguf`
- 文件大小：58,007,564,288 字节，约 54.02 GiB。

宿主机和容器中的路径指向同一个文件。原始模型位于容器内 `/ssd/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-BF16.gguf`。重组文件没有覆盖或修改原始模型。

## 文件布局与复制优化

`scripts/pack-ssd-moe.py` 生成一个独立 GGUF 文件，每个 `(layer, expert)` 对应一条 I8 字节记录。I8 只表示外层容器类型，原始权重的类型和字节内容均不改变。每条记录的布局如下：

```text
[gate 字节][gate 保护区][up 字节][up 保护区][down 字节][down 保护区][对齐填充]
```

每条记录从 4096 字节边界开始，记录长度也是 4096 的整数倍。保护区保存同一投影中下一个专家开头的字节，用于匹配现有 CUDA 复制路径额外读取的 `min(expert_size, 512)` 字节。每层最后一个专家没有下一个专家，其保护区填零，并且不会提交到计算缓冲区。

gate、up、down 三个投影共享一条整专家记录的缓存分配、读取 future、LRU 身份和复制生命周期。三个投影返回这块分配中的不同视图。在记录没有被淘汰时，一个冷专家只需读取一次整记录，而原路径通常需要分别读取三个投影。

当文件偏移、目标地址和长度均满足对齐要求时，direct I/O 会直接写入缓存分配，避免旧路径先写 bounce buffer、再将 payload 复制到缓存的过程。缓存统计会把对齐分配产生的额外空间计入容量。

默认情况下，现有 CUDA 临时输入张量仍然保留。每个投影仍需从 host cache 复制到 CUDA 计算张量，cache hit 也不例外。GB10 上可以额外启用后文的地址表零复制路径，由 MMF kernel 直接读取应用级 cache；重组本身不等于零复制。

## 生成和校验重组文件

使用已经运行的 `moe-llama-dev` 容器以及其中已有的 NumPy、PyYAML 环境。不要创建新容器，也不要覆盖已有重组文件。在容器内从 `/workspace/llama.cpp` 执行：

```bash
export PYTHONPATH=/ssd/test/m0/python${PYTHONPATH:+:$PYTHONPATH}
python3 scripts/pack-ssd-moe.py \
  /ssd/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-BF16.gguf \
  /ssd/test/ssd-pack-20261008/weights/Qwen3-30B-A3B-BF16.experts.gguf
python3 scripts/pack-ssd-moe.py \
  /ssd/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-BF16.gguf \
  /ssd/test/ssd-pack-20261008/weights/Qwen3-30B-A3B-BF16.experts.gguf \
  --verify
```

工具每次只在内存中处理一个专家，不修改原文件。它先在目标目录写临时文件并执行 `fsync`，完成后再以禁止覆盖的方式发布。正常异常会删除临时文件；如果进程被强制终止，目录中可能留下 `.partial-*` 文件。BF16 模型需要约 54 GiB 的额外空间以及少量记录对齐和元数据空间。重组文件不进行压缩。

文件头绑定原模型的文件大小、纳秒修改时间、张量形状、张量类型和源文件偏移。加载器会拒绝过期元数据、不支持的版本、缺少或多出的记录以及被截断的文件。这些元数据不是密码学内容标识，因此应保持源文件和重组文件不可变，并使用 `--verify` 做完整字节校验。完整校验同时检查 payload、保护区、对齐填充，并输出所有记录 payload 的 SHA-256。

本次 30B 文件已完成 6,144 条记录逐字节校验，payload SHA-256 为：

```text
c1711bcb40c7ccfae71d31b129b8c5b76782ee31aa69de599dcc8118485df913
```

## 启用重组权重

通过 `LLAMA_SSD_MOE_PACK` 指定重组文件，同时正常启用 SSD MoE offload：

```bash
LLAMA_SSD_MOE_PACK=/ssd/test/ssd-pack-20261008/weights/Qwen3-30B-A3B-BF16.experts.gguf \
GGML_MOE_ASYNC_DEMAND=1 \
./build/bin/llama-cli \
  -m /ssd/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-BF16.gguf \
  -ngl all --ssd-moe-offload --ssd-moe-cache-mib 8192 \
  --ssd-moe-cache-workers 4 -p Hello
```

必须使用与待验证源代码对应的可执行文件。`llama-bench` 也可使用同一环境变量。实验清单需要单独记录这个环境变量，因为现有 benchmark JSON 不会自动保存它。

未设置 `LLAMA_SSD_MOE_PACK` 时，程序继续使用原始 GGUF 权重路径。打包路径与 `GGML_MOE_MMAP_ONLY=1` 互斥。重组文件读取失败会停止当前计算，不会使用错误布局继续复制，也不会静默退回 mmap。

如果当前选中专家的整记录无法同时放入缓存，程序会明确要求增大 cache 或减小 batch。容量判断按整记录计算，并包含对齐分配开销。prefill 一次涉及更多专家，最容易触发这一限制。

## 启用 GB10 地址表零复制

在支持 pageable host page table 的 CUDA 设备上，可以让非量化 MMF kernel 直接读取重组 cache 中的专家权重：

```bash
LLAMA_SSD_MOE_PACK=/ssd/test/ssd-pack-20261008/weights/Qwen3-30B-A3B-BF16.experts.gguf \
GGML_MOE_ASYNC_DEMAND=1 \
GGML_MOE_ZERO_COPY=1 \
./build/bin/llama-cli \
  -m /ssd/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-BF16.gguf \
  -ngl all --ssd-moe-offload --ssd-moe-cache-mib 8192 \
  --ssd-moe-cache-workers 4 -p Hello
```

scheduler 为每个投影提交 128 项专家地址表，所选项指向重组记录的 gate、up 或 down slice。设备端只复制约 1 KiB 地址表，MMF kernel 根据 expert ID 直接选择源地址。cache 条目会保持占用，直到对应 CUDA 消费完成，防止异步执行期间被 LRU 淘汰并复用。

该优化仅在重组模式、CUDA 设备支持 pageable host page table、算子为 `MUL_MAT_ID` 且能够走 MMF 时生效。任何条件不满足都会回到原 payload copy 路径。它没有把 cache 改成 pinned 内存，也没有删除非专家张量或地址表的正常复制。环境变量默认关闭。

## 统计指标与诊断

- `packed_reads`：成功加载的整专家记录数。
- `host_bounce_bytes`：从 direct-I/O bounce buffer 复制到目标缓存的 payload 字节数。文件系统支持 direct I/O 且重组记录正确对齐时应为零。
- `backend_copy`：从源缓存复制到计算张量的字节量，本次优化不会消除它。
- `backend_copy_saved`：地址表零复制跳过的专家 payload 字节量。
- `zero_copy_binds`：成功绑定专家地址表的投影次数。
- `zero_copy_control_bytes`：设备地址表的控制复制量。
- `zero_copy_fallbacks`：请求零复制但设备或 kernel 条件不满足的次数。
- `resident`：当前计入容量的 cache 分配量，包括对齐余量；一条共享记录只计算一次。
- `disk`：已经完成的显式读取请求长度，不代表 SSD NAND 的实际物理流量。

打包模式的 hit 统计按“投影访问是否找到了共享记录”计算。gate miss 后，up/down 命中同一记录属于正常现象，因此不能把打包模式的 hit rate 与原先逐投影 cache 的 hit rate 直接比较并宣称缓存复用提升。应同时查看读取请求数、读取字节量和端到端耗时。

原始 GGUF 路径中的文件偏移可能不对齐，因此仍可能使用 bounce copy。如果文件系统不支持 `O_DIRECT`，程序可以使用 buffered `pread` fallback，解释 I/O 结果时必须明确区分。设置 `GGML_MOE_VERIFY_WEIGHTS=1` 后，每次提交的投影视图都会与原始 GGUF 对比，包括保护字节。该模式会预热源文件页面，不得用于性能测量。

打包 trace 中三个投影共享分配。原 M0 审计器按逐投影分配设计，不能直接用其旧解析结果作为打包模式验收。必须额外检查共享 slot 生命周期，并在正常非诊断路径上进行输出比较。也不能把重组文件环境变量直接注入已有的四基线冻结实验，因为其中的 mmap 对照组会按设计拒绝打包模式。打包路径应使用独立实验目录。

## 功能测试

```bash
python3 scripts/test-ssd-moe-pack.py
```

当前 6 项测试覆盖 FP32/BF16 记录和保护区字节、拒绝覆盖已有文件、缺少投影、源文件时间戳变化、payload 损坏以及超过 4 GiB 后的校验偏移。CPU/CUDA 编译和真实 GPU 正常路径输出一致性属于单独检查。小型功能测试的 I/O 次数不能作为真实模型加速证据。

## 独立 GPU 验证

`scripts/validate-ssd-moe-pack.py` 使用 `llama-moe-m0` 在独立目录中验证重组路径。`prepare` 阶段会固定每个领域的第一条开发请求、完整 prompt、最多 16 个 continuation token、数值阈值、源文件摘要以及三轮交错性能测试顺序。运行前需要已有的完整校验重组文件，以及 `ROOT/build/bin/llama-moe-m0` 下的新 CUDA 构建。

在实验容器中执行：

```bash
export PYTHONPATH=/ssd/test/m0/python
python3 scripts/validate-ssd-moe-pack.py prepare --root ROOT
python3 scripts/validate-ssd-moe-pack.py run --root ROOT --label correctness-resident
python3 scripts/validate-ssd-moe-pack.py run --root ROOT --label correctness-unpacked
python3 scripts/validate-ssd-moe-pack.py run --root ROOT --label correctness-packed
python3 scripts/validate-ssd-moe-pack.py compare --root ROOT
python3 scripts/validate-ssd-moe-pack.py run --root ROOT --label diagnostic-packed
python3 scripts/validate-ssd-moe-pack.py audit-trace --root ROOT
python3 scripts/validate-ssd-moe-pack.py performance --root ROOT
python3 scripts/validate-ssd-moe-pack.py report --root ROOT
```

当前输入来自已有的 BF16 冻结开发集，具体路径由验证驱动记录。正常路径会将每一步的完整词表 logits 和 greedy token 与新的 resident 运行结果比较。诊断路径分别检查原始权重字节、CUDA copy、共享记录身份、消费者依赖、淘汰生命周期和 cache 容量。性能路径关闭诊断与 logits 写盘。

M0 配置项 `ssd_stats` 默认为 false；打开后，quiet 模式仍会保留每个 context 最终的 cache counters。三轮短负载结果用于实现验收和初步动机实验，不能替代毕业论文的完整评测。

本次真实 GPU 结果、限制和机器可读产物见[中文验证报告](/home/cuichengshang/moe_offload/docs/ssd-pack-validation-20261008.zh.md)。

GB10 地址表零复制的正确性、生命周期和三轮性能结果统一记录在[SSD MoE 持续优化总记录](/home/cuichengshang/moe_offload/docs/ssd-moe-optimization-log.zh.md)。冻结结果位于 `/ssd/test/ssd-zerocopy-20261009/`：四类正式输入的完整 logits 与 resident 基线最大绝对误差为 0，生命周期审计通过；相对 P1 重组路径，平均 TTFT 降低 11.29%，平均 TPOT 降低 17.36%，三轮累计专家 payload `backend_copy` 从 1,746,749.49 MiB 降为 0，同时新增 32.0625 MiB 地址表控制复制。
