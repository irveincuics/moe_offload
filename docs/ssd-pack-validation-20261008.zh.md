# SSD MoE 权重重组验证记录（2026-10-08 至 2026-10-09）

本次实现和验证均基于 `ssd_off` 分支的当前工作区。保留已有未提交的 M0 改动，未创建 commit，未推送远端。实验只使用已经运行的 `moe-llama-dev` 容器，结果写入独立目录 `/ssd/test/ssd-pack-20261008`，没有覆盖原 `/ssd/test/m0` 冻结实验。

## 实现范围

- 为单文件 Qwen3 MoE GGUF 提供无损专家重组工具。独立文件按 `(layer, expert)` 保存 gate、up、down 三个投影，记录起点和长度均按 4096 字节对齐。
- 保留 CUDA MMQ 读取需要的尾部保护字节。三个投影共享一次整专家记录读取、一个缓存分配、future、LRU 身份和复制生命周期。
- 对齐 direct I/O 直接写入缓存分配，避免原 GGUF 非对齐读取中的 host bounce-buffer payload memcpy。
- 加载时校验格式版本、源文件尺寸和纳秒时间戳、张量类型、形状、来源偏移、记录数量、对齐及文件总长度。读取失败和缓存容量不足均显式失败。
- CUDA 临时计算张量与三个投影的 host-to-device copy 保持不变。本实现没有声称实现 GPU zero-copy、固定 GPU 专家槽或消除所有同步。

使用方法见 [ssd-moe-pack.md](/home/cuichengshang/moe_offload/llama.cpp/docs/ssd-moe-pack.md)，打包工具为 [pack-ssd-moe.py](/home/cuichengshang/moe_offload/llama.cpp/scripts/pack-ssd-moe.py)，独立验证驱动为 [validate-ssd-moe-pack.py](/home/cuichengshang/moe_offload/llama.cpp/scripts/validate-ssd-moe-pack.py)。运行时通过 `LLAMA_SSD_MOE_PACK` 指定重组文件，并正常启用 SSD MoE offload。

## 实际模型与环境

- 模型：Qwen3-30B-A3B BF16，48 层、每层 128 个专家、Top-8。
- GPU：NVIDIA GB10；驱动 580.173.02；容器 CUDA 编译器 12.9.86；目标架构 121a。
- SSD cache：8192 MiB；4 个读取 worker；batch 32；CPU threads 4。
- 重组文件：58,007,564,288 字节，约 54.02 GiB；6,144 条整专家记录。
- 重组文件位置：宿主机为 `/home/cuichengshang/moe_ssd/test/ssd-pack-20261008/weights/Qwen3-30B-A3B-BF16.experts.gguf`，容器内为 `/ssd/test/ssd-pack-20261008/weights/Qwen3-30B-A3B-BF16.experts.gguf`。
- 输入：已有 BF16 开发集每个领域的第一条请求，共问答、代码、数学、摘要四条。使用完整冻结 prompt，回放最多 16 个 continuation token。
- 性能执行顺序固定为 `unpacked -> packed -> packed -> unpacked -> unpacked -> packed`，每种路径三轮。每次重新加载模型，每条请求新建 context 和 cache。

完整配置、输入和源代码 SHA-256 保存在 `/ssd/test/ssd-pack-20261008/validation-plan.json`。可执行文件及动态库摘要保存在 `binaries.sha256`。测试期间未清除 OS page cache，也未锁定 GPU 频率。

## 验证结果

| 检查 | 结果与范围 |
|---|---|
| Python 功能测试 | 6 项通过；覆盖 FP32/BF16 payload 与保护区、拒绝覆盖、不完整层、源时间戳变化、损坏 payload、超过 4 GiB 的校验偏移 |
| 30B 全量重组校验 | 6,144 条记录逐字节通过；payload SHA-256 为 `c1711bcb40c7ccfae71d31b129b8c5b76782ee31aa69de599dcc8118485df913` |
| CUDA 构建 | 独立 Release 构建的 `llama-moe-m0` 通过，目标架构 121a；没有复用或覆盖冻结 M0 二进制 |
| 正常路径数值一致性 | resident、普通 SSD、打包 SSD 使用相同输入；两种 SSD 路径相对 resident 的全部保存 logits 最大绝对误差均为 0，greedy token 完全一致 |
| 打包权重和复制检查 | 诊断请求的 15,204 次源权重检查、15,204 次 CUDA copy 检查全部通过 |
| 共享记录生命周期 | 4,357 次 reserve、读取和最终 evict 一一对应；5,068 个 `(eval, layer, expert)` 的 gate/up/down 使用相同 slot；720 个消费者均在复制完成后执行 |
| 缓存边界 | trace 峰值 8,188.10 MiB，小于 8,192 MiB 配额；没有读取失败或 mmap fallback |
| Direct I/O | 性能运行中打包路径 155,055 次读取全部为 packed direct read，`host_bounce_bytes=0` |
| 原有专项检查 | 整记录共享、对齐、命中、淘汰、同步/异步等待、普通 GGUF 兼容、容量不足、读取失败、ASan/UBSan 均已通过；LeakSanitizer 因 ptrace 环境不可用，未声称完成泄漏验证 |
| 工作区检查 | 当前分支为 `ssd_off`；显式使用 `.moe_offload.git` 的 `git diff --check` 通过 |

全量打包最初暴露了校验器的一个问题：`GGUFReader.data_offset` 在此路径中可表现为 NumPy `uint32`，累计偏移超过 4 GiB 后发生溢出，因此在 `blk.3.expert.71` 报告伪不一致。直接抽查该记录前后及文件末尾均与源权重一致。将起始偏移显式转为 Python `int` 并加入跨 4 GiB 回归测试后，6,144 条记录全量校验通过。

## 性能结果

以下数值为每条请求三次结果先取均值，再对四条请求等权平均。TPOT 来自回放阶段逐 token decode，不包含首 token；TTFT 包含该请求的完整分块 prefill。性能运行关闭 trace、权重验证和 logits 保存，只保留 context 销毁时的累计 cache counters。

| 路径 | 平均 TTFT | 平均 TPOT | 相对普通 SSD |
|---|---:|---:|---:|
| 普通 SSD async | 12,597.41 ms | 408.74 ms/token | 基线 |
| 打包 SSD async | 11,360.45 ms | 288.62 ms/token | TTFT 缩短 9.82%；TPOT 缩短 29.39% |

打包路径的解码速度为普通 SSD 路径的 1.416 倍。三轮等权 TPOT 分别为：普通 SSD 401.39、411.16、413.67 ms/token；打包路径 286.40、291.58、287.88 ms/token。四个领域各自的 TPOT 降幅为 28.37% 至 30.41%，方向一致。

三轮共 12 个请求的累计 cache counters 如下。显式读取请求的总字节数几乎相同，说明收益来自更少的 I/O 请求、共享缓存身份以及消除 host bounce copy，而不是少读了大量权重数据。

| 指标 | 普通 SSD | 打包 SSD | 变化 |
|---|---:|---:|---:|
| direct read 请求数 | 465,270 | 155,055 | -66.67% |
| cache eviction 数 | 432,558 | 144,147 | -66.67% |
| 显式读取请求字节 | 1,397,627.46 MiB | 1,396,100.67 MiB | -0.11% |
| worker 累计 I/O 时间 | 562,455.85 ms | 452,877.58 ms | -19.48% |
| host bounce payload copy | 1.331 TiB | 0 | 消除 |
| backend copy | 1,746,749.49 MiB | 1,746,749.49 MiB | 不变 |

这里的 hit rate 不能直接横向比较：普通路径以单投影为 cache key，打包路径的 gate miss 后，up/down 会命中同一整专家记录。读取次数和实际耗时比 hit rate 更适合解释本次改动。

## 结论与限制

本次真实模型验证支持以下结论：权重重组没有改变模型输出；整专家记录使 direct read 数量约降为三分之一，消除了普通路径的 bounce payload copy；在当前 GB10、8 GiB cache、四领域短回放负载下，平均 TPOT 降低 29.39%，TTFT 降低 9.82%。

这些结果属于实现验收和初步动机实验，不能替代毕业论文的完整性能评测。后续论文实验仍需扩大请求数和生成长度，加入 8/16 GiB cache、不同 batch/worker 数、冷/热 page cache、长上下文和并发负载，并报告置信区间。当前未实施 GPU zero-copy、固定 GPU cache 或高层专家预取，因此 backend copy 总量没有下降，这也给下一阶段创新留下了明确空间。

机器可读结果位于 `/home/cuichengshang/moe_ssd/test/ssd-pack-20261008/validation-report.json`；正常路径数值审计为 `correctness-audit.json`，共享 slot trace 审计为 `trace-audit.json`，完整重组校验为 `verify-fixed.log`。原失败日志 `verify.log` 被保留，用于说明并复现超过 4 GiB 时的校验器问题。
