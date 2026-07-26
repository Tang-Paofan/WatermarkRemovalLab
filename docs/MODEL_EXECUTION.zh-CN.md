# 模型执行与复现环境

[English](MODEL_EXECUTION.md) | [简体中文](MODEL_EXECUTION.zh-CN.md)

本文说明 Watermark Removal Lab 如何分离普通本地开发与真实模型执行，以及其他贡献者如何在 AutoDL 或等价兼容主机上复现 M2 LaMa 验证。

本文补充 [M2 规范](milestones/M2-lama-image-inpainting.zh-CN.md) 与
[ADR 0002](adr/0002-m2-lama-onnx-runtime.zh-CN.md)，不定义托管推理服务。

## 1. 为什么拆分环境

当前维护者工作站不作为真实模型执行环境。它继续承担无界面核心、OpenCV 基线、CLI 行为、裁剪变换、假运行时测试和全部默认离线检查。

固定 LaMa 产物与 CUDA provider 在独立 Linux 算力主机上验证。AutoDL 是项目的首个参考主机，因为它可以提供 NVIDIA GPU 和持久工作区存储。AutoDL 不是产品依赖：任何满足同一软件、模型、provider 和完整性契约的主机都可以复现结果。

环境拆分不得削弱默认测试基线。没有 GPU 或模型文件的贡献者仍必须能够开发并验证非模型包。

## 2. 环境职责

| 环境 | 必须承担 | 不得成为以下工作的前提 |
|---|---|---|
| 本地工作站或普通 CI | 格式、Lint、类型、单元/集成测试、OpenCV、假模型会话、打包 | 模型下载、ONNX Runtime GPU、CUDA、网络访问 |
| AutoDL 参考实例 | 固定模型安装、CPU 模型冒烟、CUDA provider 冒烟、真实推理、RAM/VRAM 与耗时证据 | 源码唯一存放位置或永久产物存储 |
| 其他兼容 Linux GPU 主机 | 复现同一描述符、provider、测试与证据字段 | AutoDL 专用 API 或文件路径 |

实验运行期间，模型和已授权测试媒体位于算力主机本地。开发者直接在该主机调用普通项目 CLI。工作站向 AutoDL 上传图片的 API 不属于 M2；远程服务传输属于 M6。

## 3. 参考主机要求

Linux 主机需要：

- 验证 `CUDAExecutionProvider` 时具有 NVIDIA GPU；
- 驱动与仓库锁定的 ONNX Runtime GPU 版本兼容；
- 使用仓库支持的 Python 版本；
- 安装 Git 和仓库支持的 `uv` 版本；
- 具有足够空间容纳环境、208044816 字节模型、已授权 fixture 与输出；
- 只在用户明确安装时访问固定 Hugging Face 产物；
- 在 Git 工作区外提供可写缓存。

不能只按 GPU 名称选择环境。ONNX Runtime 需要兼容的 CUDA 与 cuDNN 库。另外，`nvidia-smi` 展示的是驱动支持的最高 CUDA 版本，不是环境实际安装的 CUDA 工具链。复现时必须同时记录驱动信息和 ONNX Runtime 实际注册的 provider。

生成验收证据时，只使用仓库锁定的官方发行包。不得替换为 nightly，也不得手工安装未记录的最新版本。

参考资料：

- [ONNX Runtime 安装矩阵](https://onnxruntime.ai/docs/install/)
- [ONNX Runtime CUDA provider 要求](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [AutoDL CUDA/cuDNN 说明](https://www.autodl.com/docs/cuda/)
- [AutoDL 基础镜像说明](https://www.autodl.com/docs/base_config/)

## 4. AutoDL 存储与会话规则

参考目录：

```text
/root/autodl-tmp/
├── wrl-models/       已验证模型缓存根目录
│   └── lama-onnx-fp32/
│       └── lama_fp32.onnx
├── wrl-work/         已授权或合成输入及临时输出
└── wrl-evidence/     日志与机器可读验收证据

<系统或选定项目路径>/
└── WatermarkRemovalLab/   Git 工作区；不提交权重或用户媒体
```

AutoDL 把 `/root/autodl-tmp` 记录为实例数据盘路径。平台也明确提示本地系统盘与数据盘没有冗余副本，实例释放后数据会丢失。模型可以从固定来源重新下载，但已评审证据和有价值的授权输入必须另行备份。

规则：

- 永远不得把 Git 凭据、SSH 私钥、访问令牌或服务密码放入仓库或证据目录；
- 不提交模型、输入媒体、生成输出、缓存或环境文件；
- 只使用合成、自制或明确获授权的媒体；
- 释放或重置实例前备份小型已评审证据；
- 不需要保留时删除敏感授权媒体；
- SSH 长任务使用 `tmux` 或 `screen`，并把日志保存到文件；
- 检查完成后关闭付费 GPU 实例。

参考资料：

- [AutoDL 实例数据保留](https://www.autodl.com/docs/instance_data/)
- [AutoDL 本地数据盘提示](https://www.autodl.com/docs/local_disk/)
- [AutoDL SSH 说明](https://www.autodl.com/docs/ssh/)

## 5. 初始化仓库

克隆同一仓库并检出正在评审的准确提交：

```bash
git clone https://github.com/Tang-Paofan/WatermarkRemovalLab.git
cd WatermarkRemovalLab
git checkout <REVIEWED_COMMIT>
git status --short
```

先安装默认开发环境并证明普通基线通过，再添加模型运行时：

```bash
uv sync
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv build --no-sources
```

完成依赖同步后，这些命令必须保持无模型、离线运行。

## 6. 安装可选运行时与模型

### 当前实现边界

仓库当前已包含固定的 `lama-onnx-fp32` 描述符、原子模型存储、互斥的 CPU/CUDA 运行时 extra 和惰性 Session owner。离线测试证明：可选运行时只会在模型完整性校验通过后导入；CPU 与 CUDA 共用同一张量契约；CUDA 不可用时绝不静默回退 CPU；图元数据会被校验；每个 owner 最多保留一个 Session。

`wrl model` 命令、真实推理调用、裁剪变换、图片流水线接入和真实模型 pytest marker 仍属于后续 M2 切片。只有这些切片形成最小推理闭环后，才开始 AutoDL 真实模型验收；当前不能把下方命令示例视为已经可用。其他贡献者可以使用以下命令复核已实现切片：

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv build --no-sources
```

模型缓存根目录按以下优先级选择：适配器传入的 `--cache-dir` 或 `Path`、`WRL_MODEL_CACHE` 环境变量、平台用户缓存。产物路径为 `<缓存根目录>/<模型 ID>/<文件名>`。检查操作只读，不会创建目录或启动下载。

M2 定义两个可选依赖组：

```bash
# 仅 CPU 的真实模型验证
uv sync --extra lama-onnx-cpu

# CUDA 验证；同一个环境不得同时安装两个运行时 extra
uv sync --extra lama-onnx-cuda
```

两个 extra 都固定 ONNX Runtime 1.26.0。GPU 包是官方 CUDA 12.8 / cuDNN 9 构建；ONNX Runtime 1.27 及以上版本已经把 PyPI 默认 GPU 构建切换到 CUDA 13，因此升级前必须重新审查兼容性并记录 AutoDL 环境。CPU 与 GPU 发行包暴露相同的 `onnxruntime` 导入包，所以项目把两个 extra 声明为冲突，并要求分别使用不同环境生成验收证据。

只使用当前检出版本中确实存在、并由 `uv` 与项目元数据记录的 extra。分阶段实现 M2 时，契约描述的 CLI 命令仍可能尚未提供；该版本的 `wrl --help` 才是权威来源。

把模型显式安装到数据盘缓存：

```bash
uv run wrl model install lama-onnx-fp32 \
  --accept-model-terms \
  --cache-dir /root/autodl-tmp/wrl-models

uv run wrl model status lama-onnx-fp32 \
  --cache-dir /root/autodl-tmp/wrl-models \
  --json
```

安装器必须从产物提交 URL 下载，校验 208044816 字节大小与预期 SHA-256，然后原子发布。不得把手工重命名的未验证下载文件放入缓存。

预期 SHA-256：

```text
1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6
```

处理命令永远不下载缺失模型。

## 7. 推理前记录环境

保存以下等价命令的输出：

```bash
git rev-parse HEAD
uv run python --version
uv --version
nvidia-smi

uv run python -c \
  "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

进行 CUDA 验收时，创建模型会话前必须在 ONNX Runtime 列表中看到 `CUDAExecutionProvider`。请求 CUDA 却只报告 `CPUExecutionProvider` 属于 CUDA 环境失败，不是成功回退。

证据必须区分：

- GPU 型号与驱动；
- Python 与 ONNX Runtime 版本；
- 请求的 provider；
- 已注册和实际 provider；
- 模型描述符 ID、字节大小与 SHA-256；
- 仓库提交和工作区是否干净。

提交的证据不得包含 AutoDL 实例 ID、IP 地址、SSH 命令、令牌、用户名或媒体私有绝对路径。

## 8. 运行真实产物验证

M2 预留两个显式 pytest marker：

```bash
# 固定真实模型；CPU 即可
uv run pytest -m model

# 固定真实模型以及兼容 CUDA provider 与 GPU
uv run pytest -m "model and gpu"
```

只运行当前 `pyproject.toml` 已声明的 marker。默认 `uv run pytest` 必须继续排除网络、模型与 GPU 要求。

真实模型验证顺序：

1. 验证模型状态与哈希；
2. 验证模型张量名称、dtype 与固定 512×512 形状；
3. 运行合成 CPU 冒烟测试；
4. 验证原始分辨率掩膜外像素与 Alpha；
5. 用 CUDA 运行同一合成请求；
6. 确认请求与实际 provider 记录；
7. 运行已授权的人工 CLI 案例；
8. 使用相同解码输入与掩膜比较 Telea、Navier-Stokes 和 LaMa；
9. 保存延迟、峰值 RAM、峰值 VRAM、裁剪变换、警告与失败。

CPU 与 CUDA 输出可能存在很小的后端数值差异。测试应对模型区域使用文档化容差，同时继续严格保持掩膜外回贴不变量。

## 9. 在算力主机运行 CLI

预期执行形式：

```bash
uv run wrl image remove INPUT.png OUTPUT.png \
  --mask MASK.png \
  --method lama \
  --provider cuda \
  --crop-padding 64 \
  --model-dir /root/autodl-tmp/wrl-models \
  --overwrite error \
  --json
```

三个媒体路径均属于算力主机。M2 不通过 HTTP API 传输它们。

对比时对同一解码输入与最终掩膜运行 `telea` 和 `ns`。检查精确像素不变量时使用 PNG，避免 JPEG 编码污染结果。

## 10. 证据记录

原始日志保存在 Git 外。经过评审的小型验收记录以后可以双语提交到 `docs/acceptance/`，并至少包含：

```json
{
  "schema_version": 1,
  "repository_commit": "<FULL_GIT_SHA>",
  "worktree_clean": true,
  "environment": {
    "kind": "autodl",
    "os": "<OS_VERSION>",
    "python": "<PYTHON_VERSION>",
    "gpu": "<GPU_MODEL>",
    "driver": "<DRIVER_VERSION>",
    "onnxruntime": "<ORT_VERSION>"
  },
  "provider": {
    "requested": "cuda",
    "registered": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "effective": ["CUDAExecutionProvider", "CPUExecutionProvider"]
  },
  "model": {
    "id": "lama-onnx-fp32",
    "size_bytes": 208044816,
    "sha256": "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6"
  },
  "checks": {
    "default_suite": "<PASS_OR_FAIL>",
    "model_cpu": "<PASS_OR_FAIL>",
    "model_cuda": "<PASS_OR_FAIL>"
  }
}
```

声称性能或质量结果时，还要增加基准样本数、预热策略、输入尺寸、掩膜类别、延迟分布、RAM/VRAM 峰值、掩膜外变化数与失败项。

## 11. 可移植复现

在 AutoDL 之外复现：

1. 保持同一 Git 提交和锁文件；
2. 只安装一个已审查 ONNX Runtime 可选依赖组；
3. 使用相同模型产物 SHA-256；
4. 保持裁剪与 CLI 配置一致；
5. 运行相同默认、模型和 provider 专用测试；
6. 记录相同证据字段；
7. 说明操作系统、驱动、provider 或硬件差异。

可比较证据依赖准确身份与已记录配置，而不是 AutoDL 品牌。不能报告准确模型哈希或实际 provider 的环境不算复现了 M2 结果。
