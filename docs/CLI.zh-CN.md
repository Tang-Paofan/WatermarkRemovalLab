# CLI 使用指南

[English](CLI.md) | [简体中文](CLI.zh-CN.md)

Watermark Removal Lab 仅用于您拥有、创建或已获授权编辑的图片。CLI 要求用户提供边界框或
掩膜，不会自动检测水印。

## 运行 CLI

按照[开发指南](DEVELOPMENT.zh-CN.md)准备仓库环境，然后运行：

```powershell
uv run wrl --help
```

以下示例使用合成文件名。执行命令前请先创建输出目录。

## 安装已审查的 LaMa 模型

LaMa 是可选的实验性单图后端。先按照[开发指南](DEVELOPMENT.zh-CN.md)安装且只安装一个
ONNX Runtime extra，再显式安装固定模型产物：

```powershell
uv run wrl model install lama-onnx-fp32 `
    --accept-model-terms `
    --cache-dir C:\wrl-models

uv run wrl model status lama-onnx-fp32 `
    --cache-dir C:\wrl-models `
    --json
```

安装命令会在下载前展示固定来源、模型卡声明的许可证、Places 数据集限制提示、预期字节数
和 SHA-256。只有两项完整性校验都通过后才会发布文件。图片处理绝不会自动下载缺失模型。
仓库不捆绑该产物；当前项目审查只批准不捆绑、非商业研究集成，详见
[MODEL_LICENSES.md](../MODEL_LICENSES.md)。

## 从单张图片移除覆盖物

使用格式为 `X,Y,WIDTH,HEIGHT` 的半开区间边界框：

```powershell
uv run wrl image remove input.png output.png --box 620,420,160,60
```

也可以提供外部掩膜：

```powershell
uv run wrl image remove input.png output.png `
    --mask mask.png `
    --mask-threshold 127 `
    --method ns `
    --radius 3 `
    --dilate 1 `
    --save-mask final-mask.png
```

强度严格大于阈值的掩膜像素会被选中。`--method` 可选 `telea`、`ns` 或 `lama`；
`--radius` 只适用于 OpenCV。

使用显式本地缓存和 provider 运行已审查的 LaMa 产物：

```powershell
uv run wrl image remove input.png output.png `
    --mask mask.png `
    --method lama `
    --provider cpu `
    --crop-padding 64 `
    --model-dir C:\wrl-models
```

`--provider`、`--crop-padding` 和 `--model-dir` 只适用于 LaMa。传入与后端不兼容的选项会
直接失败，不会被静默忽略。使用 `--json` 可以获得包含后端身份、模型 SHA-256、provider
诊断、警告与 crop 变换的单图机器可读结果。

## 批量处理图片目录

先创建输出目录，再把同一个边界框应用于所有发现的图片：

```powershell
New-Item -ItemType Directory -Force output | Out-Null

uv run wrl batch image `
    --input-dir input `
    --output-dir output `
    --box 620,420,160,60 `
    --recursive `
    --method telea `
    --radius 3 `
    --dilate 1 `
    --output-format preserve
```

不使用 `--recursive` 时，只扫描输入目录的直属文件。支持的图片按确定性的相对路径顺序
处理。`--output-format png` 把所有输出扩展名改为 `.png`；`preserve` 保留输入扩展名。

B1 目录批次和 Manifest 批次仍只接受 `telea` 与 `ns`。LaMa 批量执行、provider 感知调度
与恢复属于 B2，不会因为单图命令接入而提前开放。

如需使用掩膜，请用 `--mask-dir` 替换 `--box`：

```powershell
uv run wrl batch image `
    --input-dir input `
    --output-dir output `
    --mask-dir masks `
    --recursive
```

掩膜目录镜像输入图片的相对路径，并把输入扩展名替换为 `.png`。例如，
`input/nested/photo.jpg` 对应 `masks/nested/photo.png`。缺少掩膜会记录为单项失败；
除非设置 `--fail-fast`，其他项目仍会继续。目录掩膜固定使用阈值 127；Manifest 可以在
批次默认值或单个项目中覆盖 `mask_threshold`。

## 批量处理 Manifest

版本 1 的 JSON Lines Manifest 以一条批次记录开头，后面跟随逐项记录：

```json
{"record":"batch","schema_version":1,"media":"image","operation":"remove","defaults":{"method":"telea","radius":3,"dilate":1}}
{"record":"item","id":"sample-a","input":"inputs/a.png","output":"a.png","box":[10,20,120,40]}
{"record":"item","id":"sample-b","input":"inputs/b.png","output":"b.png","mask":"masks/b.png","method":"ns"}
```

运行命令：

```powershell
New-Item -ItemType Directory -Force output | Out-Null

uv run wrl batch run batch.jsonl `
    --output-dir output `
    --overwrite error
```

Manifest 中的输入和掩膜路径相对于 Manifest 所在目录，输出路径相对于指定的输出目录。
每个项目必须且只能提供 `box` 或 `mask` 中的一个。

## 批次结果与取消

默认情况下，每次运行写入：

```text
OUTPUT_DIR/
└── .wrl-batch/
    └── RUN_ID/
        ├── run.json
        ├── results.jsonl
        └── summary.json
```

每个项目进入终态后都会刷新 `results.jsonl`。所有已发现项目进入终态后，
`summary.json` 才会原子发布。`--results PATH` 只修改 JSON Lines 结果位置；相对路径会
在输出目录下解析。

默认失败策略是在单项失败后继续。`--fail-fast` 会将所有剩余未调度项目取消，并把原因
记录为 `fail_fast`。

按 Ctrl+C 会在安全边界请求取消。当前原子图片操作可以完成，已完成输出和已刷盘记录
继续有效；剩余未调度项目会以 `cancelled` 状态记录，原因为 `user_cancelled`。

## 退出码

| 代码 | 含义 |
|---|---|
| `0` | 所有适用项目成功或被有意跳过 |
| `2` | 参数、Manifest、路径或预检配置无效 |
| `3` | 处理完成，但一个或多个项目失败 |
| `4` | 编排、状态、IO 或输出提交发生致命错误 |
| `130` | 用户取消 |

## 当前限制

- 定位方式为手动提供边界框或掩膜。
- Telea 和 Navier-Stokes 是无模型基线，在纹理、结构或语义复杂背景上可能产生明显伪影。
- LaMa 使用固定的 512 × 512 局部 crop；较大掩膜会被缩小，可能损失细节。
- LaMa 需要固定模型产物和一个兼容的可选 ONNX Runtime 包；provider 请求绝不静默回退。
- 批量执行固定使用一个 worker。
- 不支持批次恢复和自动重试。
- PNG 可以保留受支持的 Alpha 数据；JPEG 输出有损且不能保留 Alpha。
