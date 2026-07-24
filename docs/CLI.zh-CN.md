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

强度严格大于阈值的掩膜像素会被选中。`--method` 可选 `telea` 或 `ns`。使用 `--json`
可以获得一条机器可读的单图结果。

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
- 批量执行固定使用一个 worker。
- 不支持批次恢复和自动重试。
- PNG 可以保留受支持的 Alpha 数据；JPEG 输出有损且不能保留 Alpha。
