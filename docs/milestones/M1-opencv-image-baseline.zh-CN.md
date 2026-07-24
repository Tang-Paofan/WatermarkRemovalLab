# M1 规范：OpenCV 图片基线

[English](M1-opencv-image-baseline.md) | [简体中文](M1-opencv-image-baseline.zh-CN.md)

本文档是 M1 的实现与验收契约，定义无模型、离线的图片去除核心、可脚本调用的 CLI，以及 B1 顺序批处理基础。

## 1. 目标

M1 必须：

- 使用边界框或二值掩膜移除单张图片中由用户选择的可见覆盖物；
- 支持 OpenCV Telea 和 Navier-Stokes 补全；
- 在有损编码前保持 Alpha 和最终细化掩膜外像素不变；
- 让 CLI 和批量编排器复用同一个单图应用服务；
- 在不复制算法逻辑的前提下处理确定性的顺序图片批次；
- 在没有模型权重、GPU 运行时和网络访问的环境中运行；
- 提供聚焦测试与可操作的失败信息。

## 2. 非目标

M1 不包括：

- 自动水印检测；
- SAM/SAM 2、LaMa、ONNX Runtime 或其他模型推理；
- 交互式点击或图形化掩膜编辑；
- 桌面端、Web 或 API 适配器；
- 视频解码或逐帧处理；
- 并发 worker、GPU 调度、断点恢复或自动重试；
- 时序一致性；
- 自动保留所有 EXIF、ICC 或厂商专用元数据；
- 针对付费预览水印的平台专用逻辑。

## 3. 架构切片

```text
CLI 或 B1 批量编排器
            ↓
单图去除服务
            ↓
图片 IO + 掩膜准备 + OpenCV 补全器
            ↓
仅掩膜回贴 + 原子输出
```

批量编排器只创建单图请求并收集结果，不得读取图片数组、直接调用 OpenCV 或实现另一套去除行为。

## 4. 标准数据契约

- 解码图片：`uint8`、`H × W × 3`、RGB；
- 可选 Alpha：独立 `uint8`、`H × W`；
- 内部二值掩膜：`bool`、`H × W`；
- 序列化掩膜：0 表示背景，255 表示选中；
- 内部边界框：`(x_min, y_min, x_max, y_max)`，右下坐标不包含；
- CLI 边界框：`x,y,width,height`，转换为内部边界框；
- OpenCV 适配器：显式转换 RGB 到 BGR 以及 BGR 到 RGB。

服务不得修改输入数组。结果包含新的 RGB 数组和原始 Alpha 数组。

## 5. 单图 CLI

命令形式：

```text
wrl image remove INPUT OUTPUT (--box X,Y,W,H | --mask MASK)
    [--method telea|ns]
    [--radius PIXELS]
    [--dilate PIXELS]
    [--mask-threshold 0..255]
    [--save-mask PATH]
    [--overwrite error|skip|replace]
    [--json]
```

示例：

```powershell
wrl image remove input.png output.png --box 100,50,240,80 --method telea

wrl image remove input.png output.png --mask watermark-mask.png `
  --method ns --radius 3 --dilate 2 --save-mask final-mask.png
```

必需行为：

- `--box` 和 `--mask` 互斥且必须提供一个；
- `--method` 默认为 `telea`；
- `--radius` 是正数像素半径，默认值为 3；
- `--dilate` 是非负像素半径，默认值为 0；
- `--mask-threshold` 默认为 127；
- `--overwrite` 默认为 `error`；
- `--json` 向标准输出写入一个机器可读结果对象；
- 启用 JSON 输出时，人类进度与诊断信息写入标准错误。

## 6. 边界框与掩膜行为

### 边界框输入

- `x`、`y`、`width` 和 `height` 均为整数；
- 宽度和高度必须为正；
- 完整边界框必须位于已解码且方向归一化的图片内；
- 接触图片边缘是合法的；
- 越界边界框直接失败，不进行静默裁剪。

### 外部掩膜输入

- 掩膜宽高必须与已解码图片一致；
- 如果掩膜包含 Alpha，则使用 Alpha 作为掩膜强度；
- 否则转换为灰度强度；
- 强度大于 `--mask-threshold` 的像素被选中；
- 空掩膜是带警告的成功无操作；
- 全图掩膜合法，但可能产生较差的视觉质量；
- 不猜测或自动旋转掩膜方向。

### 膨胀

`--dilate N` 使用 `(2N + 1) × (2N + 1)` 椭圆核扩张二值掩膜。膨胀后的掩膜是最终细化掩膜，并定义算法唯一允许修改的像素。

## 7. 图片 IO 与输出安全

### 必需格式支持

- M1 必须能够解码和编码 PNG、JPEG `.jpg` 与 JPEG `.jpeg`。
- 目录发现以不区分大小写的方式匹配 `.png`、`.jpg` 和 `.jpeg`。
- PNG 是必需的无损及 Alpha 保留输出格式。
- JPEG 输出仅支持不包含 Alpha 的图片，并始终标记为有损。
- BMP、WebP、TIFF 等其他格式不属于 M1 退出门槛，除非具有明确文档和集成测试。

- 在边界框校验之前完成图片方向解码。
- 输出方向进行归一化；M1 不保证完整保留全部元数据。
- 输出格式支持 Alpha 时，必须逐字节保留原始 Alpha。
- 输入包含 Alpha 而输出格式不能保留 Alpha 时必须拒绝，除非未来接口允许用户显式接受 Alpha 丢失。
- 永远不得覆盖输入路径。
- 先写入目标目录中的临时文件，验证完成后再原子替换最终路径。
- 失败后在安全情况下清理临时文件。
- 不留下不完整的最终输出。

覆盖策略：

| 策略 | 行为 |
|---|---|
| `error` | 输出已存在时，在处理前失败 |
| `skip` | 返回成功跳过结果，不修改输出 |
| `replace` | 成功处理后原子替换现有输出 |

JPEG 等有损编码器可能在编码阶段改变掩膜外像素。JSON 结果必须标记有损输出，正确性测试比较内存像素或无损 PNG 输出。

## 8. OpenCV 补全契约

- Telea 对应 OpenCV Telea 算法。
- `ns` 对应 OpenCV Navier-Stokes 算法。
- OpenCV 适配器接收 RGB 与布尔掩膜，并在内部执行所需转换。
- 适配器可以计算全图候选结果，但服务只在最终掩膜为真的位置回贴候选像素。
- 空掩膜绕过 OpenCV 并返回未改变的副本。
- 后端失败转换为领域错误，同时保留原始异常。

## 9. B1 顺序批量 CLI

B1 有两种输入方式，最终都转换为同一种批次项契约。

### 目录模式

```text
wrl batch image --input-dir INPUT_DIR --output-dir OUTPUT_DIR
    (--box X,Y,W,H | --mask-dir MASK_DIR)
    [--recursive]
    [--method telea|ns]
    [--radius PIXELS]
    [--dilate PIXELS]
    [--output-format preserve|png]
    [--overwrite error|skip|replace]
    [--results PATH]
    [--fail-fast]
```

规则：

- 按确定性的相对路径顺序发现受支持图片；
- 默认不递归；
- 使用 `--recursive` 时保留输出中的相对目录结构；
- 共享边界框应用于每张图片，并逐项校验；
- `--mask-dir` 镜像相对路径，并把输入扩展名替换为 `.png`；
- 缺少配对掩膜属于逐项失败；
- `--output-format preserve` 保留各输入扩展名；
- `--output-format png` 把各输出扩展名替换为 `.png`；
- 处理前预检输入目录与输出路径。

### 清单模式

```text
wrl batch run MANIFEST.jsonl --output-dir OUTPUT_DIR
    [--results PATH]
    [--overwrite error|skip|replace]
    [--fail-fast]
```

版本化 JSON Lines 清单以一个批次记录开始，后面是逐项记录：

```json
{"record":"batch","schema_version":1,"media":"image","operation":"remove","defaults":{"method":"telea","radius":3,"dilate":1}}
{"record":"item","id":"sample-a","input":"inputs/a.png","output":"a.png","box":[10,20,120,40]}
{"record":"item","id":"sample-b","input":"inputs/b.png","output":"b.png","mask":"masks/b.png","method":"ns"}
```

清单规则：

- 第一条记录必须包含值为 `image` 的 `media` 和值为 `remove` 的 `operation`；
- 每个项 ID 都是唯一非空字符串；
- 路径相对于清单所在目录；
- 输出路径相对于 `--output-dir`；
- 每个项必须且只能提供 `box` 或 `mask` 中的一个；
- 项字段覆盖批次默认值；
- 未知 schema 版本、缺少必需字段以及不支持的媒体/操作值在处理前失败；
- 重复输出路径和输入输出同路径在预检阶段失败；
- 拒绝越过声明根目录的路径遍历；
- 项执行顺序与清单顺序一致。

完整批量契约见 [../BATCH_PROCESSING.zh-CN.md](../BATCH_PROCESSING.zh-CN.md)。

## 10. B1 执行行为

- B1 只使用一个 worker。
- 预检为本次运行生成一个稳定的 `RUN_ID`。
- 省略 `--results` 时，结果默认写入 `<OUTPUT_DIR>/.wrl-batch/<RUN_ID>/results.jsonl`；`run.json` 与 `summary.json` 位于同一个状态目录。
- 自定义 `--results` 只覆盖 JSON Lines 结果位置，并且不得与输入、掩膜、媒体输出或批次元数据文件别名。
- 默认在某项失败后继续。
- `--fail-fast` 在首个失败项后停止调度。
- 后续项失败时，已完成输出仍然有效。
- 每项结果状态为 `succeeded`、`skipped`、`failed` 或 `cancelled`。
- 每项结束后向 JSON Lines 结果文件追加一条记录。
- CLI 汇总发现、成功、跳过、失败和取消数量。
- 快速失败把所有剩余未调度项标记为 `cancelled`，原因为 `fail_fast`。
- Ctrl+C 在当前原子步骤后停止，把所有剩余未调度项标记为 `cancelled`，原因为 `user_cancelled`，并以 130 退出。
- B1 不恢复中断任务；B2 引入经过校验的恢复机制。

## 11. 结果字段

单图和批次项结果共享机器可读结构，至少包含：

- schema 版本；
- 适用时的项 ID；
- 输入与输出路径；
- 状态；
- 方法与归一化后的选项；
- 图片宽度和高度；
- 最终掩膜选中像素数；
- 毫秒耗时；
- 输出编码是否有损；
- 警告；
- 失败时的稳定错误代码与消息。

标准结果输出不得包含图片像素、凭据、堆栈跟踪或不必要的完整私有路径。

## 12. 退出码

| 代码 | 含义 |
|---|---|
| `0` | 单项成功/跳过，或全部批次项成功/跳过 |
| `2` | CLI 参数、清单、边界框、掩膜或预检配置无效 |
| `3` | 处理完成但一个或多个项失败 |
| `4` | 编排、IO 或输出提交发生致命失败 |
| `130` | 用户取消 |

批次项失败必须同时反映在退出码与结果文件中。

## 13. 测试矩阵

### 单元测试

- CLI 边界框解析和左闭右开转换；
- 零值、负值、接触边缘和越界边界框；
- 掩膜加载、阈值、空/全图掩膜和尺寸不匹配；
- 0 和正半径的椭圆膨胀；
- RGB/BGR 往返；
- 仅掩膜回贴；
- 输入不可变与 Alpha 保留；
- 覆盖策略决策；
- 确定性目录与清单排序；
- 清单 schema、重复 ID、冲突和路径越界拒绝。

### 集成测试

- 在合成 RGB PNG 上运行 Telea 和 NS；
- RGBA PNG 的 Alpha 逐字节一致；
- 灰度输入归一化为 RGB；
- 空掩膜无操作；
- 无损输出中最终掩膜外像素一致；
- JPEG 输出被标记为有损；
- 模拟写入失败后的原子替换与清理；
- 单图 CLI 成功和无效输入退出码；
- 使用共享边界框的目录批次；
- 使用配对掩膜的目录批次；
- 使用逐项覆盖的清单批次；
- 部分失败、快速失败、跳过和替换行为；
- Ctrl+C 或取消边界不会留下不完整最终输出。

所有默认测试仅使用 CPU、离线运行，并使用合成测试素材。

## 14. 验收清单

只有满足以下条件，M1 才算完成：

- [x] 单图边界框与掩膜流程通过同一个应用服务运行；
- [x] Telea 和 NS 均可选择且有测试；
- [x] 有损编码前最终掩膜外像素保持不变；
- [x] 支持的输出能够保留 Alpha；
- [x] 输入永不被原地修改；
- [x] 输出写入原子化且覆盖行为明确；
- [x] 目录和清单 B1 批次复用单图服务；
- [x] 批次顺序与结果确定；
- [x] 部分失败可见且不会使已完成项失效；
- [x] 所有必需单元与集成测试在 CPU 离线环境通过；
- [x] CLI 帮助和示例记录限制与负责任使用边界；
- [x] 没有引入模型、GUI、视频或自动检测依赖。

验收证据记录在 [M1/B1 验收报告](../acceptance/M1-B1-acceptance.zh-CN.md)中。

## 15. 建议实现切片

1. Python 包与测试工具骨架；
2. 图片、Alpha、边界框和掩膜类型；
3. 图片 IO 与掩膜工具；
4. OpenCV 补全器与仅掩膜回贴；
5. 单图应用服务与 CLI；
6. B1 批次契约、预检和结果；
7. 目录与清单批量适配器；
8. 集成测试与 M1 文档。

每个切片都应可以独立评审，并遵循仓库提交信息规范。
