# 为 Watermark Removal Lab 贡献

[English](CONTRIBUTING.md) | [简体中文](CONTRIBUTING.zh-CN.md)

感谢您帮助改进 Watermark Removal Lab。所有贡献都必须保持项目的学习属性、可替换流水线架构、可复现评测能力以及“仅处理已授权内容”的边界。

## 贡献前须知

- 只能使用您拥有、创建或已获授权修改的媒体素材。
- 不得贡献以移除付费预览或版权保护水印为目标的平台专用逻辑。
- 不得提交模型权重、数据集、生成媒体、凭据或本地环境文件。
- 大型架构变更应在实现前讨论。
- 请阅读[编码规范](docs/CODING_STANDARDS.zh-CN.md)、[工程路线图](docs/ROADMAP.zh-CN.md)、相关里程碑规范和已接受的 [CLI 优先架构决策](docs/adr/0001-cli-first-core.zh-CN.md)。
- 修改目录、清单、队列、恢复、重试或多项处理行为前，请阅读[批量处理设计](docs/BATCH_PROCESSING.zh-CN.md)。

## 变更流程

1. 定义一个可独立评审的目标。
2. 创建短期分支，使用 `feature/`、`fix/`、`docs/`、`refactor/` 或 `chore/` 等清晰前缀。
3. 保持变更聚焦，并根据风险补充测试或文档。
4. 运行受影响组件所规定的仓库检查。
5. 创建 Pull Request，并完成所有适用的检查项。

不要在功能或修复提交中混入无关重构、格式化、生成文件或依赖更新。

## 提交信息

每个提交必须使用：

```text
TYPE:English description
```

具体描述必须使用英语且首字母大写，冒号前后均不添加空格。

| 类型 | 用途 |
|---|---|
| `FEAT` | 新增行为或能力 |
| `FIX` | 修复缺陷 |
| `ENHANCE` | 性能或资源占用优化 |
| `DOC` | 文档相关 |
| `CHORE` | 维护、测试或文件整理 |
| `BUILD` | 构建、打包或依赖逻辑 |
| `SCRIPT` | 开发或运维脚本 |
| `REFACTOR` | 不应改变行为的内部重构 |
| `STYLE` | 仅格式调整 |
| `OTHER` | 无法合理归入其他类型的变更 |

示例：

```text
FEAT:Add OpenCV image inpainting
FIX:Preserve alpha channel during export
DOC:Document binary mask conventions
```

提交应尽量少量多次，优先拆成多个小而完整的提交，避免一次提交过多无关内容。

## 代码与测试要求

- 遵循 [docs/CODING_STANDARDS.zh-CN.md](docs/CODING_STANDARDS.zh-CN.md)。
- 保持适配器依赖无界面核心层的单向依赖关系。
- 修复缺陷时添加回归测试，新增能力时添加聚焦测试。
- 测试素材必须是合成、自制或具有明确许可的素材。
- 在有损编码之前，最终细化掩膜之外的像素必须保持完全一致。
- 报告质量、延迟、RAM 和 VRAM 数据时，应提供足以复现的上下文。

## 模型与第三方材料

添加模型、权重、数据集、复制实现或媒体素材之前，必须：

1. 核实来源和准确许可证；
2. 确认许可证适用于具体产物，而不仅是其代码仓库；
3. 在 [MODEL_LICENSES.md](MODEL_LICENSES.md) 或 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 中登记；
4. 保留许可证要求的版权、归属、NOTICE 和修改说明；
5. 记录下载方式和完整性哈希，不将大型产物提交到仓库。

项目采用 Apache-2.0，并不代表第三方权重或数据集会自动变成 Apache-2.0。

## 文档语言

英文文档使用基础文件名，简体中文译文使用 `.zh-CN.md` 后缀。不同语言必须分文件维护，在顶部添加语言切换入口；修改共同含义时应同步更新两个版本。

标识符、提交主题、代码注释、文档字符串、日志字段名和机器可读输出统一使用英语。

## 发布要求

每次发布必须：

- 创建带注释的 Git tag；
- 在发布说明中引用该 tag；
- 列出可下载资源及其 MD5；
- 附带适用的模型许可证和第三方声明；
- 说明支持的平台、运行 provider、已知限制，以及视频处理属于工程基线还是具备时序一致性的实现。
