# 开发指南

[English](DEVELOPMENT.md) | [简体中文](DEVELOPMENT.zh-CN.md)

本文档规定本地 Python 环境和仓库认可的验证命令。产品架构与行为仍以编码规范、已接受的
ADR 和里程碑规范为准。

## 环境要求

- [uv](https://docs.astral.sh/uv/) 0.11 或更高版本
- Git

`.python-version` 将 Python 3.11 设为默认开发解释器，项目支持 Python 3.11 至 3.13。
后续里程碑引入运行时依赖、可选模型依赖和平台专用依赖时，必须继续使用独立依赖组。

## 准备环境

在仓库根目录运行：

```powershell
uv sync
```

`uv sync` 根据 `pyproject.toml` 和 `uv.lock` 创建或更新 `.venv`。默认环境会以可编辑方式
安装项目包和开发工具；该命令不会下载模型权重或数据集。

## 运行必要检查

```powershell
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
uv build --no-sources
```

以上命令依次检查格式、代码规范、静态类型、带覆盖率的测试以及源码包与 wheel 构建。
`uv build --no-sources` 用于确认项目包不依赖未声明的本地工作区来源。

如需先应用仓库格式化规则，再重新运行检查：

```powershell
uv run ruff format .
```

默认测试必须保持仅使用 CPU、离线、确定性执行且不下载模型。后续里程碑需要 GPU 或模型
测试时，应增加带明确标记的可选测试套件。

## 模型执行环境

当前维护者工作站不作为真实模型执行环境，但这不会阻断 M2：

- 默认本地开发与 CI 保持无模型；
- AutoDL 是固定 LaMa CPU/CUDA 验证的初始参考主机；
- 其他兼容 Linux 主机可以复现同一证据；
- 项目 CLI 直接在算力主机运行，而不是提前调用远程 API。

环境拆分、AutoDL 存储规则、可选运行时组、模型完整性校验、pytest marker、CLI 形式和验收
证据字段见 [MODEL_EXECUTION.zh-CN.md](MODEL_EXECUTION.zh-CN.md)。其中的命令只有在当前检出
版本已经包含对应 M2 实现切片时才可使用。

## 源码布局

```text
src/watermark_removal_lab/  可安装的无界面 Python 包
tests/                      单元测试与集成测试
docs/                       工程契约与架构决策
```

展示层适配器必须依赖公开应用服务；无界面包不得依赖 CLI、桌面端、Web 或框架专用对象。
