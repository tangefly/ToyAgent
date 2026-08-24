# ToyAgent 数据目录

GAIA_Text（文档增强 agent 基准）测试数据的本地副本，来源：
`/home/tanger/workspace/GAIA_Text/gaia_document_only.jsonl` 与其 `documents/` 附件树。

## 布局

| 路径 | 说明 |
| --- | --- |
| `task_1.jsonl` | GAIA 第 1 条任务（spreadsheet 库存，gold answer: `Time-Parking 2: Parallel Universe`） |
| `gaia_document_only.jsonl` | 全量 23 条任务（validation split，level 1×14 / 2×8 / 3×1） |
| `documents/<task_id>/<task_id>.txt` | 各任务附件（xlsx/pdf 的文本抽取版），与数据集内 `file_path` 相对路径一致 |

## 说明

- 数据集 `file_path` 字段已改写为相对 ToyAgent 根目录的路径（`data/documents/...`），
  `read_file` 工具从 ToyAgent 目录（cwd）解析相对路径；
- 23 条中有 13 条无附件（纯推理题），11 条带附件（共 11 个文档，最大 4KB）；
- 运行结果（输出，非输入数据）写到根目录 `gaia_results/`。
