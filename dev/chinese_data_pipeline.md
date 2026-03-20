# Chinese Data Pipeline Notes

## 变更目标
- SFT/DPO 预处理落盘结果统一为中文客服语境。
- 不改训练框架，不改 JSONL schema 字段名，只改值的归一化与质量过滤。
- 保留 rejected/quality report/dataset_info 等既有产物。

## 默认系统提示词
- 旧值（英文）: `You are a professional e-commerce customer support assistant.`
- 新值（中文）: `你是一名专业的中文电商客服助手。请基于用户问题和上下文，提供准确、礼貌、清晰、可执行的回复，避免编造信息。`
- 定义位置: `align/prompts.py`

## 类别本地化
- 单一映射入口: `align/data.py` 中 `CATEGORY_LABELS` + `CATEGORY_ALIASES`
- 最终落盘类别（SFT/DPO 一致）:
  - `退货退款`
  - `物流配送`
  - `商品咨询`
  - `订单修改`
  - `售后维修`
  - `投诉处理`
  - `通用客服`
- 兼容旧英文类别输入（如 `returns_refunds`、`after_sales`、`complaint_soothing`），自动映射到中文类别。

## 清洗与过滤规则
- 文本清洗（instruction/input/output/chosen/rejected）会去除：
  - `#E-s[...]`
  - `[数字x]` / `[姓名x]` / `[Name]` 等占位符模板
  - `<think>...</think>`
  - `&nbsp;`
- 最小质量约束：
  - 拒绝空/无实质内容回复
  - 拒绝“纯泛化收尾”回复（如“请问还有其他可以帮到您的吗”）
  - 拒绝低信息 chosen（DPO）

## 审计脚本
- 新增: `scripts/audit_sft_data.py`
- 输出：
  - `reports/sft_data_audit.md`
  - `reports/sft_bad_cases.jsonl`
- 统计项：
  - total samples
  - placeholder leak
  - short response
  - generic closing
  - possible category mismatch
  - language mixture
  - empty response
  - top repeated responses
  - category distribution
  - rejection reason distribution（若有 rejected 文件）

## 重新生成数据
1. SFT 预处理
```bash
python scripts/prepare_data.py --profile smoke
```
2. DPO 预处理
```bash
python scripts/prepare_pref.py --profile smoke
```
3. 审计
```bash
python scripts/audit_sft_data.py --input data/processed/sft_train.jsonl --rejected data/interim/sft_rejected.jsonl
```

## 兼容性说明
- SFT/DPO schema 字段名保持不变。
- LLaMA-Factory 依赖的 `dataset_info.json` 结构保持不变。
- DPO 不会因类别改为中文而失配（类别映射和推断逻辑已统一）。
