# CONTEXT 交接

> 磁盘代码是真相。通关草案 → `FSM_DESIGN.txt`；为什么 → `DECISIONS.md`；组集 → `TRAIN_ALIGN.md`。
> 本文件**只写当天增量**，禁止复述另三份。新旧想法：用新的，事后告知。组集 vs TRAIN_ALIGN 已确认步骤打架时先问。
> DECISIONS §4 未落地，不准覆盖 txt。

生成：2026-09-07 23:05 (UTC+8)

## 本会话已做
- 默认过图 YOLO → **`mix_a`**（勿擅自切回 `solarwarden_b`，除非对照）
- txt：开打范围外走近再放；组 `gaps_ms`；效率不足 100% 不计范围；分布 **S_pos / S_size**；OCR 进图建段
- `blueprint.txt` 非过时内容并入 `DECISIONS.md`「长期意图」；blueprint 仅跳转（若仍在盘）
- 本文改为**薄交接**；全文归档见 `archive_notes/CONTEXT_HANDOFF_FULL_2026-09-07*.md`
- Cursor 规则：`.cursor/rules/handoff.mdc`（alwaysApply）

## 下一对话先做
- **边打边调**；提取规则已变 → 先「重新提取本图」/重提 `skill_features`
- 手感 + 叠图；不要另开架构，不要用 DECISIONS §4 盖 txt
- 组集 fill/相对仍未确认

## 路径
- 工程：`d:\Desktop\T\test`
- 推荐段：`recordings/深渊：最终调律者/20260902_074831_SolarWarden`
- 特征 / 键位：`skill_features/深渊：最终调律者/SolarWarden.json`；`skill_binds/SolarWarden.json`
- YOLO：`D:/Atrain/runs/mix_a/weights/best.pt`
- 回放：`python fsm_replay.py`　GUI：`python gui_module.py`
