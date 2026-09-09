# CONTEXT 交接

> 磁盘代码是真相。通关草案 → `FSM_DESIGN.txt`；为什么 → `DECISIONS.md`；组集 → `TRAIN_ALIGN.md`。
> 本文件**只写当天增量**，禁止复述另三份。新旧想法：用新的，事后告知。组集 vs TRAIN_ALIGN 已确认步骤打架时先问。
> 通关以 `FSM_DESIGN.txt` 为准。

生成：2026-09-09 11:29 (UTC+8)

## 本会话已做
- CAST 左右脸；前进/走近超时重入
- 卡住恢复改为 json `stuck_recover` 序列 + 面板 `stuck_recover_s`（默认 ESC 点按，再右左 2s、上下 1s 倍数）

## 下一对话先做
- **边打边调**；提取规则已变 → 先「重新提取本图」/重提 `skill_features`
- 手感 + 叠图；不要另开架构盖 txt
- 组集 fill/相对仍未确认

## 路径
- 工程：`d:\Desktop\T\test`
- 推荐段：`recordings/深渊：最终调律者/20260902_074831_SolarWarden`
- 特征 / 键位：`skill_features/深渊：最终调律者/SolarWarden.json`；`skill_binds/SolarWarden.json`
- YOLO：`D:/Atrain/runs/mix_a/weights/best.pt`
- 回放：`python fsm_replay.py`　GUI：`python gui_module.py`
