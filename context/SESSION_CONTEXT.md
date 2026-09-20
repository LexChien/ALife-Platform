# SESSION_CONTEXT / 工作接續

更新／Updated: 2026-09-21（Asia/Taipei）。範圍／Scope: Linux 本機文件與既有證據整理／local documentation and saved-evidence audit.

## 先讀 / Start here

1. [STATUS](../docs/STATUS.md)：目前成果與限制／Current results and limits.
2. [TASKS](TASKS.md)：尚待完成項目／Open work.
3. [9/21 核對紀錄 / Audit](../log/2026-09-21/progress_evidence_audit.md)：更正原因／Correction rationale.
4. [文件導覽 / Docs](../docs/README.md)、[工作索引 / Worklogs](../docs/WORKLOG_INDEX.md).

## 最新可確認進度 / Supported progress

- ZH：9/20 保存 101 tests OK、HTTP 3/3、NCA/Lenia 4/4、接觸轉化敘事 3/3。本次未重跑。
  EN: September 20 records show 101 tests OK, HTTP 3/3, NCA/Lenia 4/4 and contact-conversion narrative 3/3. No reruns in this audit.
- ZH：目前 Clone E3 all-True report 來自 mock_gemma／runtime.mode=dummy；較早真 subprocess 經 heuristic fallback／strip 得到暗語回覆，但 drift 回答截斷、判準過寬。D2 保留未完成。
  EN: Current E3 all-True results are mock_gemma with runtime.mode=dummy. Earlier subprocess recall was extracted through heuristic fallback/strip, but drift output is truncated and its gate permissive. D2 stays open.
- ZH：Q4 SmolLM2 最小 LoRA 已保存；9/17 完整交付文檔只到前置測試，完整 tiny run lineage 仍是 training，不能推論現在有程序在執行。Gemma 特化、held-out test／merged／reload 缺完成證據。
  EN: Minimal SmolLM2 Q4 LoRA is saved. The September 17 delivery note stops at preflight; full tiny-run lineage remains training, which does not establish a currently running process. Gemma specialization and full test/export/reload evidence remain absent.
- ZH：Plan 25 有交付包，E3 品質聲明依本次核對限縮；Plan 24 是未完成的完整平台門檻。
  EN: Plan 25 has a delivery pack with qualified E3 claims; Plan 24 remains the incomplete full-platform scope.

## 版本與保存 / Versions and preservation

ZH：9/17 frozen source 在 runs/validation/20260917_review_fixes/frozen/source/，其 96 項修復測試與 9/20 的 101 項不是同一版本。先讀 [驗證文件](../docs/VALIDATION_AND_REPLAY.md)。tools/run_python.py 為本機環境啟動器；本次未重驗環境。

EN: The September 17 frozen source and its 96 repair tests differ from the September 20 101-test worktree. Read the validation guide first. tools/run_python.py is the local launcher; this audit did not revalidate the environment.

ZH：HEAD=3de9f9c；快取 origin/main=ad5371d，0 ahead／11 behind，未 fetch。大量既有未提交程式與產物保留。本次只改文件／索引，未 commit、push 或同步另一台主機。

EN: HEAD is 3de9f9c; cached origin/main is ad5371d, 0 ahead/11 behind without fetching. Preserve existing uncommitted code and artifacts. This task changes documents/indexes only; no commit, push or cross-machine sync occurred.

## 專案原則 / Principles

ZH：ASAL 保留於 research/asal_engine/，共用 runtime 於 core/。新日誌放 log/YYYY-MM-DD/；匯入歷史保留於 research/asal_engine/history/log/。修改 ASAL runtime／工作流程前搜尋舊紀錄並讀該子目錄 AGENTS.md。

EN: ASAL remains in research/asal_engine/ and shared runtime in core/. New logs go under log/YYYY-MM-DD/; imported history stays in research/asal_engine/history/log/. Search past logs and read the subsystem AGENTS.md before changing ASAL runtime or workflow.

```bash
python3 tools/query_worklog.py jetson --limit 10
python3 tools/query_worklog.py cuda openclip --limit 10
python3 tools/query_worklog.py mock_gemma --limit 10
```
