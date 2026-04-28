# ALIFE PLATFORM CRITICAL DIRECTORY MANDATES
# 絕對鐵律：目錄讀寫規範 (CRITICAL DIRECTORY MANDATES)

這份文件記錄了本專案最核心、不可妥協的目錄與日誌讀寫鐵律。任何 AI 代理 (包含 Gemini、Codex 等) 在啟動會話或執行任何文件操作前，**必須且強制**優先閱讀並嚴格遵守本文件的規定。

## 鐵律一：`research/asal_engine/` 是絕對的唯讀禁區 (STRICTLY READ-ONLY)

- **定義**：`/Users/lexchien/Documents/AI/ALife-Platform/research/asal_engine/` 及其底下所有的子目錄（包含 `history/`, `runs/`, `substrates/` 等）。
- **規範**：
  - 這個目錄存放的是過去 ASAL 研究的歷史心血、基準測試與過往的工作日誌。
  - 它僅供**參考 (Reference)** 與**讀取 (Read)**，用來理解演算法或作為後續平台化開發的對照組。
  - **絕對禁止 (NEVER)** 在此目錄內新增、修改、刪除任何檔案或資料夾。
  - **絕對禁止** 將新的工作日誌寫入 `research/asal_engine/history/log/`。

## 鐵律二：唯一合法的工作日誌寫入點 (THE ONLY VALID LOG PATH)

- **定義**：所有的開發紀錄、進度更新、技術筆記、除錯過程等當前工作日誌 (Active Work Logs)。
- **規範**：
  - **只能、且必須**寫入根目錄下的 **`/Users/lexchien/Documents/AI/ALife-Platform/log/`** 資料夾中。
  - 必須依照日期建立子資料夾存放（例如 `log/2026-04-28/`）。
  - **絕對禁止** 發明新的歸檔路徑或將新日誌與歷史研究日誌混淆。
  - 遵循 `log/READ_WORK_LOG.md`、`log/gemini_log.md`、`log/codex_log.md` 既定的命名與歸檔慣例。

## 鐵律三：新開發與整合的歸屬 (NEW DEVELOPMENT SCOPE)

- 所有為了 `ALife-Platform` 統一架構 (Phase 1 及其後) 所進行的新開發、架構重構或系統整合，都必須發生在平台層或新的對應目錄中：
  - 共用基礎：`core/`
  - 基礎模型：`foundation_models/`
  - 數位分身：`digital_clone/`
  - 生成式 AI：`genai/`
  - 應用入口：`apps/`
  - 共用配置：`configs/`
  - 規劃文件：`docs/PLAN/`

---
**嚴重警告 (SEVERE WARNING)**：
違反上述任何一條鐵律，等同於破壞專案的歷史完整性與管理規範，這是不可原諒的低級錯誤。AI 代理必須為其目錄操作負起完全的責任。任何越界的讀寫行為將導致使用者終止合作與資源供應。
