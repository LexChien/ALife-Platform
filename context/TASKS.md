# TASKS

## Current follow-up, 2026-09-21 / 目前接續待辦

ZH：本清單依 [證據核對](../log/2026-09-21/progress_evidence_audit.md) 更新；本次僅整理，未執行下列開發。已勾選的歷史交付不代表 Plan 24 全部驗收。

EN: Updated from the evidence audit; this documentation task did not execute the development below. Historical checkmarks do not imply full Plan 24 acceptance.

- [ ] D2 / E3：分離 mock 與真模型報告，驗證實際 runtime；以完整原始回答與獨立記憶／壓力案例重驗，避免關鍵字或 persona 已注入暗語造成假通過。 / Separate mock/real reports and verify runtime; revalidate complete raw answers and held-out cases, avoiding keyword-only or persona-injected recall passes.
- [ ] D1：擴大 NCA／Lenia／Boids seeds、固定預算比較與人工影像檢查；4/4 執行成功不等於敘事品質。 / Broaden matched-budget seeds and visual review; successful execution is not narrative quality.
- [ ] 補保存實驗的收尾：9/17 ASAL probe 的 MP4／人工檢視仍待完成；Clone quality_v2 真模型只保存 writer／restart_recall，renamed_identity 及後續案例缺完整輸出／總報告。 / Close saved-run gaps: ASAL probe video/manual review and Clone quality_v2 remaining sessions/final report. [來源 / Evidence](../log/2026-09-21/record_completeness_check.md)
- [ ] D3：補學習式影像產物、語音生成 metadata 與圖聲品質驗收。 / Save learned-image artifacts, speech metadata and media-quality evaluation.
- [ ] D4：完成 tiny LoRA held-out test、adapter／merged 匯出與重載驗證；保持 SmolLM2 與 Gemma lineage 分離。 / Complete held-out evaluation, exports and reload validation; keep SmolLM2 and Gemma lineage separate.
- [ ] D5：可重現環境、部署啟停與真實 live telemetry；服務 log／合成 heartbeat 不代替目前在線或推理證據。 / Reproducible deployment and actual telemetry; historical logs/synthetic heartbeat do not establish current uptime or inference.
- [ ] D6：待各範圍驗收後，從對應來源版本重跑完整回歸並更新總驗收包。 / Rerun full regression against the delivery version and refresh final acceptance after remaining gates close.
- [x] 整理 STATUS／導覽／接續／模型來源、Plan 24／25 狀態與工作索引。 / Reconcile status, navigation, handoff, lineage, plan status and worklog catalog.

## Repair closure, 2026-09-17 / 2026-09-17 修復收尾

- [x] OpenCLIP 全精度值域與非法輸入檢查／OpenCLIP dtype/range validation.
- [x] 敘事連續持續時間、碎片、質量流失、明確 accepted 與原因／Narrative duration, fragments, mass loss, explicit acceptance and reasons.
- [x] 固定 persona ID、metadata 權威重建、真實跨程序隔離／Stable persona IDs, authoritative metadata recovery, real cross-process isolation.
- [x] Python／subprocess 共用 prompt profile，extractor 可停用／Shared driver prompt profiles with extractor opt-out.
- [x] 局部 RNG、無損軌跡與來源／分數檢查／Local RNG, lossless trajectories, source and score checks.
- [x] 三種合法 schema、範例、格式評估、前置檢查／Three valid schemas, examples, format evaluation, and preflight.
- [x] 首輪 90 項通過；最終固定來源修復回歸 96 項通過、零跳過／Initial 90 passed; final frozen repair suite: 96 passed, zero skipped.
- [x] 新增服務 3 項測試：HTTP 整合 — 2026-09-20 `tests.test_service` 3/3 OK／Three service HTTP tests passed 2026-09-20

ZH：以下既有 Q0–Q5 勾選表示其日誌宣告的特定交付範圍，不能解讀為 Clone 生成引用／漂移、廣泛生物品質或 Gemma 特化全數通過。未完成的擴充範圍由計畫 24 繼續追蹤；本輪詳細證據見 `log/2026-09-17/review_fixes_completion.md`。

EN: Existing Q0–Q5 checkmarks below refer to the bounded deliveries declared in their logs, not universal success for Clone grounding/drift, biological quality, or Gemma specialization. Plan 24 tracks remaining expanded work; repair evidence is in `log/2026-09-17/review_fixes_completion.md`.

- [ ] Replace `foundation_models/tiny_vlm_stub.py` with real TinyVLM backend
- [x] Import full ASAL `run_asal.py` logic into `research/asal_engine/`
- [x] Connect `core/artifacts.py` to ASAL engine outputs
- [x] Add substrate registry for Boids / NCA / Lenia / ReactionDiffusion
- [x] Normalize output summary schema across ASAL / Digital Clone / GenAI
- [x] Add log-aware developer workflow using `tools/query_worklog.py`
- [x] Implement Phase 0 Missing Foundational Modules (RuntimeManager, ChromaDB Shared Storage, MLflow Experiment Tracking)
- [x] Upgrade Digital Clone Memory to use ChromaDB Vector Store
- [x] Upgrade ASAL configs to use OpenCLIP instead of random embedder
- [x] Fix ASAL OpenCLIP input types, randomness reproducibility, and Boids damping (Priority 1-3)
- [x] Improve Cell Division/Fusion Narrative acceptance criteria (Priority 4)
- [x] Repair Digital Clone cross-session memory retrieval and persona isolation (Priority 5)
- [x] Complete owned-model specialization engineering: prompt_profile, lineage tracking, dataset contracts, and training scaffold (Priority 6)

- [x] Q0: Sync `docs/STATUS.md` and stale matrix rows after Priority 1–6 (`docs/PLAN/23`)
- [x] Q1: ASAL multi-seed narrative quality evidence under upgraded criteria
- [x] Q2: Narrow phase-aware Boids dynamic control from plan 22
- [x] Q3: Digital Clone real-LLM cross-session memory and drift evaluation
- [x] Q4: First clone SFT dataset + tiny LoRA smoke with lineage
- [x] Q5: Align `requirements.txt` / environment reproducibility notes

## Plan 25 delivery (2026-09-20; audited 2026-09-21) / Plan 25 交付與核對

- [x] E0: Write `docs/PLAN/25_RELAY_PROGRESS_AND_NEXT_DELIVERY.md` + sync INDEX/STATUS/TASKS/log
- [x] E1: Close remaining HTTP service integration tests (D0 remainder) — 2026-09-20 `unittest tests.test_service` 3/3 OK
- [x] E2: Advance ASAL NCA/Lenia + multi-seed visual protocol pack under `runs/asal/` — 2026-09-20 4/4 ok
- [x] E3: Save Clone grounding/drift evaluation evidence / 保存評估證據；目前 all-True report 為 mock，較早真模型只有有限 recall／keyword gate 證據。品質重驗列於上方 D2。 / Current all-True report is mock; earlier real runs support limited recall/keyword metrics. Quality revalidation remains under D2 above.
- [x] E4: Final bilingual acceptance package for user verification — 2026-09-20

ZH：Plan 25 的評估交付與品質通過分開記錄；Plan 24 D0–D6 完整門檻仍保留。

EN: Plan 25 evaluation delivery and quality acceptance are tracked separately. Plan 24 D0–D6 remains the full acceptance checklist.
