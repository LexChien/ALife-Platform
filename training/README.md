# Owned-Model Engineering / 自有模型工程

ZH：本目錄提供 clone、planner、judge 的資料契約、SFT 前置檢查與格式評估。三筆 examples 僅驗證契約；後續工作另新增 clone_smoke_v1 切分資料與 SmolLM2 真實 LoRA smoke，見下方。Gemma 專項 ModelSpec 的 LoRA 欄位仍描述預定用途，不代表已產出對應的 Gemma 權重。

EN: This directory provides data contracts, SFT preflight, and format evaluation for clone, planner, and judge. The three examples only validate contracts; subsequent work separately added split clone_smoke_v1 data and a real SmolLM2 LoRA smoke, described below. LoRA fields in Gemma specialization ModelSpecs still describe intended use, not existing Gemma weights.

## Layout / 目錄

- `datasets/*.schema.json`：Draft 7 機器規格。EN: Machine-readable Draft 7 contracts.
- `datasets/*_schema.md`：完整雙語契約與範例。EN: Bilingual contracts and examples.
- `datasets/contracts.py`：型別／跨欄位檢查與執行期格式序列化。EN: Type/cross-field validation and runtime-format serialization.
- `eval/validate_datasets.py`：驗證 schema 與 JSONL。EN: Schema and JSONL validation.
- `eval/evaluate_predictions.py`：格式通過率，judge 另列合法輸出的 RMSE。EN: Format pass rate and judge RMSE on valid outputs.
- `lora/train_placeholder.py`：寫入 prepared.jsonl 與前置檢查 manifest，沒有 trainer。EN: Writes prepared.jsonl and a preflight manifest; no trainer is implemented.
- `export/export_placeholder.py`：匯出尚未實作，回傳 exit code 2。EN: Export remains unimplemented and returns exit code 2.

## Validation / 驗證

ZH：在 repository 根目錄執行。9/16 最初檢查時只有系統 Python 提供 jsonschema；9/17 接續時 `.venv` 已具備，可用 `python3 tools/run_python.py` 統一啟動。其他環境可使用 `training/requirements.txt` 建立驗證依賴。

EN: Run from the repository root. Initially on 9/16 only system Python supplied jsonschema; the resumed `.venv` has it on 9/17 and can be launched through `python3 tools/run_python.py`. Other environments can install validation dependencies from `training/requirements.txt`.

```bash
python3 -m training.eval.validate_datasets
python3 -m training.eval.validate_datasets --profile clone --dataset training/datasets/examples/clone.jsonl
python3 -m unittest discover -s tests -p 'test_training_contracts.py' -v
```

ZH：前置檢查命令如下。`--base-model-reference` 只記錄宣告來源，工具不驗證或下載；必須在真實訓練前确认可訓練權重及來源。GGUF 推論資產不能直接視為 LoRA 訓練基底。

EN: The preflight command below records the declared `--base-model-reference` without verifying or downloading it. Confirm trainable weights and their source before actual training. The GGUF inference artifact cannot be assumed to be a LoRA training base.

```bash
python3 -m training.lora.train_placeholder --profile clone \
  --dataset training/datasets/examples/clone.jsonl \
  --output runs/training/clone_preflight \
  --base-model-reference 'UNVERIFIED_TRAINABLE_BASE'
python3 -m training.eval.evaluate_predictions --profile clone \
  --dataset training/datasets/examples/clone.jsonl \
  --predictions runs/training/clone_preflight/prepared.jsonl
```

ZH：上例將標準答案當成預測，只是格式流程檢查，不能作為模型評估成績。前置檢查 manifest 固定標示 `prepared_not_trained`，並記錄資料 SHA-256、筆數與未確認的基底來源。

EN: The example uses targets as predictions solely to test the formatting flow; it is not a model evaluation result. The preflight manifest explicitly records `prepared_not_trained`, dataset SHA-256, row count, and an unverified base reference.

## Existing training work and next stage / 已有訓練與下一階段

ZH：後續工作已有 `datasets/build_clone_smoke.py`、`lora/clone_smoke.py` 及 `tools/run_q4_minimal_lora_smoke.py`。完整資料切分為 128／24／24；`runs/training/clone_lora_minimal_q4/` 的既有 SmolLM2 smoke 實際使用 32／8，狀態為 `trained_minimal_smoke`。本輪不重跑訓練、不將此成果稱為 Gemma 特化；獨立測試集的廣泛行為品質仍需驗證。來源及限制見 [MODEL_LINEAGE](../docs/MODEL_LINEAGE.md)。

EN: Subsequent work added `datasets/build_clone_smoke.py`, `lora/clone_smoke.py`, and `tools/run_q4_minimal_lora_smoke.py`. Full dataset splits are 128/24/24; the existing SmolLM2 smoke under `runs/training/clone_lora_minimal_q4/` used 32/8 with status `trained_minimal_smoke`. This continuation does not rerun training or label that result as Gemma specialization. Broader behavior on held-out test data still needs validation. See [MODEL_LINEAGE](../docs/MODEL_LINEAGE.md) for provenance and limits.
