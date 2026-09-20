# Clone synthetic smoke v1 / Clone 合成冒煙資料集 v1

ZH：這是為本專案以確定性程式建立的英文合成資料，沒有使用者個資或抓取資料。資料僅測試輸出格式、從給定記憶取值、缺資料時承認不知道，以及固定身分壓力；不是自然對話基準，也不代表長期人格品質。所有 access code 都是虛構字串。

EN: This English synthetic dataset is authored deterministically for this project, with no personal or scraped data. It tests output formatting, extraction from supplied memory, admitting unknown facts, and fixed-identity pressure. It is not a natural-conversation benchmark or evidence of long-term persona quality. All access codes are fictional strings.

| Split / 切分 | Recall / 記憶 | Unknown / 未知 | Identity pressure / 身分壓力 | Total / 合計 |
|---|---:|---:|---:|---:|
| Train / 訓練 | 96 | 16 | 16 | 128 |
| Validation / 驗證 | 16 | 4 | 4 | 24 |
| Test / 測試 | 16 | 4 | 4 | 24 |

ZH：每列遵循 `clone.schema.json`；評分中繼資料獨立放在 `*.cases.json`，不傳入模型。Manifest 保存每檔及每列 SHA-256。切分間沒有重複列、subject ID 或正確 code 組合；code 以固定 seed 打散，無法從 subject ID 直接推導。切分共享任務模板、Lex 人格及部分單字，因此不能把結果描述為跨領域泛化。

EN: Rows follow `clone.schema.json`; scoring metadata resides separately in `*.cases.json` and is never sent to the model. The manifest records file and row SHA-256 values. Splits have disjoint rows, subject IDs, and correct code pairs; a fixed seed shuffles codes so they cannot be derived directly from subject IDs. Task templates, the Lex persona, and some individual words are shared, so results must not be described as out-of-domain generalization.

ZH：Validation response-token NLL 用於選擇 checkpoint；test 不參與 loss backward 或 checkpoint 選擇。評分讀取真實 greedy decoding，檢查 answer body 中正確值且沒有 distractor，格式分數另外列出。這是確定性字串 rubric，不是人工或 LLM 語意評分。

EN: Validation response-token NLL selects the checkpoint. Test data never participates in backpropagation or checkpoint selection. Evaluation consumes actual greedy decoding and checks that the answer body contains the correct value without the distractor; format scores are separate. This is a deterministic string rubric, not human or LLM semantic judgment.

```bash
python3 tools/run_python.py -m training.datasets.build_clone_smoke
python3 tools/run_python.py -m unittest discover -s tests -p 'test_training*.py' -v
```

ZH：資料來源及授權由 `manifest.json` 記錄；內容為專案合成資料、CC0-1.0。生成器位於 `training/datasets/build_clone_smoke.py`。

EN: `manifest.json` records provenance and licensing: project-authored synthetic data, CC0-1.0. The generator is `training/datasets/build_clone_smoke.py`.
