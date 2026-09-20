# ASAL executable baselines / ASAL 可執行基線

## Scope / 範圍

English: ASAL now exposes five simulators through the same `reset(theta, seed)`,
`step(substeps)`, and `render()` interface: reaction diffusion, boids, Lenia,
neural cellular automata (NCA), and controlled particles. The narrative scorer
measures image geometry and temporal continuity. A passing score does not prove
biological realism or autonomous reproduction.

中文：ASAL 現在以相同的 `reset(theta, seed)`、`step(substeps)`、`render()`
介面提供五種模擬器：反應擴散、boids、Lenia、神經細胞自動機（NCA）、受控粒子。
敘事評分器量測影像幾何與時間連續性。分數通過不代表生物真實性或自主繁殖已獲證明。

## Controlled particles / 受控粒子

English: `controlled_cells` integrates overdamped particle motion with local
repulsion, within-cohort cohesion, and an explicit external phase potential.
The same particles persist through all frames. The split phase moves two cohort
anchors apart; fusion brings them together. Bilinear deposition conserves raster
mass before a fixed Gaussian density display. Rendering never removes small
components, respawns particles, or paints phase-specific shapes. This is a
controlled mechanics baseline; the phase sequence is provided by the experiment.

中文：`controlled_cells` 積分具局部排斥、群內凝聚及明確階段外部勢場的過阻尼
粒子運動。所有影格保有相同粒子；分裂階段分開兩群錨點，融合階段使其靠攏。
雙線性沉積在固定高斯密度顯示前保持光柵質量。渲染不移除小碎片、不重生粒子，
也不繪製各階段專屬形狀。這是受控力學基線，階段次序由實驗提供。

English: Its five theta values are anchor mobility, local repulsion strength,
cohort cohesion, split-anchor distance in pixels, and stochastic force amplitude.
`configs/asal/controlled_cell_fusion.yaml` provides bounded search values. A frame
advances one physical time unit; more substeps refine the numerical integration.

中文：五個 theta 值依序為錨點遷移率、局部排斥強度、群內凝聚、分裂錨點距離
（像素）、隨機力振幅。`configs/asal/controlled_cell_fusion.yaml` 提供有界搜尋值。
一個影格前進一個物理時間單位；增加 substeps 會細化數值積分。

## NCA and Lenia / NCA 與 Lenia

English: NCA perceives identity, Sobel gradients, and a Laplacian locally, then
applies a shared two-layer tanh network with stochastic cell updates. A bounded
diffusion/growth prior stabilizes the untrained baseline. The five theta values
control diffusion, growth, decay, neural contribution, and update probability.
Network weights can be exported and loaded as NPZ. Default weights are seeded
and untrained; this delivery makes no learned regeneration claim. The local
perception design follows [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/).

中文：NCA 在局部感知自身狀態、Sobel 梯度與 Laplacian，再套用共享的兩層 tanh
網路及隨機細胞更新。有界的擴散／生長先驗用來穩定未訓練基線。五個 theta
依序控制擴散、生長、衰減、神經更新貢獻、更新機率。網路權重可用 NPZ 匯出／載入。
預設權重由 seed 決定但未訓練；本交付不宣稱已學到再生能力。局部感知設計參照
[Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)。

English: Lenia uses a normalized ring kernel, periodic FFT convolution, Gaussian
growth, and bounded Euler integration. Its theta values are growth mean, growth
width, time step, radius/lattice-size ratio, and initial patch density. The seeded
patch can live or die depending on parameters; it is not a curated species.
See [Lenia — Biology of Artificial Life](https://arxiv.org/abs/1812.05433).

中文：Lenia 使用正規化環狀核、週期邊界 FFT 卷積、高斯生長函數及有界 Euler
積分。theta 依序為生長均值、生長寬度、時間步長、核半徑／格點大小比例、初始
斑塊密度。seed 斑塊會依參數存活或消亡，不是經挑選的物種。參見
[Lenia — Biology of Artificial Life](https://arxiv.org/abs/1812.05433)。

## Run and replay / 執行與重播

English: Run from the repository root using the project interpreter launcher.
These CPU baselines need NumPy, Pillow, imageio, and a working ffmpeg backend for
MP4. The heuristic judge does not establish OpenCLIP semantic quality.

中文：在專案根目錄使用專案 Python 啟動器執行。這些 CPU 基線需要 NumPy、Pillow、
imageio，以及可用的 ffmpeg 後端輸出 MP4。啟發式評審不能證明 OpenCLIP 語意品質。

```bash
python tools/run_python.py apps/asal_cli.py --config configs/asal/controlled_cell_fusion.yaml
python tools/run_python.py apps/asal_cli.py --config configs/asal/nca_baseline.yaml
python tools/run_python.py apps/asal_cli.py --config configs/asal/lenia_baseline.yaml
python tools/run_python.py -m research.asal_engine.replay RUN_DIRECTORY --resimulate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/run_python.py tools/asal_quality.py --outdir runs/validation/asal_new_evidence --include-automata
```

English: Each run preserves raw RGB frames, real simulation states, resolved
configuration, seed/theta, scores, source hashes, package versions, GIF/MP4,
substrate statistics, and narrative rejection reasons. Replay verifies artifact
hashes and rescoring; `--resimulate` additionally regenerates every frame and
stored state from seed/theta, requiring exact array equality. Changed source
hashes are reported as unverified even if the score is identical.

中文：每次執行保留原始 RGB 影格、真實模擬狀態、已解析設定、seed/theta、分数、
原始碼雜湊、套件版本、GIF/MP4、基質統計、敘事拒絕原因。重播會驗證檔案雜湊
並重新計分；`--resimulate` 另由 seed/theta 重建每個影格與已存狀態，要求陣列
逐位相等。原始碼雜湊若變更，即使分數相同，也會回報未驗證。

## Benchmark interpretation / 基準解讀

English: `tools/asal_quality.py` fixes paired seeds, candidate count, simulated
frames, phase windows, judge, and the existing acceptance thresholds for both
boids and controlled particles. It reports measured wall time and Wilson 95%
intervals. An additional stricter clean-motion gate requires at least 65% phase
coverage, a 50% consecutive run, at most 10% fragments in qualified frames, and
at least 80% adjacent-frame mass retention. This gate never changes the original
reward or acceptance result. A zero-split-force ablation is an explicit failure
control. Automated video decoding and all-frame contact sheets support manual
review, which must be recorded separately after viewing the actual images.

中文：`tools/asal_quality.py` 為 boids 與受控粒子固定配對 seeds、候選數、模擬
影格數、階段區間、評審及既有驗收門檻，並回報實測耗時及 Wilson 95% 區間。
額外的嚴格乾淨運動門檻要求各階段至少 65% 合格覆蓋、50% 連續合格長度、合格
影格最多 10% 碎片，以及相鄰影格至少 80% 質量保留。此門檻不修改原始獎勵或
验收結果。將分裂外力歸零的消融實驗是明確的失敗對照。自動影片解碼與所有影格
接觸表支援人工檢查；必須實際看過影像後，另行記錄檢查結果。
