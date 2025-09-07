# 高速化の評価

FPGAやCPU向けにコードを最適化したとき、「どれだけ速くなったか？」を **定量的に評価** するにはいくつかの標準的な指標・手段があります。

---

## 🔹 1. 実行時間（Latency）

* **最も基本的な評価方法**
* 最適化前後で同じ入力を処理させ、処理完了までの時間を比較する。
* 計測方法：
  * **CPU側** : `time.time()` (Python) や `std::chrono` (C++)
  * **FPGA側** : Vitis AI / Runtime API で 1バッチごとの実行時間を取得
* 指標例：
  * 最適化前: 100ms/フレーム
  * 最適化後: 10ms/フレーム

    → **10倍高速化**

---

## 🔹 2. スループット（Throughput）

* 1秒あたりに処理できるサンプル数 (FPS: frames per second, images/sec など)。
* CNN推論では **FPS (frames per second)** がよく使われる。
* 計測例：1秒間に 200 枚の画像を処理 → 200 FPS

---

## 🔹 3. リソース効率（Performance per Watt）

* FPGAは「同じ性能でより省電力」という利点がある。
* **処理性能 (OPS, FPS) ÷ 消費電力(W)** で比較すると分かりやすい。
* 例：
  * CPU: 200 FPS, 20W → 10 FPS/W
  * FPGA: 200 FPS, 5W → 40 FPS/W

    → **効率4倍**

---

## 🔹 4. ハードウェアリソース使用率

FPGAではコード最適化の効果を **LUT/FF/BRAM/DSP使用率** でも評価します。

* LUT/FF → 論理回路規模
* BRAM → オンチップメモリ使用量
* DSP → 乗算器使用数（CNNで重要）

  最適化により、DSPの使用効率が向上して「同じチップでより大きなモデル」が載ることもあります。

---

## 🔹 5. 実際のツールによる評価

* **Vitis AI Profiler**
  * レイヤーごとの実行時間を測定
  * CPU/FPGAのボトルネックを可視化
* **perf (Linux)** , **gprof (C++)**
* CPUコードの関数ごとの実行時間を測定
* **電力計測ツール (Xilinx Power Estimator, Boardモニタ)**
  * 最適化前後での消費電力変化を確認可能

---

## ✅ まとめ

コード最適化の効果は、以下の複数の指標で定量的に評価できます。

1. **実行時間 (Latency)**
2. **スループット (FPS / images/sec)**
3. **効率 (Performance per Watt)**
4. **FPGAリソース使用率 (LUT, DSP, BRAM)**
