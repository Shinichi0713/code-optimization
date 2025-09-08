# ハードウェアで差がつくこと

同じCNNの推論をしても、使う **SoC（Zynq, Versal, Intel SoCなど）によって最適化の効果やアプローチに差** が出ます。整理するとこうなります👇

---

## 🔹 1. CPU部分の性能差

SoCは **FPGAだけでなくCPUコア（ARM Cortex-A系など）** を持っています。

* **Zynq-7000**
  * ARM Cortex-A9（デュアルコア, ~1GHz） → 性能控えめ
  * 前処理/後処理がボトルネックになりやすい
* **Zynq UltraScale+ (ZUシリーズ)**
  * ARM Cortex-A53 / R5 → より高性能
* **Versal ACAP**
  * ARM Cortex-A72 → さらに強力、ソフト処理部分も高速

👉 同じFPGA最適化をしても、**CPU側が遅ければ全体のスループットに限界** が出る。

つまり「最適化効果 = CPU性能 × FPGA性能」に依存。

---

## 🔹 2. FPGAリソースの違い

* **LUT, FF, DSP, BRAMの数がSoCごとに異なる**
* CNNは畳み込みで大量のDSP（乗算器）を使うので、DSP数が多いSoCほど並列度を上げられる。
  * 小規模SoC（Zynq-7010） → ResNet-18 程度が限界
  * 大規模SoC（Versal VCK190） → ResNet-50/101 もリアルタイム可能

👉 「どのくらい並列化できるか」がSoC依存。

> 高速かのカギはどれくらい並列処理できるか。
>
> これはSoC次第で変わる。

---

## 🔹 3. メモリ帯域

* CNN推論は **外部DDRメモリとのデータ転送速度** がボトルネックになりやすい。
* **Zynq-7000** : DDR3 (帯域狭め)
* **Zynq UltraScale+** : DDR4
* **Versal** : HBM(High Bandwidth Memory) 搭載モデルあり → 圧倒的に高速

👉  **同じ最適化でも、HBM搭載SoCなら性能が数倍に跳ね上がる** 。

---

## 🔹 4. DPU (Deep Learning Processing Unit) 世代の違い

* Vitis AIが使う「DPU IPコア」にも世代差あり。
  * DPUv2（Zynq UltraScale+ 向け）
  * DPUv3 / AI Engine（Versal向け） → CNN処理をさらに効率化

👉 SoCが新しいほど「同じモデルでも省リソース＆高性能で実行」できる。

---

## 🔹 5. 電力効率

* FPGAは低消費電力で並列処理できるのが強み。
* 小規模Zynq → 消費電力は低いが処理能力も低い
* Versal + HBM → 高性能だが消費電力も上がる
* そのため「最適化効果を **性能/W (FPS/W)** で比較」するとSoCごとに差が出る

---

## ✅ まとめ

最適化の効果は、SoCのハード構成によって次の点で差がつきます👇

1. **CPU性能** （前処理/制御がどこまで速いか）
2. **FPGAリソース量 (DSP, LUT, BRAM)** （並列化の限界）
3. **メモリ帯域 (DDR3, DDR4, HBM)** （データ転送効率）
4. **DPU世代 / AI Engine** （FPGA上のCNN専用ハードIPの進化）
5. **電力効率 (Performance per Watt)**

---

👉 もし実際にお使いのSoCボード（例: **ZCU104, ZCU102, Versal VCK190** など）が決まっているなら、その環境で「どのくらい最適化効果が期待できるか」を具体的に比較できます。

ご利用予定のボードはどれに近いですか？
