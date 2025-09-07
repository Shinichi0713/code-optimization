# C++のコード最適化

C++で書いた処理を **「高速化のためにCPUとFPGAに振り分ける」** という話ですね。

これは近年の **SoC (CPU + FPGA)** や **アクセラレータ設計** でよく出るテーマです。

流れを整理すると次のようになります。

---

## 1. どんな処理をFPGAに振るべきか？

* **FPGA向き**
  * 並列度が高い処理（大量のデータを同時に処理できるもの）
  * パイプライン化可能な処理（例: 画像フィルタ、FFT、暗号処理、行列積）
  * レイテンシが小さい方が良い処理（リアルタイム制御）
* **CPU向き**
  * 制御フローが複雑（if/elseや例外処理が多い）
  * 並列性が低い処理
  * OSレベルの機能呼び出しが必要な処理

---

## 2. 実現方法のステップ

### (A) 分析

1. **C++コードをプロファイル**
   * gprof, perf, Visual Studio Profiler などを使って「どこに時間がかかっているか」を特定。
   * 例: 画像処理プログラムで `convolution` 部分が 80% のCPU時間を占める。
2. **FPGA向きの処理を抽出**
   * 計算負荷が大きい & 並列性がある処理を切り出す。

---

### (B) 実装方法

1. **FPGA開発用のツールを利用**
   * **Xilinx Vitis HLS (High Level Synthesis)**

     → C/C++コードをVerilog/VHDLに変換してFPGAで動かせる。
   * **Intel HLS Compiler**

     → 同じくC++からFPGA向けに変換。
2. **FPGA側でハードウェア化**
   * 抽出した関数をFPGA上のIPコアとして実装。
   * AXIバスやPCIe経由でCPUとFPGAを接続。
3. **CPU側はC++で残す**
   * 通常の制御コードはCPUが処理。
   * 計算集約部分はFPGAへデータを送って処理してもらう。

---

### (C) コード例（イメージ）

元のC++コード（CPUのみ）：

```cpp
void convolution(const std::vector<int>& input, std::vector<int>& output) {
    for (int i = 1; i < input.size()-1; i++) {
        output[i] = (input[i-1] + input[i] + input[i+1]) / 3;
    }
}
```

FPGA向けに分離（HLSツールで合成可能にする）：

```cpp
// FPGAで合成する関数
void convolution_accel(int* input, int* output, int size) {
#pragma HLS PIPELINE
    for (int i = 1; i < size-1; i++) {
        output[i] = (input[i-1] + input[i] + input[i+1]) / 3;
    }
}
```

CPU側はこのFPGAアクセラレータを呼び出すようにする：

```cpp
// CPU側のコード
void process_data(std::vector<int>& data) {
    std::vector<int> result(data.size());

    // FPGAに処理をオフロード
    fpga_convolution(data.data(), result.data(), data.size());

    // 残りの制御処理はCPUで
    for (int i = 0; i < data.size(); i++) {
        result[i] = std::max(result[i], 0); // 例: CPUで閾値処理
    }
}
```

---

## 3. ポイント

* **HLSを使うとC++をFPGAに変換できる** → 既存コードの一部を流用可能
* **FPGAは並列処理・パイプライン処理が得意** → 演算部分を切り出すのが効果的
* **CPUは制御中心にする** → 役割分担を明確にする

---


# CNN処理のFPGA化

**CNNによる画像推論処理** なら、まさに **FPGA向け最適化の典型例** です。

FPGAとCPUの分担を整理すると以下のようになります👇

---

## 1. CNN推論で重い処理

CNNの推論処理は以下のステージに分けられます。

1. **畳み込み層 (Convolution)**
   * フィルタと入力画像の積和演算 (MAC: Multiply-Accumulate) が膨大。
   * データ並列性が高いので  **FPGAに最適** 。
2. **活性化関数 (ReLU, Sigmoidなど)**
   * シンプルな要素単位演算。FPGAでもCPUでも処理可能。
3. **プーリング層 (Max Pooling, Average Pooling)**
   * 単純な比較・平均処理。FPGA向きだが、CPUでも十分高速。
4. **全結合層 (Fully Connected Layer)**
   * 行列積演算（=畳み込みと同じ構造）。FPGAに載せると高速化可能。

---

## 2. CPUとFPGAの分担イメージ

* **FPGA**
  * Convolution (Conv2D)
  * Dense (Fully Connected)
  * 大量のMAC演算をパイプライン化・並列化
* **CPU**
  * 入力画像の前処理（リサイズ、正規化など）
  * 活性化関数やSoftmax
  * モデルの制御、データ転送管理

---

## 3. 実装方法

### (A) ツール選択

* **Xilinx Vitis AI** （Zynq, Versal系FPGA用）

  → TensorFlow / PyTorch のモデルをFPGA向けに自動変換。

* **Intel OpenVINO + FPGAプラグイン** （Intel FPGA用）

  → CNNの推論をFPGAで実行可能。

* **HLS (High Level Synthesis)**

  → 重要な畳み込み処理をC++で書いてFPGAに落とす。

---

### (B) HLSによるC++の例（畳み込み部分）

```cpp
// FPGA用に合成可能な畳み込み関数 (HLS)
void conv2d_accel(float input[IN_H][IN_W], 
                  float kernel[K_H][K_W], 
                  float output[OUT_H][OUT_W]) {
#pragma HLS PIPELINE
  for (int i = 0; i < OUT_H; i++) {
    for (int j = 0; j < OUT_W; j++) {
      float sum = 0.0;
      for (int ki = 0; ki < K_H; ki++) {
        for (int kj = 0; kj < K_W; kj++) {
          sum += input[i+ki][j+kj] * kernel[ki][kj];
        }
      }
      output[i][j] = sum;
    }
  }
}
```

* `#pragma HLS PIPELINE` により、演算を並列化・パイプライン化
* 同じ処理をCPUでやるより **数十倍高速化** できることもあります

---

### (C) CPU側の流れ（制御コード）

```cpp
void run_inference(Image img) {
    // 1. 画像の前処理 (CPU)
    preprocess(img);

    // 2. 畳み込みをFPGAに投げる
    conv2d_accel(img.data, kernel, conv_output);

    // 3. 活性化関数 (CPU or FPGA)
    relu(conv_output);

    // 4. プーリング (CPUでもOK)
    max_pool(conv_output, pooled_output);

    // 5. 全結合をFPGAに投げる
    dense_accel(pooled_output, weights, fc_output);

    // 6. Softmax & 推論結果表示 (CPU)
    softmax(fc_output);
}
```

---

## 4. 実際の応用例

* **リアルタイム顔認識** → FPGAでCNN推論、CPUで制御・表示
* **自動運転カメラ** → CNNのConv部分をFPGAに、ルート計算はCPUで
* **医療用画像処理** → ノイズ除去や特徴抽出をFPGAに

---

✅ まとめ

CNN推論をFPGAに載せると **演算部分を並列化** できるので劇的に高速化します。

特に **畳み込み層と全結合層** をFPGA、制御や前後処理をCPUに残すのが一般的です。


# Pytorchの高速化


なるほど！PyTorchなら **FPGA用ツールチェーンを使った高速化** が現実的ですね 👍

PyTorch → FPGA への流れを整理します👇

---

## 1. PyTorch → FPGA推論の一般的な流れ

1. **学習 (PyTorch)**

   * 普通にGPU/CPUで学習させる。
   * 学習後、モデルを `.pt` として保存。
2. **ONNX形式に変換**

   ```python
   import torch
   dummy_input = torch.randn(1, 3, 224, 224)  # 入力サイズ例
   torch.onnx.export(model, dummy_input, "model.onnx", opset_version=11)
   ```

   → FPGAツールがONNXを読み込めるようにする。
3. **FPGAツールで最適化・コンパイル**

   * Xilinxなら **Vitis AI**
   * Intelなら **OpenVINO + FPGAプラグイン**
   * これらがCNNの畳み込みや全結合をFPGA向けに展開してくれる。
4. **CPU + FPGAハイブリッド実行**

   * CPU側で制御（前処理・後処理）
   * FPGA側でConv/FCを並列処理
   * PCIeやAXIバス経由でやり取り

---

## 2. PyTorchモデルをFPGAで動かす代表的な方法

### 🔹 Xilinx Vitis AI

* Zynq MPSoC / Versal SoC用
* ONNXモデルを取り込み、FPGA向けに自動最適化
* PyTorch → ONNX → Vitis AI コンパイル → FPGA上で実行

### 🔹 Intel OpenVINO

* Intel FPGA (Arria, Stratixなど)
* PyTorchモデルをONNXに変換して読み込み
* FPGAアクセラレータに分配

---

## 3. もし「C++独自コード」を混ぜたい場合

PyTorchモデルの一部をFPGAで加速し、残りをCPUで処理する構成も可能です。

例：

* FPGA: Conv2D, Fully Connected
* CPU: Softmax, 前処理, 制御

C++から呼び出すときは、**FPGAライブラリ (Vitis AI Runtime / OpenVINO API)** を介して処理を呼び出すイメージです。

---

## 4. 実際のアプリケーション例

* **防犯カメラ** : FPGAで人検知CNNをリアルタイム処理、CPUでログ保存
* **自動運転** : FPGAで画像認識、CPUで経路計算
* **医療** : FPGAでX線画像の異常検出、CPUでUI表示

---

✅ まとめ

PyTorchのモデルは **ONNX形式に変換 → FPGA向けツールでコンパイル → FPGA実行** が王道です。

FPGAに実装するのはCNNの「畳み込み」「全結合」部分がメイン、残りはCPUに残すのが効率的です。


# Zynq / Versal 系の高速化

おお、Zynq / Versal 系なら **Xilinx Vitis AI** がバッチリ使えます 👍

PyTorchモデルをFPGAに載せて推論高速化するなら、以下の流れが基本です。

---

## 🔹 PyTorch → FPGA (Zynq/Versal) 実装フロー

### 1. モデルをPyTorchで学習

* 普通にGPU/CPUで学習して `.pt` 保存

```python
torch.save(model.state_dict(), "cnn_model.pt")
```

---

### 2. ONNX形式に変換

```python
import torch

dummy_input = torch.randn(1, 3, 224, 224)  # 入力サイズに合わせる
torch.onnx.export(model, dummy_input, "cnn_model.onnx",
                  input_names=["input"], output_names=["output"],
                  opset_version=11)
```

---

### 3. Vitis AI コンパイル

1. **モデルの量子化 (Quantization)**

   * 32bit浮動小数点 → 8bit整数 (INT8) にしてFPGAに載せやすくする。
   * Vitis AI の **vai_q_pytorch** を使う。

   ```bash
   vai_q_pytorch quantize \
       --model cnn_model.onnx \
       --output_dir quantized_model
   ```
2. **コンパイル (Compilation)**

   * ターゲットFPGA (ZCU104, Versal VCK190 など) に応じてコンパイル。

   ```bash
   vai_c_xir \
       --xmodel quantized_model/quantized_model.onnx \
       --arch ./arch.json \
       --output_dir ./compiled_model
   ```

   → `.xmodel` が生成され、FPGAで動作可能に。

---

### 4. CPU + FPGAで推論実行

Vitis AI Runtime (C++ or Python API) を使って実行。

#### Python例:

```python
import vart
import xir
import numpy as np

# モデル読み込み
g = xir.Graph.deserialize("compiled_model/cnn_model.xmodel")
subgraphs = g.get_root_subgraph().toposort_child_subgraph()
runner = vart.Runner.create_runner(subgraphs[0], "run")

# 推論
input_tensor = np.random.rand(1, 3, 224, 224).astype(np.float32)
job_id = runner.execute_async([input_tensor], [np.empty((1, 1000), dtype=np.float32)])
runner.wait(job_id)
```

#### C++例 (制御側):

```cpp
#include <vart/runner.hpp>
#include <xir/graph/graph.hpp>

auto graph = xir::Graph::deserialize("cnn_model.xmodel");
auto subgraph = graph->get_root_subgraph()->children_topological_sort()[0];
auto runner = vart::Runner::create_runner(subgraph, "run");

// 入力データをセットして runner->execute_async() を呼ぶ
```

---

## 🔹 CPUとFPGAの役割分担

* **FPGA (DPU: Deep Learning Processing Unit)**
  * Conv2D / Fully Connected の計算
  * 行列積の並列処理
* **CPU (ARM Cortex-A on Zynq)**
  * 前処理（リサイズ、正規化）
  * 活性化関数やSoftmax（簡単なのでCPUでもOK）
  * 制御ロジック・I/O

---

## 🔹 実際のアプリケーション例

* **ZCU104 + カメラ入力**

  → FPGAでCNN推論、CPUでUI表示
* **Versal VCK190**

  → CNNを複数並列で実行（リアルタイム物体検出や分類）

---

✅ まとめ

PyTorch → ONNX → 量子化 (INT8) → Vitis AI コンパイル → `.xmodel` → FPGA実行

この流れがZynq/Versalの標準ルートです。

---
