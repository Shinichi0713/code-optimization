# ONNX変換

PyTorchモデルを **ONNX (Open Neural Network Exchange)** に変換すると、中身の性質や使い方が大きく変わります。

---

## 🔹 PyTorch モデルの状態

* Python環境で動作（PyTorchランタイムが必要）
* 動的計算グラフ（eager execution）
* GPU/CPUに対してはPyTorchの最適化済みカーネルを使う

👉  **柔軟性は高いが、PyTorch環境が必須** 。FPGAや組込み環境では動かしづらい。

---

## 🔹 ONNX 変換後のモデル

1. **計算グラフに固定される**
   * PyTorchの動的グラフが **静的グラフ** に変換される。
   * 各レイヤー（Conv, ReLU, Pooling…）がONNXの標準オペレーターに置き換えられる。
2. **フレームワーク非依存になる**
   * ONNX形式は中間表現（IR）。
   * TensorRT, OpenVINO, TVM, Vitis AI, ONNX Runtime など、さまざまなランタイムやハードウェアで実行可能。
3. **最適化しやすくなる**
   * ONNX Optimizer や各ベンダーのコンパイラが、演算を再配置・融合（fuse）できる。
   * 例: Conv + BatchNorm + ReLU → 1つのカーネルにまとめて効率化。
4. **FPGA/SoCで使いやすい**
   * Xilinx Vitis AI では、ONNXモデルを DPU (Deep Learning Processing Unit) 向けにコンパイル可能。
   * PyTorchのままではコンパイルできないので  **ONNX変換が必須ステップ** 。

---

## 🔹 具体例

### PyTorch 側

```python
import torch
import torchvision.models as models

model = models.resnet18(pretrained=True)
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "resnet18.onnx")
```

### ONNX 側（変換後のイメージ）

```plaintext
graph(%input : Float(1, 3, 224, 224)) {
    %1 = Conv(%input, %conv1_weight, %conv1_bias)
    %2 = BatchNormalization(%1, ...)
    %3 = Relu(%2)
    ...
    %output = Softmax(%last_layer)
    return %output
}
```

👉 各レイヤーが **演算ノードのつながり** として表現される。

👉 これをハードウェアが理解して、最適な形で並列化やアクセラレーションを行う。

---

## ✅ まとめ

PyTorch → ONNX 変換すると：

* **動的モデル → 静的グラフモデル** に変わる
* **フレームワーク依存から解放**され、様々なランタイムやハードで実行可能
* **最適化やFPGA実装に適した形式**になる

---

👉 FPGA (Xilinx Zynq / Versal) でCNN推論したい場合、

**PyTorchモデル → ONNX → Vitis AI (DPU用バイナリ)** という流れになります。

---

もしよければ、実際に使いたいCNN（ResNet, YOLO, UNetなど）を教えていただければ、ONNX変換後に **Vitis AIでどんな最適化が可能か** を具体的に整理しますよ。どのモデルを考えていますか？
