# LoRA (Low-Rank Adaptation) 詳細ガイド

このドキュメントでは、LoRA（Low-Rank Adaptation）の仕組み、パラメータの設定方法、およびファインチューニングへの影響について詳しく解説します。

## 目次

1. [LoRAとは](#loraとは)
2. [LoRAの仕組み](#loraの仕組み)
3. [LoRAパラメータ詳細](#loraパラメータ詳細)
4. [パラメータ設定ガイド](#パラメータ設定ガイド)
5. [Adapterの仕組み](#adapterの仕組み)
6. [実践的な設定例](#実践的な設定例)
7. [トラブルシューティング](#トラブルシューティング)

## LoRAとは

**LoRA (Low-Rank Adaptation)** は、大規模言語モデルを効率的にファインチューニングするための手法です。

### 従来のファインチューニングの問題点

```
従来の方法:
- 全パラメータを更新 → 大量のメモリとストレージが必要
- GPT-3.5: 175B パラメータ → 175B個すべてを調整
- メモリ使用量: 数百GB〜数TB
```

### LoRAの解決策

```
LoRAの方法:
- 低ランク行列で近似 → 少数のパラメータのみ更新
- GPT-3.5: 175B パラメータ → 数百万〜数千万個のみ調整
- メモリ使用量: 数GB〜数十GB
```

## LoRAの仕組み

### 数学的な基盤

LoRAは、重み行列の更新を低ランク行列の積で近似します：

```
元の重み行列: W ∈ R^(d×k)
更新後の重み: W' = W + ΔW

LoRAでは:
ΔW = A × B
ここで A ∈ R^(d×r), B ∈ R^(r×k), r << min(d,k)
```

### 視覚的な説明

```
従来のファインチューニング:
┌─────────────────┐
│   Original W    │  ← 全体を更新
│   (d × k)       │
└─────────────────┘

LoRA:
┌─────────────────┐    ┌──────┐   ┌──────────┐
│   Original W    │ +  │  A   │ × │    B     │
│   (d × k)       │    │(d×r) │   │  (r×k)   │
│   (frozen)      │    └──────┘   └──────────┘
└─────────────────┘       ↑
                      少数パラメータのみ更新
```

### Transformerモデルでの適用

```python
# Attention層への適用例
class LoRALinear:
    def __init__(self, original_linear, r, alpha):
        self.original = original_linear  # 凍結
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scaling = alpha / r
        
    def forward(self, x):
        original_output = self.original(x)
        lora_output = self.lora_B(self.lora_A(x)) * self.scaling
        return original_output + lora_output
```

## LoRAパラメータ詳細

### 1. ランク (r) - 最重要パラメータ

**定義**: 低ランク行列の次元数

```python
# train_lora.pyでの設定
--lora-r 16  # デフォルト値
```

**影響**:
- **小さい値 (r=4-8)**: 
  - ✅ メモリ効率が良い
  - ✅ 高速訓練
  - ❌ 表現力が限定的
  - 📊 適用例: 簡単なタスク、リソース制約環境

- **中程度 (r=16-32)**:
  - ⚖️ バランスが良い
  - 📊 適用例: 一般的なファインチューニング

- **大きい値 (r=64-128)**:
  - ✅ 高い表現力
  - ❌ メモリ使用量増加
  - ❌ 訓練時間増加
  - 📊 適用例: 複雑なタスク、大規模データセット

**計算例**:
```
Gemma 3-1B の query_proj層 (1152 → 1152):
- 元のパラメータ数: 1152 × 1152 = 1,327,104
- LoRA r=16: (1152×16) + (16×1152) = 36,864 (約2.8%のパラメータ)
- LoRA r=32: (1152×32) + (32×1152) = 73,728 (約5.6%のパラメータ)
```

### 2. アルファ (α) - スケーリング係数

**定義**: LoRA出力のスケーリング係数

```python
# train_lora.pyでの設定
--lora-alpha 32  # デフォルト値
```

**数学的役割**:
```
最終出力 = 元の出力 + (α/r) × LoRA出力
スケーリング = α/r
```

**影響**:
- **α < r (例: α=8, r=16)**:
  - スケーリング = 0.5
  - LoRAの影響を抑制
  - 安定した学習、変化が緩やか

- **α = r (例: α=16, r=16)**:
  - スケーリング = 1.0
  - バランスの取れた影響

- **α > r (例: α=32, r=16)**:
  - スケーリング = 2.0
  - LoRAの影響を強化
  - より大きな変化、学習率に注意

**実践的な設定**:
```python
# 保守的な設定（安定重視）
r=16, alpha=16  # scaling=1.0

# 標準的な設定
r=16, alpha=32  # scaling=2.0

# アグレッシブな設定（大きな変化を期待）
r=16, alpha=64  # scaling=4.0
```

### 3. ドロップアウト (dropout)

**定義**: 過学習防止のための正則化手法

```python
# train_lora.pyでの設定
--lora-dropout 0.05  # デフォルト値 (5%)
```

**影響**:
- **0.0 (ドロップアウトなし)**:
  - 最大の学習能力
  - 過学習のリスク

- **0.05-0.1 (軽い正則化)**:
  - バランスの取れた学習
  - 一般的な設定

- **0.2-0.3 (強い正則化)**:
  - 過学習を強く抑制
  - 学習能力が制限される可能性

### 4. ターゲットモジュール (target_modules)

**定義**: LoRAを適用する層の指定

```python
# 本プロジェクトでの設定例
target_modules = [
    # Attention層（必須）
    "q_proj", "v_proj", "k_proj", "o_proj",
    # FFN層（口調学習に重要）
    "gate_proj", "up_proj", "down_proj"
]
```

**各モジュールの役割**:

1. **Attention層**:
   - `q_proj` (Query): 質問の生成
   - `k_proj` (Key): キーの生成
   - `v_proj` (Value): 値の生成
   - `o_proj` (Output): 出力の結合

2. **FFN層 (Feed Forward Network)**:
   - `gate_proj`: ゲート機構
   - `up_proj`: 次元拡張
   - `down_proj`: 次元縮小

**適用パターン**:
```python
# 最小限（メモリ効率重視）
target_modules = ["q_proj", "v_proj"]

# 標準（推奨）
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

# 最大限（性能重視）
target_modules = [
    "q_proj", "v_proj", "k_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

## パラメータ設定ガイド

### タスク別推奨設定

#### 1. 対話・チャットボット
```bash
python scripts/train_lora.py \
  --model google/gemma-3-1b-it \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --learning-rate 5e-5
```

**理由**: バランスの取れた設定で自然な対話を学習

#### 2. 特定ドメインの専門知識
```bash
python scripts/train_lora.py \
  --model google/gemma-3-1b-it \
  --lora-r 32 \
  --lora-alpha 64 \
  --lora-dropout 0.1 \
  --learning-rate 3e-5
```

**理由**: 高いランクで複雑な知識を学習、ドロップアウトで過学習防止

#### 3. コード生成・プログラミング
```bash
python scripts/train_lora.py \
  --model google/gemma-3-1b-it \
  --lora-r 24 \
  --lora-alpha 48 \
  --lora-dropout 0.05 \
  --learning-rate 2e-5
```

**理由**: 構造化された出力に対応、低い学習率で安定学習

#### 4. 言語スタイル・口調の変更
```bash
python scripts/train_lora.py \
  --model google/gemma-3-1b-it \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --learning-rate 1e-4
```

**理由**: 低ランクでも効果的、FFN層が重要

### データサイズ別推奨設定

| データサイズ | r | α | dropout | 学習率 | 理由 |
|-------------|---|---|---------|--------|------|
| 小 (< 1K例) | 8 | 16 | 0.1 | 1e-4 | 過学習防止 |
| 中 (1K-10K) | 16 | 32 | 0.05 | 5e-5 | バランス |
| 大 (> 10K) | 32 | 64 | 0.05 | 2e-5 | 高表現力 |

### リソース制約別推奨設定

#### メモリ制約環境 (< 8GB VRAM)
```bash
--lora-r 4 --lora-alpha 8 --batch-size 1 --gradient-accumulation-steps 16
```

#### 標準環境 (8-16GB VRAM)
```bash
--lora-r 16 --lora-alpha 32 --batch-size 2 --gradient-accumulation-steps 8
```

#### 高性能環境 (> 16GB VRAM)
```bash
--lora-r 32 --lora-alpha 64 --batch-size 4 --gradient-accumulation-steps 4
```

## Adapterの仕組み

### LoRAアダプターの構造

```
model/
├── adapter_config.json     # LoRA設定
├── adapter_model.safetensors  # LoRA重み
├── tokenizer.json         # トークナイザー設定
├── tokenizer_config.json  # トークナイザー設定
└── training_info.json     # 訓練情報
```

### adapter_config.json の内容例

```json
{
  "alpha": 32,
  "auto_mapping": null,
  "base_model_name_or_path": "google/gemma-3-1b-it",
  "bias": "none",
  "fan_in_fan_out": false,
  "inference_mode": true,
  "init_lora_weights": true,
  "layers_pattern": null,
  "layers_to_transform": null,
  "lora_dropout": 0.05,
  "modules_to_save": null,
  "peft_type": "LORA",
  "r": 16,
  "rank_pattern": {},
  "revision": null,
  "target_modules": [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
  ],
  "task_type": "CAUSAL_LM"
}
```

### アダプターの読み込み過程

```python
# 1. ベースモデルの読み込み
base_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")

# 2. LoRAアダプターの適用
model = PeftModel.from_pretrained(base_model, "path/to/adapter")

# 3. 内部での処理
# - adapter_config.jsonを読み込み
# - LoRA層を元のモデルに注入
# - 元の重みは凍結、LoRA重みのみアクティブ
```

### アダプターのマージ

```python
# アダプターをベースモデルにマージ
merged_model = model.merge_and_unload()

# マージ後の状態:
# W_new = W_original + (alpha/r) * A * B
```

### 複数アダプターの管理

```python
# 異なるタスク用のアダプター
model.load_adapter("path/to/chat_adapter", adapter_name="chat")
model.load_adapter("path/to/code_adapter", adapter_name="code")

# アダプターの切り替え
model.set_adapter("chat")    # チャット用
model.set_adapter("code")    # コード生成用
```

## 実践的な設定例

### 例1: 日本語対話特化モデル

```bash
# データ準備
# data/japanese_chat.jsonl - 日本語の対話データ

# 訓練実行
python scripts/train_lora.py \
  --model google/gemma-3-1b-it \
  --data data/japanese_chat.jsonl \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --epochs 3 \
  --batch-size 2 \
  --learning-rate 5e-5 \
  --output models/japanese_chat
```

**設定理由**:
- `r=16, α=32`: 自然な対話に十分な表現力
- `dropout=0.05`: 軽い正則化で過学習を防止
- `lr=5e-5`: Gemma 3に適した学習率

### 例2: プログラミングアシスタント

```bash
# データ準備
# data/code_assistant.jsonl - プログラミング関連の質問回答

# 訓練実行
python scripts/train_lora.py \
  --model google/gemma-3-1b-it \
  --data data/code_assistant.jsonl \
  --lora-r 24 \
  --lora-alpha 48 \
  --lora-dropout 0.05 \
  --epochs 5 \
  --batch-size 4 \
  --learning-rate 3e-5 \
  --output models/code_assistant
```

**設定理由**:
- `r=24, α=48`: コードの構造的な理解に必要な表現力
- より多くのエポックで複雑なパターンを学習

### 例3: 軽量・高速モデル

```bash
# リソース制約環境向け
python scripts/train_lora.py \
  --model google/gemma-3-1b-it \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.1 \
  --epochs 2 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-4 \
  --output models/lightweight
```

**設定理由**:
- `r=8`: 最小限のパラメータ数
- 勾配蓄積でメモリ効率を改善

## トラブルシューティング

### 学習が進まない場合

**症状**: Loss が下がらない
```bash
# 解決策1: 学習率を上げる
--learning-rate 1e-4  # デフォルトの5e-5から上げる

# 解決策2: αを上げてLoRAの影響を強化
--lora-alpha 64  # デフォルトの32から上げる

# 解決策3: ランクを上げる
--lora-r 32  # デフォルトの16から上げる
```

### 過学習の場合

**症状**: 訓練Lossは下がるが汎化性能が悪い
```bash
# 解決策1: ドロップアウトを強化
--lora-dropout 0.1  # デフォルトの0.05から上げる

# 解決策2: 学習率を下げる
--learning-rate 3e-5  # デフォルトの5e-5から下げる

# 解決策3: ランクを下げる
--lora-r 8  # デフォルトの16から下げる
```

### メモリ不足の場合

```bash
# 解決策1: バッチサイズとランクを下げる
--batch-size 1 --lora-r 8

# 解決策2: 勾配蓄積を活用
--gradient-accumulation-steps 16

# 解決策3: ターゲットモジュールを制限
# スクリプト内で target_modules = ["q_proj", "v_proj"] のみに変更
```

### 収束が遅い場合

```bash
# 解決策1: 学習率スケジューラーを調整
# warmup_ratio=0.1 (デフォルト0.3から下げる)

# 解決策2: AdamWのパラメータ調整
# weight_decay を 0.01 に設定

# 解決策3: より大きなランクを使用
--lora-r 32 --lora-alpha 64
```

## まとめ

LoRAは効率的なファインチューニング手法として、以下の利点があります：

✅ **メモリ効率**: 元モデルの1-10%のパラメータのみ更新
✅ **高速学習**: 少ないパラメータで高速な収束
✅ **モジュラー**: 複数のアダプターを切り替え可能
✅ **スケーラブル**: 様々なサイズのモデルに適用可能

適切なパラメータ設定により、高品質なファインチューニングを実現できます。タスクの性質、データサイズ、リソース制約を考慮して最適な設定を選択してください。
