# LoRA Fine-tuning Project

このプロジェクトは、LoRA（Low-Rank Adaptation）を使用してLLM（Large Language Model）をファインチューニングし、GGUF形式での高速推論も可能にする包括的なプロジェクトです。

## 事前知識 -LoRAとは-

- [LoRA Guide](LORA_GUIDE.md) - 詳細なLoRAガイド

## 制約と注意事項

- [Development Note](DEVELOPMENT_NOTE.md) - 開発時の知見・制約などを記載  
    ※はじめにこちらに目を通しておくことをお勧めします

## プロジェクト構成

```
fine-tune-llm/
├── data/                     # 訓練データ
│   └── fine_tune_data.jsonl
├── models/                   # 訓練済みモデル
├── output/                   # 出力ファイル
├── gguf/                     # GGUF形式のモデル
├── history/                  # チャット履歴（自動生成）
├── prompts/                  # システムプロンプト
│   ├── default.txt
│   ├── coding_assistant.txt
│   ├── creative_writer.txt
│   └── tutor.txt
├── scripts/                  # 実行スクリプト
│   ├── train_lora.py        # LoRA訓練
│   ├── chat_with_lora.py    # チャットインターフェース（PyTorch）
│   ├── merge_lora_adapter.py # アダプターマージ
│   ├── convert_to_gguf.py   # GGUF変換
│   └── jsonl_checker.py     # データ検証
├── llama.cpp/               # GGUF変換・実行ツール
├── requirements.txt         # Python依存関係
├── requirements-minimal.txt # 最小限の依存関係
└── README.md
```

## 機能

- LoRAを使用した効率的なファインチューニング
- 対話型チャットインターフェース（PyTorchベース・GGUFベース）
- モデルのマージとエクスポート機能
- GGUF形式への変換と量子化
- llama.cppによる高速推論
- カスタマイズ可能なシステムプロンプト
- 自動的な会話履歴の保存と管理
- tokenizer.model自動生成によるGGUF変換サポート

## スクリプト詳細

### 主要スクリプト

#### 1. `train_lora.py` - LoRAファインチューニング
**処理概要**:
- HuggingFaceモデルの読み込みとLoRA設定の適用
- データセットの前処理とトークナイゼーション
- LoRAアダプターの訓練実行
- 訓練済みアダプターとトークナイザーの保存
- tokenizer.model自動生成（GGUF変換対応）

**主要機能**:
- 複数のLoRAパラメータ設定（rank, alpha, dropout）
- MPS/CUDA/CPU自動検出とデバイス最適化
- データセット検証とエラーハンドリング
- 訓練情報のJSON出力

#### 2. `chat_with_lora.py` - PyTorchベースチャット
**処理概要**:
- ベースモデルとLoRAアダプターの読み込み
- 対話型チャットインターフェースの提供
- チャット履歴の自動保存・管理
- システムプロンプトのカスタマイズ対応

**主要機能**:
- リアルタイム会話生成
- 会話履歴の自動JSON保存（history/ディレクトリ）
- チャット中のコマンド（clear, history, save, exit）
- EOS token設定の最適化

#### 3. `merge_lora_adapter.py` - アダプターマージ
**処理概要**:
- LoRAアダプターをベースモデルに統合
- マージされたモデルの単一ファイル出力
- tokenizer.model自動取得・保存
- マージ情報のメタデータ保存

**主要機能**:
- LoRA重みの完全統合（merge_and_unload）
- Safe serialization対応
- 元のモデル情報の保持
- マージ処理の詳細ログ出力

#### 4. `convert_to_gguf.py` - GGUF形式変換
**処理概要**:
- HuggingFaceモデルのGGUF形式変換
- llama.cpp変換スクリプトのラッパー
- tokenizer.model自動生成・取得
- 量子化処理の実行

**主要機能**:
- マージモデル・LoRAアダプター両対応
- 複数量子化形式サポート（Q4_K_M, Q5_K_M等）
- tokenizer.model不足時の自動補完
- llama.cpp依存関係の自動検証

### ユーティリティスクリプト

#### 5. `jsonl_checker.py` - データ検証
**処理概要**:
- 訓練データの形式検証
- JSONLファイルの構造チェック
- messagesフィールドの妥当性確認
- エラー詳細レポート生成

**主要機能**:
- Chat template形式の検証
- 文字エンコーディング確認
- データ統計情報の出力
- 修復提案の表示

## 必要な環境

- Python 3.9+
- PyTorch 2.1+
- Transformers 4.46.0+（Gemma 3サポート）
- PEFT 0.13.0+
- llama.cpp（GGUF推論用）
- その他の依存関係（requirements.txtを参照）

## インストール

### 1. リポジトリをクローン
```bash
git clone https://github.com/shark-bot-0118/fine-tune-llm.git
cd fine-tune-llm
```

### 2. Python依存関係をインストール
```bash
# 仮想環境の作成（推奨）
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# または .venv\Scripts\activate  # Windows

# 最小限のインストール
pip install -r requirements-minimal.txt

# 完全なインストール（開発ツール含む）
pip install -r requirements.txt
```

### 3. Hugging Faceモデルの準備

#### Hugging Faceとは
[Hugging Face](https://huggingface.co/)は、機械学習モデルとデータセットの共有プラットフォームです。数万の事前訓練済みモデルが公開されており、研究・商用利用が可能です。

**主な特徴**:
- 🤗 **豊富なモデル**: GPT、BERT、LLaMA、Gemmaなど最新モデル
- 🔄 **簡単ダウンロード**: コマンド一行でモデル取得
- 📄 **詳細な文書**: モデルカード、使用方法、ライセンス情報
- 🔐 **アクセス制御**: 一部モデルはアカウント認証が必要

#### モデルのインストール方法

**自動ダウンロード（推奨）**:
```bash
# スクリプト実行時に自動ダウンロード
python scripts/train_lora.py --model google/gemma-3-1b-it
# 初回実行時にHugging Face Hubから自動的にモデルがダウンロードされます
```

**手動事前ダウンロード**:
```bash
# Hugging Face CLIを使用
pip install huggingface_hub
huggingface-cli download google/gemma-3-1b-it

# Pythonスクリプトで事前ダウンロード
python -c "
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('google/gemma-3-1b-it')
tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-1b-it')
print('Model downloaded successfully!')
"
```

#### アクセス制限付きモデルの取得

一部のモデル（Meta LLaMA、Google Gemmaなど）は利用申請が必要です：

1. **Hugging Faceアカウント作成**: https://huggingface.co/join
2. **モデルページで利用申請**: 該当モデルページで"Request access"
3. **トークンの取得**: https://huggingface.co/settings/tokens
4. **認証の設定**:

```bash
# 方法1: huggingface-cliでログイン
huggingface-cli login

# 方法2: 環境変数で設定
export HUGGINGFACE_HUB_TOKEN="your_token_here"

# 方法3: .env ファイルに記載
echo "HUGGINGFACE_HUB_TOKEN=your_token_here" >> .env
```

#### 推奨モデル

| モデル名 | サイズ | 特徴 | アクセス |
|---------|-------|------|---------|
| `google/gemma-3-1b-it` | 1.3B | 軽量、日本語対応 | 申請必要 |
| `google/gemma-3-4b-it` | 4B | 高性能、マルチモーダル対応 | 申請必要 |
| `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | 8B | 推論特化、SOTA性能 | 自由 |
| `Qwen/Qwen3-4B` | 4B | 最新版、119言語対応 | 自由 |

#### ダウンロード場所

ダウンロードされたモデルは以下に保存されます：
```bash
# デフォルトキャッシュ場所
~/.cache/huggingface/hub/  # Linux/macOS
C:\Users\{username}\.cache\huggingface\hub\  # Windows

# カスタムキャッシュディレクトリ
export HF_HOME="/path/to/custom/cache"
```

### 4. llama.cppのセットアップ（GGUF変換・実行用）

llama.cppは高速なGGUF推論のために必要です：

```bash
# llama.cppのクローン（既に含まれている場合はスキップ）
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# CMakeを使用したビルド（推奨）
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)  # Linux
make -j$(sysctl -n hw.ncpu)  # macOS

# Makefileを使用したビルド（非推奨）
# make  # CMakeビルドが失敗した場合のみ
```

#### Apple Silicon（M1/M2/M3）での最適化
```bash
# Metal（GPU）サポートを有効化
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
```

#### NVIDIA GPUでの最適化
```bash
# CUDA サポートを有効化
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
```

## 使用方法

### 1. データの準備

訓練データを`data/fine_tune_data.jsonl`形式で準備します。各行はJSON形式で以下の構造を持ちます：

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### 2. LoRAファインチューニング

```bash
# 基本的な訓練
python scripts/train_lora.py --model google/gemma-3-1b-it

# カスタム設定での訓練
python scripts/train_lora.py \
  --model google/gemma-3-1b-it \
  --epochs 5 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --lora-r 32

# カスタムデータでの訓練
python scripts/train_lora.py \
  --model google/gemma-3-1b-it \
  --data ./my_data.jsonl \
  --output ./my_model
```

### 3. PyTorchベースチャットインターフェース

```bash
# デフォルトプロンプトでチャット
python scripts/chat_with_lora.py \
  --model google/gemma-3-1b-it \
  --adapter models/trained_model/final_model

# プログラミングアシスタントモード
python scripts/chat_with_lora.py \
  --model google/gemma-3-1b-it \
  --adapter models/trained_model/final_model \
  --system-prompt prompts/coding_assistant.txt

# 会話履歴を自動保存（タイムスタンプ付きファイル名）
python scripts/chat_with_lora.py \
  --model google/gemma-3-1b-it \
  --adapter models/trained_model/final_model \
  --save-history

# 会話履歴をカスタムファイル名で保存
python scripts/chat_with_lora.py \
  --model google/gemma-3-1b-it \
  --adapter models/trained_model/final_model \
  --save-history my_chat.json
```

#### チャット中のコマンド
- `exit`, `quit`: 終了（自動で履歴を保存）
- `clear`: 会話履歴をクリア
- `history`: 会話履歴を表示
- `save`: 手動で履歴を保存

#### 会話履歴機能
- `history/`ディレクトリが自動で作成されます
- 終了時に自動的に履歴が保存されます
- JSON形式でタイムスタンプやメタデータも保持
- カスタムファイル名を指定可能（history/内に保存）

### 4. モデルのマージ

```bash
# LoRAアダプターをベースモデルにマージ
python scripts/merge_lora_adapter.py \
  --model google/gemma-3-1b-it \
  --lora models/trained_model/final_model \
  --output output/merged_model

# 安全なシリアル化を使用
python scripts/merge_lora_adapter.py \
  --model google/gemma-3-1b-it \
  --lora models/trained_model/final_model \
  --output output/merged_model \
  --safe-serialization
```

### 5. GGUF形式への変換

#### マージされたモデルをGGUF化（推奨）
```bash
# 基本的な変換
python scripts/convert_to_gguf.py \
  --model output/merged_model \
  --output gguf/gemma-3-1b-it-merged

# 量子化付きで変換
python scripts/convert_to_gguf.py \
  --model output/merged_model \
  --output gguf/gemma-3-1b-it-merged \
  --quantize Q4_K_M
```

#### LoRAアダプターを直接GGUF化
```bash
python scripts/convert_to_gguf.py \
  --lora models/trained_model/final_model \
  --base-model google/gemma-3-1b-it \
  --output gguf/lora_adapter.gguf
```

#### 既存GGUFファイルの量子化
```bash
python scripts/convert_to_gguf.py \
  --quantize-only gguf/model.gguf gguf/model_q4.gguf \
  --quant-type Q4_K_M
```

#### llama-cli直接実行（最高速）
```bash
cd llama.cpp
build/bin/llama-cli \
  -m ../gguf/gemma-3-1b-it-merged/Merged_Model-1.3B-F16.gguf \
  -p "あなたは親切なAIアシスタントです。\\n\\nUser: こんにちは\\n\\nAssistant:" \
  -n 100 --temp 0.7 --repeat-penalty 1.1
```

## システムプロンプト

プロジェクトには用途別のシステムプロンプトが用意されています：

- `prompts/default.txt`: 汎用アシスタント
- `prompts/coding_assistant.txt`: プログラミング専門
- `prompts/creative_writer.txt`: 創作活動専門
- `prompts/tutor.txt`: 教育・学習専門

カスタムプロンプトも作成可能です。

## 設定

### 環境変数（オプション）
- `BASE_MODEL_NAME`: デフォルトのベースモデル名
- `PREFIX`: 出力ファイルのプレフィックス

### コマンドライン引数
各スクリプトは`--help`オプションで詳細なヘルプを表示します：

```bash
python scripts/train_lora.py --help
python scripts/chat_with_lora.py --help
python scripts/run_gguf_model.py --help
python scripts/convert_to_gguf.py --help
```

## トラブルシューティング

### よくある問題

1. **メモリ不足エラー**
   - バッチサイズを小さくする: `--batch-size 1`
   - 勾配蓄積ステップを増やす: `--gradient-accumulation-steps 16`

2. **Gemma 3モデルのエラー**
   - Transformersライブラリを更新: `pip install transformers>=4.46.0`

3. **GGUF変換エラー**
   - llama.cppがビルドされているか確認
   - tokenizer.modelが自動生成されない場合は手動でHuggingFaceからダウンロード

4. **llama.cppビルドエラー**
   - CMakeを使用: `cmake .. && make`
   - 必要に応じてコンパイラを更新

5. **チャットが応答しない**
   - EOS token設定を確認
   - 温度パラメータを調整: `--temperature 0.7`
   - リピートペナルティを調整: `--repeat-penalty 1.1`

### パフォーマンス最適化

- **Apple Silicon**: MPSデバイスが自動で使用されます
- **NVIDIA GPU**: CUDAが自動で検出されます
- **量子化**: Q4_K_MまたはQ5_K_Mがバランスが良いです
- **スレッド数**: CPUコア数に合わせて`--threads`を調整

## パフォーマンス比較

| 実行方法 | 速度 | メモリ使用量 | 推奨用途 |
|---------|------|-------------|----------|
| PyTorch（chat_with_lora.py） | 遅い | 高い | 開発・デバッグ |
| GGUF（run_gguf_model.py） | 速い | 中程度 | 一般的な使用 |
| llama-cli直接実行 | 最速 | 低い | 本番環境 |

## ライセンス

MIT License

## 貢献

プルリクエストやイシューの報告を歓迎します。

## 参考リンク

- [LoRA論文](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [GGUF Format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Gemma 3 Model](https://huggingface.co/google/gemma-3-4b-it)
- [DeepSeek R1 Model](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B)
- [Qwen3 Model Collection](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f)
