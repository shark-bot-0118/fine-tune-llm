# LoRA Fine-tuning Project

このプロジェクトは、LoRA（Low-Rank Adaptation）を使用してLLM（Large Language Model）をファインチューニングし、チューニングしたモデルでテストを実施可能です。  
チューニングしたモデルGGUF形式へ量子化し量子化したモデルの利用も可能です。

## プロジェクト構成

```
fine_tune_llm/
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
│   ├── chat_with_lora.py    # チャットインターフェース
│   ├── merge_lora_adapter.py # アダプターマージ
│   ├── convert_to_gguf.py   # GGUF変換
│   └── run_llm.py           # ベースモデル実行
├── llama.cpp/               # GGUF変換・実行ツール
├── requirements.txt         # Python依存関係
└── README.md
```

## 機能

- LoRAを使用した効率的なファインチューニング
- 対話型チャットインターフェース
- モデルのマージとエクスポート機能
- GGUF形式への変換と量子化
- カスタマイズ可能なシステムプロンプト
- 自動的な会話履歴の保存と管理

## 必要な環境

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.36.0+
- PEFT 0.7.0+
- その他の依存関係（requirements.txtを参照）

## インストール

1. リポジトリをクローン
```bash
git clone https://github.com/shark-bot-0118/fine-tune-llm.git
cd fine_tune_llm
```

2. 依存関係をインストール
```bash
# 最小限のインストール
pip install -r requirements-minimal.txt

# 完全なインストール（開発ツール含む）
pip install -r requirements.txt
```

3. llama.cppのセットアップ（GGUF変換用）
```bash
cd llama.cpp
make  # またはCMakeを使用
```

## 使用方法

### 1. データの準備

訓練データを`data/fine_tune_data.jsonl`形式で準備します。各行はJSON形式で以下の構造を持ちます：

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### 2. ファインチューニング

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

### 3. チャットインターフェース

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

#### HuggingFaceモデルをGGUF化
```bash
# 基本的な変換
python scripts/convert_to_gguf.py \
  --model google/gemma-3-1b-it \
  --output gguf/gemma-3-1b-it

# 量子化付きで変換
python scripts/convert_to_gguf.py \
  --model output/merged_model \
  --output gguf/gemma-3-1b-it \
  --quantize Q4_K_M
```

#### LoRAアダプターを直接GGUF化
```bash
python scripts/convert_to_gguf.py \
  --lora models/trained_model/final_model \
  --base-model google/gemma-3-1b-it \
  --output gguf/lora_adapter.gguf
```

## システムプロンプト

プロジェクトには用途別のシステムプロンプトが用意されています：

- `prompts/default.txt`: 汎用アシスタント
- `prompts/coding_assistant.txt`: プログラミング専門
- `prompts/creative_writer.txt`: 創作活動専門
- `prompts/tutor.txt`: 教育・学習専門

カスタムプロンプトも作成可能です。詳細は`prompts/README.md`を参照してください。

## ライセンス

MIT License
