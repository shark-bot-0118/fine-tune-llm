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
│   ├── chat_with_lora.py    # チャットインターフェース
│   ├── merge_lora_adapter.py # アダプターマージ
│   ├── convert_to_gguf.py   # GGUF変換
│   └── run_llm.py           # ベースモデル実行
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

### 3. llama.cppのセットアップ（GGUF変換・実行用）

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
- [Gemma 3 Model](https://huggingface.co/google/gemma-3-1b-it)
