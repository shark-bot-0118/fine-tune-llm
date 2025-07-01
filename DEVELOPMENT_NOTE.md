# Development Note

このファイルは、LoRAファインチューニングプロジェクトの開発過程で得られた知見、制約、課題などを記録します。

## 開発環境での検証結果

### 動作確認済み環境
- Python 3.9.x - 3.11.x
- PyTorch 2.1+
- CUDA 11.8+ / Apple Silicon (MPS)
- メモリ: 最低8GB RAM推奨

### 推奨モデルサイズ
- 1B-3Bパラメータ: 一般的な開発環境で動作
- 7B+パラメータ: GPU必須、16GB+ VRAM推奨

## 制約・注意点

### メモリ制約
- **バッチサイズ**: デフォルトは4だが、メモリ不足時は1-2に調整必要
- **勾配蓄積**: メモリ不足時は`gradient_accumulation_steps`を16-32に増加
- **量子化**: 実運用では4bit量子化(Q4_K_M)が速度・品質のバランス良好

### モデル固有の制約
- **Gemma 3**: transformers>=4.46.0必須
- **tokenizer.model**: 自動生成されない場合は手動ダウンロード必要
- **EOS token**: モデルによって設定が異なる

### GGUF変換の制約
- **llama.cpp**: 必ずCMakeでビルド推奨
- **量子化**: F16→Q4への変換で2-3GB一時的にメモリ使用
- **Metal/CUDA**: 環境に応じて最適化フラグが必要

## 開発中に発見した課題

### パフォーマンス
- PyTorchベース: 開発・デバッグには良いが推論速度は遅い
- GGUFベース: 推論速度は3-5倍高速だが、デバッグが困難
- メモリ使用量: GGUF化で50-70%削減可能

### 互換性
- **古いtransformers**: Gemma 3で予期しないエラー
- **Makefileビルド**: 非推奨、CMake推奨
- **Windowsビルド**: 追加の設定が必要な場合あり

## 推奨ワークフロー

### 開発フェーズ
1. 小さなデータセットでPyTorchベースで動作確認
2. LoRAアダプターの品質をチャットで確認
3. 満足したらマージ→GGUF変換

### プロダクションフェーズ
1. マージしたモデルをGGUF化
2. 適切な量子化レベルを選択(Q4_K_M推奨)
3. llama-cli直接実行で最高速度を実現

## よくあるエラーと対処法

### "Out of memory"
```bash
# バッチサイズを下げる
--batch-size 1 --gradient-accumulation-steps 16
```

### "No module named 'torch'"
```bash
pip install torch torchvision torchaudio
```

### "GGUF conversion failed"
```bash
# llama.cppを再ビルド
cd llama.cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

### "Model doesn't respond"
```bash
# EOS token設定を確認、温度パラメータを調整
--temperature 0.7 --repeat-penalty 1.1
```

## 今後の改善点

- [ ] 自動的なメモリ使用量検出とバッチサイズ調整
- [ ] より詳細なエラーハンドリング
- [ ] Windowsでのビルド自動化
- [ ] GPU使用率モニタリング機能
- [ ] より多くのモデル形式への対応

## 参考情報

### 有用なリソース
- LoRAアダプターのハイパーパラメータ調整は試行錯誤が必要
- システムプロンプトの効果は大きい、用途別に最適化推奨
- 会話履歴機能は長期チャットで有用

### デバッグのコツ
- まずは小さなモデル(1B)で動作確認
- ログレベルを上げて詳細な情報を確認
- メモリ使用量を常にモニタリング 
