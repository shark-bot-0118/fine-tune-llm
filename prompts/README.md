# System Prompts

このディレクトリには、チャットボットで使用するシステムプロンプトが含まれています。

## 利用可能なプロンプト

### default.txt
基本的なAIアシスタント用のプロンプト。汎用的な質問応答に適しています。

### assistant.txt  
優秀なAIアシスタント用のプロンプト。より専門的で詳細な回答を提供します。

### coding_assistant.txt
プログラミング専門のアシスタント用プロンプト。コーディング、デバッグ、技術的な問題解決に特化しています。

### creative_writer.txt
創作活動専門のプロンプト。小説、詩、エッセイなどの創作をサポートします。

### tutor.txt
教育専門のプロンプト。学習者に合わせた丁寧な指導を行います。

## 使用方法

```bash
# デフォルトプロンプトを使用
python chat_with_lora.py --model google/gemma-3-1b-it --adapter ./adapter

# カスタムプロンプトを指定
python chat_with_lora.py --model google/gemma-3-1b-it --adapter ./adapter --system-prompt ./prompts/coding_assistant.txt
```

## カスタムプロンプトの作成

新しいプロンプトを作成する場合は、以下の点を考慮してください：

1. **明確な役割定義**: AIの役割と専門分野を明確に記述
2. **行動指針**: 回答時に従うべき原則やガイドライン
3. **具体的な例**: 必要に応じて回答スタイルの例を含める
4. **制約事項**: 避けるべき内容や行動の制限

プロンプトファイルはUTF-8エンコーディングのテキストファイルとして保存してください。