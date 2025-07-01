import json

line_number = 0
with open("../data/fine_tune_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line_number += 1
        try:
            json.loads(line)
            # print(f"Line {line_number}: OK")
        except json.JSONDecodeError as e:
            print(f"Error in line {line_number}: {e}")
            print(f"Problematic line content: {line.strip()}")
            break # 最初のエラーで停止

print(f"Completed checking {line_number} lines")