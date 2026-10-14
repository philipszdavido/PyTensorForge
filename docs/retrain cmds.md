mkdir -p data/other && mv data/pretrain/fineweb-edu.jsonl data/other/
rm -rf checkpoints checkpoints-story checkpoints-test exports

python cli.py tokenize data/pretrain --output tokenizer.json --vocab-size 8192 \
    --special-token "<|system|>" --special-token "<|user|>" \
    --special-token "<|assistant|>" --special-token "<|end|>"

python cli.py train configs/pretrain.yaml --device mps

python scripts/stories_to_chat.py data/pretrain --out data/chat-story
python cli.py train configs/chat-story.yaml
python cli.py export checkpoints-story/checkpoint_latest.ptf --output exports/mygpt \
    --tokenizer tokenizer.json --dtype float16
python cli.py serve exports/mygpt --name my-gpt

Keep the Mac awake: run caffeinate -i python cli.py train configs/pretrain.yaml --device mps

caffeinate -i python cli.py resume checkpoints/checkpoint_latest.ptf \
    --config configs/pretrain.yaml --device mps

