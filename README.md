# DECO-HumanLM

Code for **Mind the State-to-Response Gap: Structured Reward Decomposition for Personalized
User Simulation**.

## Models

- [0.6B DECO](https://huggingface.co/Leoputan/deco-humanlm-0.6b-deco)
- [0.6B PDN](https://huggingface.co/Leoputan/deco-humanlm-0.6b-pdn)
- [0.6B DECO-P](https://huggingface.co/Leoputan/deco-humanlm-0.6b-deco-p)
- [8B DECO LoRA](https://huggingface.co/Leoputan/deco-humanlm-8b-deco-lora)
- [8B PDN LoRA](https://huggingface.co/Leoputan/deco-humanlm-8b-pdn-lora)

DECO scores a generated response on six user-state dimensions—stance, emotion, belief, value,
goal, and communication—and aggregates them with equal weight. PDN standardizes each dimension
within the rollout group before aggregation.

## Structure

- `scripts/train.sh`: configurable GRPO training for 0.6B and 8B models
- `training/`: HumanLM reward metric and integration patch
- `evaluation/judge.py`: six state scores plus overall response alignment
- `evaluation/embedding.py`: embedding cosine similarity
- `configs/`: dimension definitions and training examples
- `src/deco_humanlm/rewards.py`: standalone DECO and PDN implementation

## Setup

```bash
pip install -e '.[generation,embedding,dev]'

git clone https://github.com/ehejin/verl-recipe-humanlm.git
cd verl-recipe-humanlm
git checkout 6a7dbd3f143fc0a9af599ed7a458fc503341f846
cd ..

python scripts/install_humanlm_overlay.py ./verl-recipe-humanlm
```

The training code extends the
[HumanLM recipe](https://github.com/ehejin/verl-recipe-humanlm/tree/6a7dbd3f143fc0a9af599ed7a458fc503341f846/humanlm).
Copy the patched `humanlm` directory to `<VERL_DIR>/recipe/humanlm` in the VERL checkout.

Set the API key required by the selected LiteLLM judge, such as `OPENAI_API_KEY` or
`ANTHROPIC_API_KEY`.

## Training

Create a local configuration from the 0.6B or 8B example and replace its paths and judge model:

```bash
cp configs/train_0.6b.env.example local-0.6b.env
bash scripts/train.sh local-0.6b.env deco

cp configs/train_8b.env.example local-8b.env
bash scripts/train.sh local-8b.env pdn
```

The 0.6B example uses full-parameter training. The 8B example uses LoRA with rank 16 and alpha
32. All training settings can be changed in the configuration or with trailing Hydra overrides.

## Judge evaluation

```bash
python evaluation/judge.py \
  --input outputs/generations.jsonl \
  --output outputs/judge_scores.jsonl \
  --model provider/model-name \
  --api-key-env PROVIDER_API_KEY \
  --limit 100
```

Each response receives scores for stance, emotion, belief, value, goal, communication, and
overall response alignment. Any LiteLLM-supported judge can be selected with `--model`.

## Embedding evaluation

```bash
python evaluation/embedding.py \
  --input outputs/generations.jsonl \
  --output outputs/embedding_scores.jsonl \
  --model /path/or/huggingface-id \
  --limit 100
```

## Tests

```bash
python -m unittest discover -s tests
python scripts/install_humanlm_overlay.py /path/to/verl-recipe-humanlm --check
```

## Citation

```bibtex
@article{tan2026deco,
  title  = {Mind the State-to-Response Gap: Structured Reward Decomposition for Personalized User Simulation},
  author = {Tan, Leo},
  year   = {2026}
}
```
