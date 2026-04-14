# Projet Observatoire

## Dataset Download

Download the WIESP2022-NER dataset from Hugging Face into the local `data/raw/` directory:

```bash
mkdir -p data/raw/WIESP2022-NER-dataset

curl -L https://huggingface.co/datasets/adsabs/WIESP2022-NER/resolve/main/WIESP2022-NER-TRAINING.jsonl -o data/raw/WIESP2022-NER-dataset/WIESP2022-NER-TRAINING.jsonl
curl -L https://huggingface.co/datasets/adsabs/WIESP2022-NER/resolve/main/WIESP2022-NER-VALIDATION.jsonl -o data/raw/WIESP2022-NER-dataset/WIESP2022-NER-VALIDATION.jsonl
curl -L https://huggingface.co/datasets/adsabs/WIESP2022-NER/resolve/main/WIESP2022-NER-TESTING.jsonl -o data/raw/WIESP2022-NER-dataset/WIESP2022-NER-TESTING.jsonl
```
