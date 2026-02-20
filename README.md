## spiDiff — Surface Protein Imputation using Diffusion Models

This repository contains a PyTorch implementation of a diffusion-based conditional generative model (DiT-style transformer backbone) for single-cell protein expression conditioned on cell embeddings.

## Repository layout

- `train.py` — training script
- `sample.py` — sampling script
- `spidiff/` — package containing the model and diffusion code
  - `spidiff/models.py` — DiT-like model (`spiDiffusion`) and embedding utilities
  - `spidiff/train_helper.py` — helpers for logging, checkpointing, and utilities used during training
  - `spidiff/diffusion/` — diffusion implementation (Gaussian diffusion, samplers, schedulers)

## Expected dataset format

The scripts expect preprocessed single-cell data saved as an AnnData `.h5ad` file with the following fields:

- `adata.obsm['protein_expression']` — protein expression matrix (sparse)
- `adata.obsm['scgpt_embedding']` — scGPT cell embedding (512-d)
- `adata.obsm['geneformer']` Geneformer cell embedding (256-d) 
- `adata.uns['protein_esm2_embedding']` or `adata.uns['protein_protbert_emb']` — protein embeddings used as name/sequence embeddings; chosen via `--prot_emb_type` (choices: `esm2` or `protbert`).
- `adata.obs['train_test_split']` — a column with values `train` and `test` so the scripts can pick training vs test cells.

## Main parameters (high level)

- Gene embedding type: `--gene_emb_type` in `{scgpt, geneformer, concat}`
- Protein embedding type: `--prot_emb_type` in `{esm2, protbert}`
- DiT model depth / width: `--DiT_num_blocks`, `--hidden_size`, `--num_heads`
- Training: learning rate `--lr`, `--total_epochs`, `--max_steps`, `--global_batch_size`, `--world_size`

See `train.py` and `sample.py` for the full set of command-line arguments.

## Dependencies

Conda setup:

```bash
conda env create -f environment.yml
conda activate spidiff
```

## Training

On a single node with one GPU you can run:

```bash
torchrun --nnodes=1 --nproc_per_node=1 train.py \
  --expr_name demo \
  --data_path ./data \
  --gene_emb_type concat \
  --prot_emb_type protbert \
  --DiT_num_blocks 12 \
  --hidden_size 384 \
  --num_heads 6 \
  --global_batch_size 32 \
  --world_size 1
```

Notes:
- `train.py` expects an AnnData file at `./data/<expr_name>.h5ad`.
- Output and checkpoints are written under `--results_dir` (default `./results`). The script creates `results/<gene_emb>_<prot_emb>/checkpoints` and `samples/` inside the experiment folder.
- Checkpoints are saved as `{step}.pt` and include both `model` and `ema` state dicts.

## Sampling

Use `sample.py` to generate samples from a saved checkpoint. 

```bash
python sample.py \
  --expr_name demo \
  --data_path ./data \
  --ckpt ./results/concat_protbert/checkpoints/0010000.pt \
  --save_path ./results/concat_protbert/samples \
  --num_sampling_steps 50 \
  --sampling_batch_size 1024 \
  --device cuda
```

Important notes:
- `sample.py` loads the checkpoint and prefers the `ema` key (exponential moving average) if present.
- Prepare the same AnnData `.h5ad` (but `obs['train_test_split'] == 'test'` is used by the script) and ensure embeddings are present as described above.
- The sampling code uses `ddim_sample_loop` (deterministic-ish sampling); you can change this behavior in `spidiff/diffusion/` if desired.

## Output

- Checkpoints: `results/<expr>/<prot>/checkpoints/<step>.pt` (contains `model`, optionally `ema`, and optimizer state)
- Generated samples: saved as PyTorch tensors under `results/.../samples` with descriptive filenames including the checkpoint name and sampling steps


# License
This source code is licensed under the MIT license found in the `LICENSE` file
in the root directory of this source tree.
