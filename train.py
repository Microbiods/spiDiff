import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import destroy_process_group
from torch.utils.data import DataLoader, Dataset
import numpy as np
from copy import deepcopy
import argparse
import os
import random
import anndata
import scanpy as sc
import argparse
from spidiff.models import spiDiffusion
from spidiff.diffusion import create_diffusion
from spidiff.train_helper import *


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        rank: int,
        gpu_id: int,
        model_args: argparse.Namespace,
    ) -> None:
        self.rank = rank
        self.gpu_id = gpu_id
        self.train_data = train_data
        self.args = model_args
        self.best_val_pcc = -float("inf")

        self.model = model
        self.ema = deepcopy(model).to(gpu_id)
        requires_grad(self.ema, False)
        self.model = DDP(self.model.to(gpu_id), device_ids=[self.gpu_id])
        self.diffusion = create_diffusion(timestep_respacing="")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), 
                                           lr=self.args.lr, weight_decay=0)
        update_ema(self.ema, self.model.module, decay=0)
        self.args.logger.info(f"Rank {rank} - Initializing Trainer... DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

        self.train_steps=0
        self.log_steps=0
        self.running_loss=0

    def _run_batch(self, x, t, modelkwargs):
                       
        loss_dict = self.diffusion.training_losses(self.model, x, t, modelkwargs)
        loss = loss_dict["loss"].mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        update_ema(self.ema, self.model.module)

        self.running_loss += loss.item()
        self.train_steps += 1
        self.log_steps += 1
        if self.log_steps % 500 == 0:
            torch.cuda.synchronize()
            avg_loss = torch.tensor(self.running_loss / self.log_steps, device=x.device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / dist.get_world_size()
            self.args.logger.info(f"Step={self.train_steps:07d} | Training Loss: {avg_loss:.5f}")
            self.running_loss = 0
            self.log_steps = 0

        if self.train_steps % self.args.ckpt_every == 0 and self.train_steps > 0:
            if self.rank == 0:
                self._save_checkpoint()
            dist.barrier()    

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)

        for x, y in self.train_data: 
            x = x.unsqueeze(1).to(self.gpu_id)  
            y = y.to(self.gpu_id)               
            t = torch.randint(0, self.diffusion.num_timesteps, (x.size(0),), device=x.device)
            model_kwargs = dict(y=y)
            self._run_batch(x, t, model_kwargs)

    def _save_checkpoint(self):
        checkpoint = {
                      "model": self.model.module.state_dict(),
                      "ema": self.ema.state_dict(),
                      "opt": self.optimizer.state_dict()
                    }
        checkpoint_path = f"{self.args.checkpoint_dir}/{self.train_steps:07d}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.args.logger.info(f"Saved checkpoint to {checkpoint_path}")

    def train(self, max_epochs: int):
        ##
        self.model.train()
        self.ema.eval()
        ##
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.train_steps >= self.args.max_steps:
                break
                


def assemble_dataset(input_args):
   
    # read adata 
    adata = anndata.read_h5ad(f"{input_args.data_path}/{input_args.expr_name}.h5ad")
    
    if adata.X.max() > 100:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        print("Normalized adata with log1p.")

    adata = adata[adata.obs["train_test_split"] == "train"].copy() 

    input_args.logger.info(f"Read adata with shape: {adata.shape} and obs: {adata.obsm['protein_expression'].shape}")

    # read gene embedding
    if input_args.gene_emb_type == 'scgpt':
        count_ebd = adata.obsm['scgpt_embedding']  # 512
    
    elif input_args.gene_emb_type == 'geneformer':
        count_ebd = adata.obsm['geneformer']  # 256

    elif input_args.gene_emb_type == 'concat':
        scgpt_ebd = adata.obsm['scgpt_embedding']  # 512
        geneformer_ebd = adata.obsm['geneformer']  # 256
        count_ebd = np.concatenate([scgpt_ebd, geneformer_ebd], axis=1) # 768
    else:
        raise ValueError(f"Unsupported gene embedding type: {input_args.gene_emb_type}")

    prot_mtx = adata.obsm["protein_expression"].toarray()
    # normalize
    prot_mtx = np.log1p(prot_mtx)
    # z-score normalization for protein matrix?
    prot_mean = np.mean(prot_mtx, axis=0)
    prot_std = np.std(prot_mtx, axis=0) + 1e-8
    input_args.logger.info(f"Protein matrix mean: {prot_mean.mean():.4f}, std: {prot_std.mean():.4f}")
    prot_mtx = (prot_mtx - prot_mean) / prot_std  # z-score normalization


    # read protein embedding from "esm2", "protbert"
    if input_args.prot_emb_type == 'esm2':
        prot_ebd = adata.uns['protein_esm2_embedding'] 

    elif input_args.prot_emb_type == 'protbert':
        prot_ebd = adata.uns['protein_protbert_emb'] 

    else:
        raise ValueError(f"Unsupported protein embedding type: {input_args.prot_emb_type}")


    # replace nan with zero
    if np.isnan(prot_ebd).any():
        # print(f'File {file} has NaN in {emb} embedding! Replacing NaN with zero.')
        prot_ebd = np.nan_to_num(prot_ebd, nan=0.0)
    
    count_ebd = torch.from_numpy(count_ebd).float()
    count_ebd.requires_grad_(False)
    
    alldataset = CustomDataset(torch.from_numpy(prot_mtx).float(), 
                               count_ebd) 

    input_args.input_prot_size = prot_mtx.shape[1] 
    input_args.cond_size = count_ebd.shape[1]

    return alldataset, input_args, prot_ebd 

def load_train_objs(args):
    train_set, args, prot_ebd = assemble_dataset(args)
    model = spiDiffusion(
        input_size=args.input_prot_size,
        depth= args.DiT_num_blocks,
        hidden_size=args.hidden_size, 
        num_heads=args.num_heads, 
        label_size=args.cond_size,

        prot_emb_type=args.prot_emb_type,
        prot_emb=prot_ebd
    )
    args.logger.info(f"Dataset contains {len(train_set):,} cells ({args.data_path})")
    return train_set, model, args


def prepare_dataloader(args, dataset: Dataset, batch_size: int, train: bool = True) -> DataLoader:
    """
    DDP-safe DataLoader.
    - Uses DistributedSampler only when torch.distributed is initialized.
    - Shuffles via sampler for train; no shuffle for eval.
    - drop_last=True for train (recommended for DDP); False for eval.
    """
    is_distributed = dist.is_available() and dist.is_initialized()

    sampler = None
    shuffle = False  

    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=train,
            seed=getattr(args, "global_seed", 0),
            drop_last=train,  
        )
    else:
        shuffle = train  

    num_workers = int(getattr(args, "num_workers", 0))
    prefetch_factor = 4 if num_workers > 0 else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
        drop_last=train,  # drop last only for training
    )

def main(world_size: int, 
         available_gpus: list,
         input_args):
    
    # Set up DDP
    dist.init_process_group(backend="nccl", world_size=world_size)
    rank = dist.get_rank()
    device = available_gpus[rank]
    seed = input_args.global_seed * dist.get_world_size() + rank
    # print("Rank: ", rank, " | Device: ", device, " | Seed: ", seed)
    # set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # set up output folder and logger
    if rank == 0:
        # print("Rank 0 mkdir & set up logger...")
        # mkdir for logs and checkpoints
        os.makedirs(input_args.results_dir, exist_ok=True)  
        input_args.experiment_dir = f"{input_args.results_dir}/{input_args.gene_emb_type}_{input_args.prot_emb_type}"
        input_args.checkpoint_dir = f"{input_args.experiment_dir}/checkpoints"  
        os.makedirs(input_args.checkpoint_dir, exist_ok=True)
        os.makedirs(f"{input_args.experiment_dir}/samples", exist_ok=True)    
        input_args.logger = create_logger(input_args.experiment_dir)
        input_args.logger.info(f"Experiment directory created at {input_args.experiment_dir}")
    else:
        input_args.logger=create_logger(None)
    input_args.logger.info(f"Rank: {rank} | Device: {device} | Seed: {seed}")
    
    # set up training objects
    train_set, model, args = load_train_objs(input_args)
    input_args.logger.info(f"Dataset, model, and args finished loading.")
    train_data = prepare_dataloader(args, train_set, 
                                    int(args.global_batch_size // dist.get_world_size()))
    
    input_args.logger.info(f"Dataloader finished loading.")
    trainer = Trainer(model, train_data,
                      rank, int(device.split(":")[-1]), 
                      args)
    input_args.logger.info(f"Trainer finished loading.")
    input_args.logger.info(f"Starting...")
    trainer.train(args.total_epochs)
    destroy_process_group()


if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    # data related arguments
    parser.add_argument("--expr_name", type=str, default="demo") 
    parser.add_argument("--results_dir", type=str, default="./results", help="Path to save results")
    parser.add_argument("--data_path", type=str, default="./data", help="Dataset path")

    parser.add_argument("--gene_emb_type", type=str, default="concat", choices=["scgpt", 'geneformer','concat'], help="Gene embedding type")
    parser.add_argument("--prot_emb_type", type=str, default="protbert", choices=["esm2", "protbert"], help="Protein embedding type")
    parser.add_argument("--freeze_prot_emb", action='store_true', help="Whether to freeze protein embedding during training")

    # model related arguments
    parser.add_argument("--DiT_num_blocks", type=int, default=12, help="DiT depth") 
    parser.add_argument("--hidden_size", type=int, default=384, help="DiT hidden dimension")
    parser.add_argument("--num_heads", type=int, default=6, help="DiT heads")

    # training related arguments
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--total_epochs", type=int, default=100000)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--global_batch_size", type=int, default=32) 
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8, help="Number of CPUs to run the job")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--ckpt_every", type=int, default=10000, help="Number of iterations to save checkpoints.") 

    input_args = parser.parse_args()

    print("Input arguments: ", input_args)

    world_size = input_args.world_size # single machine multi-gpu training only

    available_gpus = ["cuda:"+str(i) for i in range(torch.cuda.device_count())]

    print("Available GPUs: ", available_gpus)
    main(world_size, available_gpus, input_args)
