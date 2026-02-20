import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader, Dataset
from spidiff.models import spiDiffusion
from spidiff.diffusion import create_diffusion
import argparse
import numpy as np
import os
import anndata
import scanpy as sc


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]



def find_model(model_name, device=""):
    assert os.path.isfile(model_name), f'Could not find checkpoint at {model_name}'
    if device == "":
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(model_name, map_location=device)
    if "ema" in checkpoint:
        checkpoint = checkpoint["ema"]
    return checkpoint


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = args.device

    model = spiDiffusion(
        input_size=args.input_prot_size,
        depth= args.DiT_num_blocks,
        hidden_size=args.hidden_size, 
        num_heads=args.num_heads, 
        label_size=args.cond_size,

        prot_emb_type=args.prot_emb_type,
        prot_emb=args.prot_ebd
    )   
    
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path, device=args.device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))

    loader = DataLoader(args.dataset, batch_size=args.sampling_batch_size, shuffle=False, num_workers=8, pin_memory=True)
    all_samples = None
    first_batch = True
    i = 0
    for _, y in loader: 
        y = y.to(device) 
        z = torch.randn(y.shape[0], 1, args.input_prot_size, device=device)
        model_kwargs = dict(y=y)

        # samples = diffusion.p_sample_loop(
        #     model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        # )
        samples = diffusion.ddim_sample_loop(
            model.forward,
            z.shape,
            z,                
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device,
            eta=0.0,             
        )

        if first_batch:
            all_samples = samples.detach().cpu()
            first_batch = False
        else:
            all_samples = torch.cat((all_samples, samples.detach().cpu()), dim=0)
        print(str(i) + "/" + str(len(loader)) + " DONE")
        i += 1
    torch.save(all_samples, args.save_path + "/generated_samples_ckpt_" + args.ckpt.split("/")[-1].split(".")[0] + "_sam_" + str(args.num_sampling_steps) + "_gen_" + str(args.sample_num_per_cond) + ".pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--expr_name", type=str, default="GSE263617") 
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--ckpt", type=str, default="./results/concat_protbert/checkpoints/0010000.pt") 
    parser.add_argument("--save_path", type=str, default="./results/concat_protbert/samples") 
    
    parser.add_argument("--gene_emb_type", type=str, default="concat", choices=["scgpt", 'geneformer', 'concat'], help="Gene embedding type")
    parser.add_argument("--prot_emb_type", type=str, default="protbert", choices=["esm2", "protbert"], help="Protein embedding type")
    
    # model parameters
    parser.add_argument("--DiT_num_blocks", type=int, default=12, help="DiT depth")
    parser.add_argument("--hidden_size", type=int, default=384, help="DiT hidden dimension")
    parser.add_argument("--num_heads", type=int, default=6, help="DiT heads")

    # sampling parameter
    parser.add_argument("--sample_num_per_cond", type=int, default=20, help="Number of samples generated for each input condition")
    parser.add_argument("--num_sampling_steps", type=int, default=50, help="Sampling steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling_batch_size", type=int, default=1024, help="Batch size when sampling. Reduce if GPU memory is limited")

    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()

    adata = anndata.read_h5ad(f"{args.data_path}/{args.expr_name}.h5ad")
    
    if adata.X.max() > 100:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        print("Normalized adata with log1p.")

    adata = adata[adata.obs["train_test_split"] == "test"].copy() 
    
    print(f"Read adata with shape: {adata.shape} and obs: {adata.obsm['protein_expression'].shape}")

    # read gene embedding 
    if args.gene_emb_type == 'scgpt':
        count_ebd = adata.obsm['scgpt_embedding']  # 512

    elif args.gene_emb_type == 'geneformer':
        count_ebd = adata.obsm['geneformer']  # 256

    elif args.gene_emb_type == 'concat':
        scgpt_ebd = adata.obsm['scgpt_embedding']  # 512
        geneformer_ebd = adata.obsm['geneformer']  # 256
        count_ebd = np.concatenate([scgpt_ebd, geneformer_ebd], axis=1) # 768

    else:
        raise ValueError(f"Unsupported gene embedding type: {args.gene_emb_type}")

    # read protein embedding from "random", "gene", "esm2", "protbert"
    if args.prot_emb_type == 'esm2':
        prot_ebd = adata.uns['protein_esm2_embedding'] # from esm2 1280

    elif args.prot_emb_type == 'protbert':
        prot_ebd = adata.uns['protein_protbert_emb'] # from protbert 1024

    else:
        raise ValueError(f"Unsupported protein embedding type: {args.prot_emb_type}")

     # replace nan with zero
    if np.isnan(prot_ebd).any():
        prot_ebd = np.nan_to_num(prot_ebd, nan=0.0)
    

    args.prot_ebd = prot_ebd
    args.input_prot_size = prot_ebd.shape[0]

    args.raw_cond = torch.from_numpy(count_ebd).float() 
    args.cond_size = count_ebd.shape[1]

    args.cond = torch.zeros_like(args.raw_cond.repeat((args.sample_num_per_cond, 1)))
    print("Total number of samples to generate: ", args.cond.shape)
    for i in range(args.sample_num_per_cond):
        args.cond[i::args.sample_num_per_cond] = args.raw_cond.clone()

    # create dataset
    args.dataset = CustomDataset(args.cond, args.cond) 
    
    main(args)