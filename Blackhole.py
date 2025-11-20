import time
from glob import glob
from tqdm import tqdm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch import nn
import torchvision.utils as tvu
from sklearn import svm
import pickle
import torch.optim as optim
import json
import joblib
import torch.nn.functional as F

from models.ddpm.diffusion import DDPM
from models.improved_ddpm.script_util import i_DDPM
from utils.text_dic import SRC_TRG_TXT_DIC
from utils.diffusion_utils import get_beta_schedule, denoising_step
from datasets.data_utils import get_dataset, get_dataloader
from configs.paths_config import DATASET_PATHS, MODEL_PATHS, HYBRID_MODEL_PATHS, HYBRID_CONFIG
from datasets.imagenet_dic import IMAGENET_DIC
from utils.align_utils import run_alignment
#from utils.distance_utils import euclidean_distance, cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.svm import LinearSVC
from torchvision.models import vgg16

class VGGPerceptual(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = vgg16(weights="IMAGENET1K_FEATURES").features.eval()
        # light slices for speed; adjust if you want stronger perceptual
        self.slices = torch.nn.ModuleList([
            vgg[:4].eval(),   # relu1_2
            vgg[4:9].eval(),  # relu2_2
            vgg[9:16].eval(), # relu3_3
        ])
        for p in self.parameters(): p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
        self.to(device)

    def forward(self, x, y):
        # expects [-1,1] -> normalize to ImageNet stats
        def norm(z):
            z = (z.clamp(-1,1) + 1)/2.0
            return (z - self.mean)/self.std
        x, y = norm(x), norm(y)
        loss = 0.0
        for s in self.slices:
            x, y = s(x), s(y)
            loss = loss + F.l1_loss(x, y)
        return loss
def _flat(x): return x.view(x.size(0), -1)


def _cos(a, b): return F.cosine_similarity(a, b, dim=-1)


def _sample_neighbors(h_r, M, sigma_h, alpha_max):
    """L3–L6: neighbors around h_r → [M, 512, 8, 8]"""
    dev = h_r.device
    outs = []
    for _ in range(M):
        h_rand = h_r + sigma_h * torch.randn_like(h_r)
        d = _flat(h_rand - h_r)
        d = d / (d.norm(p=2, dim=1, keepdim=True) + 1e-8)
        alpha = torch.empty(1, device=dev).uniform_(0.0, alpha_max)
        outs.append(h_r + d.view_as(h_r) * alpha.view(1,1,1,1))
    return torch.cat(outs, dim=0)


def _decode_h_batch_with_denoiser(self, x_lat_1b, H_batch, seq, seq_next):
    """Run reverse denoising from x_lat while injecting edit_h; returns images in [-1,1]."""
    B = H_batch.size(0)
    x = x_lat_1b.repeat(B,1,1,1).contiguous()
    h = H_batch.contiguous()
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t      = torch.full((B,), i, device=self.device)
        t_next = torch.full((B,), j, device=self.device)
        x, h = denoising_step(
            x, t=t, t_next=t_next, models=self.model, logvars=self.logvar,
            sampling_type=self.args.sample_type, b=self.betas, eta=1.0,
            learn_sigma=self.learn_sigma, ratio=self.args.model_ratio,
            hybrid=self.args.hybrid_noise, hybrid_config=HYBRID_CONFIG,
            edit_h=h
        )
    return x.clamp(-1, 1)

def _fit_svm_boundary(H_neighbors, y_np, C):
    X = _flat(H_neighbors).cpu().numpy()
    svm = LinearSVC(C=C, max_iter=5000)
    svm.fit(X, y_np.astype(np.int32))
    w = torch.from_numpy(svm.coef_.astype(np.float32)).to(H_neighbors.device)  # [1, Dh]
    b0 = float(svm.intercept_[0])
    w  = F.normalize(w, dim=1)
    return w, b0, svm


def _orient_to_dissimilar(h_r, w, b0):
    margin = (_flat(h_r) @ w.t()).item() + b0
    return (-w if margin > 0 else w)  # dq


def _hyperplane_distance(H_flat, w, b0):
    # H_flat: [M, Dh], w: [1, Dh]
    return torch.abs(H_flat @ w.t() + b0) / (w.norm(p=2) + 1e-8)



class BIAUnlearn(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
                             (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))

        if self.args.edit_attr is None:
            self.src_txts = self.args.src_txts
            self.trg_txts = self.args.trg_txts
        else:
            self.src_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][0]   //identity attribute added in Utils file that attribute
            self.trg_txts = SRC_TRG_TXT_DIC[self.args.edit_attr][1]


 
    def blackhole_formation(self):
        print(self.args.exp)

        # ----------- Model -----------#
        

        if self.config.data.dataset in ["CelebA_HQ"]:
            model = DDPM(self.config)
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                if self.config.data.dataset == "CelebA_HQ":
                    init_ckpt = torch.load(local_path, map_location=self.device)  # Updated to load from local path
                else:
                    init_ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
                
            learn_sigma = False
            print("Original diffusion Model loaded.")
        elif self.config.data.dataset in ["FFHQ"]:
            model = i_DDPM(self.config.data.dataset)
            if self.args.model_path:
                init_ckpt = torch.load(self.args.model_path)
            else:
                init_ckpt = torch.load(MODEL_PATHS[self.config.data.dataset])
            learn_sigma = True
            print("Improved diffusion Model loaded.")
        else:
            print('Not implemented dataset')
            raise ValueError
        model.load_state_dict(init_ckpt)
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.eval()
        
        
        # ----------- Precompute Latents -----------#
        print("Prepare identity latent")
        seq_inv = np.linspace(0, 1, self.args.n_inv_step) * self.args.t_0
        seq_inv = [int(s) for s in list(seq_inv)]
        seq_inv_next = [-1] + list(seq_inv[:-1])


        n = self.args.bs_train
        img_lat_pairs_dic = {}
        for mode in ['train', 'test']:
            img_lat_pairs = []
            pairs_path = os.path.join('precomputed/',
                                      f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            print(pairs_path)
            if os.path.exists(pairs_path):
                print(f'{mode} pairs exists')
                img_lat_pairs_dic[mode] = torch.load(pairs_path)
                for step, (x0, x_id, x_lat, mid_h, label) in enumerate(img_lat_pairs_dic[mode]):
                    tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))
                    tvu.save_image((x_id + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                  f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                    if step == self.args.n_precomp_img - 1:
                        break
                continue
            else:
                train_dataset, test_dataset = get_dataset(self.config.data.dataset, DATASET_PATHS, self.config)
                loader_dic = get_dataloader(train_dataset, test_dataset, bs_train=self.args.bs_train,
                                            num_workers=self.config.data.num_workers)
                loader = loader_dic[mode]

            for step, (img, label) in enumerate(loader):
            # for step, img in enumerate(loader):

                x0 = img.to(self.config.device)
                tvu.save_image((x0 + 1) * 0.5, os.path.join(self.args.image_folder, f'{mode}_{step}_0_orig.png'))

                x = x0.clone()
                model.eval()
                label = label.to(self.config.device)

                # print("check x and label:", x.size(), label)



                with torch.no_grad():
                    with tqdm(total=len(seq_inv), desc=f"Inversion process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_prev = (torch.ones(n) * j).to(self.device)

                            x, mid_h_g = denoising_step(x, t=t, t_next=t_prev, models=model,
                                               logvars=self.logvar,
                                               sampling_type='ddim',
                                               b=self.betas,
                                               eta=0,
                                               learn_sigma=learn_sigma)

                            progress_bar.update(1)
                    x_lat = x.clone()
                    tvu.save_image((x_lat + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                   f'{mode}_{step}_1_lat_ninv{self.args.n_inv_step}.png'))

                    with tqdm(total=len(seq_inv), desc=f"Generative process {mode} {step}") as progress_bar:
                        for it, (i, j) in enumerate(zip(reversed((seq_inv)), reversed((seq_inv_next)))):
                            t = (torch.ones(n) * i).to(self.device)
                            t_next = (torch.ones(n) * j).to(self.device)

                            x, _ = denoising_step(x, t=t, t_next=t_next, models=model,
                                               logvars=self.logvar,
                                               sampling_type=self.args.sample_type,
                                               b=self.betas,
                                               learn_sigma=learn_sigma,
                                               # edit_h = mid_h,
                                               )

                            progress_bar.update(1)

                    img_lat_pairs.append([x0, x.detach().clone(), x_lat.detach().clone(), mid_h_g.detach().clone(), label])
                    # img_lat_pairs.append([x0, x.detach().clone(), x_lat.detach().clone(), mid_h_g.detach().clone()])
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'{mode}_{step}_1_rec_ninv{self.args.n_inv_step}.png'))
                if step == self.args.n_precomp_img - 1:
                    break

            img_lat_pairs_dic[mode] = img_lat_pairs
            pairs_path = os.path.join('precomputed/',
                                      f'{self.config.data.category}_{mode}_t{self.args.t_0}_nim{self.args.n_precomp_img}_ninv{self.args.n_inv_step}_pairs.pth')
            torch.save(img_lat_pairs, pairs_path)


    def blackhole_boundary_search(self, img_lat_pairs_dic):
        print("Start boundary search (H-only, Black-Hole / BIA Step-1)")
        print(f"Sampling type: {self.args.sample_type.upper()} with eta {self.args.eta}")

        # schedules
        if self.args.n_train_step != 0:
            seq_train = [int(s) for s in (np.linspace(0, 1, self.args.n_train_step) * self.args.t_0)]
            print('Uniform skip type (train)')
        else:
            seq_train = list(range(self.args.t_0)); print('No skip (train)')
        seq_train_next = [-1] + list(seq_train[:-1])

        if self.args.n_test_step != 0:
            seq_test = [int(s) for s in (np.linspace(0, 1, self.args.n_test_step) * self.args.t_0)]
            print('Uniform skip type (test)')
        else:
            seq_test = list(range(self.args.t_0)); print('No skip (test)')
        seq_test_next = [-1] + list(seq_test[:-1])

        # BIA params (put these into self.args if you want)
        M_neighbors = getattr(self.args, "bh_num_samples", 128)
        sigma_h     = getattr(self.args, "bh_sigma", 0.15)
        alpha_max   = getattr(self.args, "bh_alpha_max", 0.35)
        tau         = getattr(self.args, "bh_tau", 0.35)      # ArcFace sim threshold
        svm_C       = getattr(self.args, "bh_svm_C", 1.0)
        bs_label    = getattr(self.args, "bh_batch_size", 16)

        os.makedirs('boundary', exist_ok=True)
        exp_id = os.path.split(self.args.exp)[-1]

        # we will compute a local boundary for each precomputed sample
        for mode in ['train', 'test']:
            if mode not in img_lat_pairs_dic:
                continue
            pairs = img_lat_pairs_dic[mode]
            print(f"Found {len(pairs)} precomputed pairs in '{mode}'")

            for idx, (x0, x_id, x_lat, h_lat, label) in enumerate(pairs):
                x0    = x0.to(self.device).contiguous()      # [-1,1]
                x_lat = x_lat.to(self.device).contiguous()
                h_lat = h_lat.to(self.device).contiguous()

                t0 = time.time()
                
                Hn = _sample_neighbors(h_lat, M_neighbors, sigma_h, alpha_max).to(self.device)

                
                with torch.no_grad():
                    e_ref = self.arcface(x0)  # [1,512] normalized

                sims, labs = [], []
                for s in range(0, Hn.size(0), bs_label):
                    Hb = Hn[s:s+bs_label]
                    xb = _decode_h_batch_with_denoiser(self, x_lat, Hb, seq_test, seq_test_next)
                    eb = self.arcface(xb)                           # [B,512]
                    sim_b = _cos(eb, e_ref)                         # [B]
                    y_b = (sim_b > tau).long()                      # 1=similar, 0=dissimilar
                    sims.append(sim_b); labs.append(y_b)
                y_np = torch.cat(labs, dim=0).cpu().numpy()
                # L9: train SVM boundary in h-only
                w, b0, svm = _fit_svm_boundary(Hn, y_np, C=svm_C)
                dq = _orient_to_dissimilar(h_lat, w, b0)

                # save .sav
                save_name_h = f'boundary/{exp_id}_{mode}_{idx}_h.sav'
                bundle = {
                    "w": w.detach().cpu().numpy(),           # [1, Dh]
                    "b0": float(b0),
                    "dq": dq.detach().cpu().numpy(),         # oriented
                    "tau": float(tau),
                    "svm": svm,
                    "meta": {
                        "exp_id": exp_id, "mode": mode, "idx": int(idx),
                        "num_samples": int(M_neighbors),
                        "sigma_h": float(sigma_h),
                        "alpha_max": float(alpha_max),
                        "C": float(svm_C),
                        "t0": int(self.args.t_0),
                        "n_inv_step": int(self.args.n_inv_step),
                        "Dh": int(w.numel()),
                        "label": int(label) if torch.is_tensor(label) else int(label),
                    }
                }
                joblib.dump(bundle, save_name_h)
                print(f"[BIA-H] ({mode} #{idx}) boundary saved → {save_name_h} "
                      f"({time.time()-t0:.2f}s)")
                      
                print("Evaluating SVM accuracy for h-space ...")

            # prepare test data
            if 'test' in img_lat_pairs_dic:
                test_pairs = img_lat_pairs_dic['test']
                n_test = len(test_pairs)
                test_data_h = np.empty([n_test, 512 * 8 * 8])
                test_label  = np.empty([n_test,], dtype=int)

                for step, (x0_t, x_id_t, x_lat_t, mid_h_t, label_t) in enumerate(test_pairs):
                    test_data_h[step, :] = mid_h_t.view(1, -1).cpu().numpy()
                    test_label[step] = int(label_t.cpu().numpy())

                # reload the classifier and predict
                bundle = joblib.load(save_name_h)
                classifier_h = bundle["svm"]
                correct = np.sum(test_label == val_pred_h)
                acc_h = correct / n_test

                print(f"[Accuracy | h-space] {correct}/{n_test} = {acc_h:.4f}")
                
                
        return None
        
        
        
    def blackhole_unlearning_wrap(self, img_lat_pairs_dic):
    """
    BIA Step-2 in:
      - For each (x0, x_lat, h_lat), load boundary .sav
      - Resample neighbors, pick K nearest to hyperplane
      - Walk along dq with step Δ (adapt/shrink)
      - Decode edited h via denoiser; save images
    """
        device = self.device
        os.makedirs("edit_output", exist_ok=True)
        os.makedirs("boundary", exist_ok=True)

        # schedules
        if self.args.n_test_step != 0:
            seq_test = [int(s) for s in (np.linspace(0,1,self.args.n_test_step)*self.args.t_0)]
        else:
            seq_test = list(range(self.args.t_0))
        seq_test_next = [-1] + list(seq_test[:-1])

        # params
        M_neighbors = getattr(self.args, "bh_num_samples", 128)    # resampling to select K nearest
        sigma_h     = getattr(self.args, "bh_sigma", 0.15)
        alpha_max   = getattr(self.args, "bh_alpha_max", 0.35)
        K           = getattr(self.args, "bh_k_nearest", 8)
        step_min    = getattr(self.args, "bh_step_min", 100.0)
        step_max    = getattr(self.args, "bh_step_max", 120.0)
        max_trials  = getattr(self.args, "bh_max_clip_trials", 5)

        # losses
        tau_m       = getattr(self.args, "bh_margin_tau", 0.25)  # target ArcFace margin (lower is stricter)
        lam_id      = getattr(self.args, "lambda_id",  1.0)
        lam_l2      = getattr(self.args, "lambda_l2",  1.0)
        lam_perc    = getattr(self.args, "lambda_perc", 0.5)
        wrap_thresh = getattr(self.args, "bh_wrapper_thresh", 0.35)  

        # perceptual
        perc = VGGPerceptual(device)

    
        def id_sim(img_batch, e_ref):
            e_b = self.arcface(img_batch)     # [B,512] normalized
            return _cos(e_b, e_ref)           # [B]

        exp_id = os.path.split(self.args.exp)[-1]

        for mode in ['train', 'test']:
            if mode not in img_lat_pairs_dic:
                continue
            pairs = img_lat_pairs_dic[mode]
            print(f"[BIA-Step2] {mode}: {len(pairs)} items")

            for idx, (x0, x_id, x_lat, h_lat, label) in enumerate(pairs):
                x0    = x0.to(device).contiguous()
                x_lat = x_lat.to(device).contiguous()
                h_lat = h_lat.to(device).contiguous()

                # load boundary bundle
                sav_path = os.path.join(
                    "boundary", f"{exp_id}_{mode}_{idx}_h.sav"
                )
                assert os.path.exists(sav_path), f"Boundary not found: {sav_path}"
                bundle = joblib.load(sav_path)
                w  = torch.from_numpy(bundle["w"]).to(device)   # [1, Dh]
                b0 = float(bundle["b0"])
                dq = torch.from_numpy(bundle["dq"]).to(device)  # [1, Dh]
                tau_saved = float(bundle["tau"])

                t0 = time.time()

                # resample neighbors around h_lat and pick K nearest to hyperplane
                Hn = _sample_neighbors(h_lat, M_neighbors, sigma_h, alpha_max).to(device)  # [M,512,8,8]
                Hn_flat = _flat(Hn)                                                         
                dists = _hyperplane_distance(Hn_flat, w, b0).squeeze(1)                     
                topk = torch.topk(-dists, k=min(K, Hn_flat.size(0)))                        
                idxs = topk.indices
                Hsel = Hn_flat[idxs]                                                       

                # reference identity embedding
                with torch.no_grad():
                    e_ref = self.arcface(x0)  # [1,512] normalized

                edited_imgs = []
                metrics = []

                for k_i in range(Hsel.size(0)):
                    hb = Hsel[k_i:k_i+1]  # [1, Dh]
                    
                    Delta = torch.empty(1, device=device).uniform_(step_min, step_max)
                    h_before = hb.view_as(h_lat)

                    
                    x_before = _decode_h_batch_with_denoiser(self, x_lat, h_before, seq_test, seq_test_next)

                    # try walking & guard with wrapper loss
                    for trial in range(max_trials):
                        h_after = (hb + dq * Delta).view_as(h_lat)  # walk along oriented normal
                        x_after = _decode_h_batch_with_denoiser(self, x_lat, h_after, seq_test, seq_test_next)

                        # compute losses
                        with torch.no_grad():
                            sim_after = id_sim(x_after, e_ref)        # [1]
                            l_id  = torch.clamp(sim_after - tau_m, min=0.0).mean()
                            l_l2  = F.mse_loss(x_after, x_before)
                            l_per = perc(x_after, x_before)
                            l_wrap = lam_l2 * l_l2 + lam_perc * l_per
                            total = lam_id * l_id + l_wrap

                        if l_wrap.item() < wrap_thresh:
                            # accept this step
                            edited_imgs.append(x_after.detach().cpu())
                            metrics.append({
                                "k": int(k_i),
                                "trial": int(trial),
                                "Delta": float(Delta.item()),
                                "sim_after": float(sim_after.item()),
                                "L_id": float(l_id.item()),
                                "L2": float(l_l2.item()),
                                "Perceptual": float(l_per.item()),
                                "L_wrap": float(l_wrap.item()),
                                "L_total": float(total.item())
                            })
                            break
                        else:
                            # shrink step and retry
                            Delta = Delta * 0.5

                    
                    if len(edited_imgs) < k_i + 1:
                        edited_imgs.append(x_after.detach().cpu())
                        metrics.append({
                            "k": int(k_i),
                            "trial": int(max_trials),
                            "Delta": float(Delta.item()),
                            "sim_after": float(sim_after.item()),
                            "L_id": float(l_id.item()),
                            "L2": float(l_l2.item()),
                            "Perceptual": float(l_per.item()),
                            "L_wrap": float(l_wrap.item()),
                            "L_total": float(total.item()),
                            "note": "max_trials reached"
                        })

                # save images + metrics
                base = f"{exp_id}_{mode}_{idx}"
                for j, img in enumerate(edited_imgs):
                    out_path = os.path.join("edit_output", f"{base}_bia_h_edit{j}.png")
                    tvu.save_image((img + 1) * 0.5, out_path)
                with open(os.path.join("edit_output", f"{base}_bia_h_metrics.json"), "w") as f:
                    json.dump({
                        "exp_id": exp_id, "mode": mode, "index": int(idx),
                        "tau_saved": tau_saved, "tau_margin": tau_m,
                        "params": {
                            "K": K, "step_min": step_min, "step_max": step_max,
                            "wrap_thresh": wrap_thresh, "max_trials": max_trials,
                            "lam_id": lam_id, "lam_l2": lam_l2, "lam_perc": lam_perc
                        },
                        "metrics": metrics
                    }, f, indent=2)

                print(f"[BIA-Step2][{mode} #{idx}] edits={len(edited_imgs)} "
                      f"({time.time()-t0:.2f}s) → edit_output/{base}_bia_h_edit*.png")
                      
        return None