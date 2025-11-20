import argparse
import traceback
import logging
import yaml
import sys
import os
import torch
import numpy as np

from blackhole import BIAUnlearn
from configs.paths_config import HYBRID_MODEL_PATHS

# ----------------------------------------
# helpers
# ----------------------------------------
def dict2namespace(config):
    ns = argparse.Namespace()
    for k, v in config.items():
        setattr(ns, k, dict2namespace(v) if isinstance(v, dict) else v)
    return ns

def _ensure_dirs(args):
    os.makedirs(args.exp, exist_ok=True)
    os.makedirs('checkpoint', exist_ok=True)
    os.makedirs('precomputed', exist_ok=True)
    os.makedirs('boundary', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    args.image_folder = os.path.join(args.exp, 'image_samples_new')
    os.makedirs(args.image_folder, exist_ok=True)

# ----------------------------------------
# arg parsing
# ----------------------------------------
def parse_args_and_config():
    parser = argparse.ArgumentParser(description="BIAUnlearn / BIA runner")

		# ---- Black-Hole (BIA) pipeline 
	parser.add_argument('--bh_precompute', action='store_true', help='BIA Step-0: precompute latents')
	parser.add_argument('--bh_boundary', action='store_true', help='BIA Step-1: boundary search in h-space')
	parser.add_argument('--bh_unlearn', action='store_true', help='BIA Step-2: unlearning + wrapping in h-space')
	parser.add_argument('--bh_run_all', action='store_true', help='Run all BIA steps: precompute -> boundary -> unlearn')

	# ---- BIA hyperparams 
	parser.add_argument('--bh_num_samples', type=int, default=128)
	parser.add_argument('--bh_sigma', type=float, default=0.15)
	parser.add_argument('--bh_alpha_max', type=float, default=0.35)
	parser.add_argument('--bh_tau', type=float, default=0.35)
	parser.add_argument('--bh_svm_C', type=float, default=1.0)
	parser.add_argument('--bh_batch_size', type=int, default=16)
	parser.add_argument('--bh_k_nearest', type=int, default=8)
	parser.add_argument('--bh_step_min', type=float, default=100.0)
	parser.add_argument('--bh_step_max', type=float, default=120.0)
	parser.add_argument('--bh_max_clip_trials', type=int, default=5)
	parser.add_argument('--bh_wrapper_thresh', type=float, default=0.35)
	parser.add_argument('--bh_margin_tau', type=float, default=0.25)
	parser.add_argument('--lambda_id', type=float, default=1.0)
	parser.add_argument('--lambda_l2', type=float, default=1.0)
	parser.add_argument('--lambda_perc', type=float, default=0.5)

  
      
    parser.add_argument('--recon_exp', action='store_true')
    parser.add_argument('--find_best_image', action='store_true')

    # Defaults / meta
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config under ./configs')
    parser.add_argument('--seed', type=int, default=909090)
    parser.add_argument('--exp', type=str, default='./runs/', help='Base output dir / experiment name')
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--verbose', type=str, default='info', help='info | debug | warning | critical')
    parser.add_argument('--ni', type=int, default=1, help='No interaction (non-blocking overwrite)')
    parser.add_argument('--align_face', type=int, default=1)

    # Text
    parser.add_argument('--edit_attr', type=str, default=None)
    parser.add_argument('--src_txts', type=str, action='append')
    parser.add_argument('--trg_txts', type=str, action='append')
    parser.add_argument('--target_class_num', type=str, default=None)

    # Sampling
    parser.add_argument('--t_0', type=int, default=400)
    parser.add_argument('--n_inv_step', type=int, default=40)
    parser.add_argument('--n_train_step', type=int, default=6)
    parser.add_argument('--n_test_step', type=int, default=40)
    parser.add_argument('--sample_type', type=str, default='ddim')
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--start_distance', type=float, default=-150.0)
    parser.add_argument('--end_distance', type=float, default=150.0)
    parser.add_argument('--edit_img_number', type=int, default=20)

    # Train & Test
    parser.add_argument('--do_train', type=int, default=1)
    parser.add_argument('--do_test', type=int, default=1)
    parser.add_argument('--save_train_image', type=int, default=1)
    parser.add_argument('--bs_train', type=int, default=1)
    parser.add_argument('--bs_test', type=int, default=1)
    parser.add_argument('--n_precomp_img', type=int, default=5000)
    parser.add_argument('--n_train_img', type=int, default=4000)
    parser.add_argument('--n_test_img', type=int, default=200)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--img_path', type=str, default=None)
    parser.add_argument('--deterministic_inv', type=int, default=1)
    parser.add_argument('--hybrid_noise', type=int, default=0)
    parser.add_argument('--model_ratio', type=float, default=1.0)

    # Loss & Optimization
    parser.add_argument('--l1_loss_w', type=float, default=0.0)
    parser.add_argument('--id_loss_w', type=float, default=0.0)
    parser.add_argument('--n_iter', type=int, default=1)
    parser.add_argument('--scheduler', type=int, default=1)
    parser.add_argument('--sch_gamma', type=float, default=1.3)

    
    parser.add_argument('--bh_num_samples', type=int, default=128)
    parser.add_argument('--bh_sigma', type=float, default=0.15)
    parser.add_argument('--bh_alpha_max', type=float, default=0.35)
    parser.add_argument('--bh_tau', type=float, default=0.35)
    parser.add_argument('--bh_svm_C', type=float, default=1.0)
    parser.add_argument('--bh_batch_size', type=int, default=16)
    parser.add_argument('--bh_k_nearest', type=int, default=8)
    parser.add_argument('--bh_step_min', type=float, default=100.0)
    parser.add_argument('--bh_step_max', type=float, default=120.0)
    parser.add_argument('--bh_max_clip_trials', type=int, default=5)
    parser.add_argument('--bh_wrapper_thresh', type=float, default=0.35)
    parser.add_argument('--bh_margin_tau', type=float, default=0.25)
    parser.add_argument('--lambda_id', type=float, default=1.0)
    parser.add_argument('--lambda_l2', type=float, default=1.0)
    parser.add_argument('--lambda_perc', type=float, default=0.5)

    args = parser.parse_args()

    # load config
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    
    
    # logging
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f'Unsupported log level: {args.verbose}')
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s'))
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(level)

    # dirs
    _ensure_dirs(args)

    # device & seeds
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info(f"Using device: {device}")
    new_config.device = device
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    return args, new_config

# ----------------------------------------
# main
# ----------------------------------------
def main():
    args, config = parse_args_and_config()
    print(">" * 80)
    logging.info(f"Exp instance id = {os.getpid()}")
    logging.info(f"Exp comment = {args.comment}")
    logging.info("Config loaded")
    print("<" * 80)

    runner = BIAUnlearn(args, config)

    try:
        # ---------- Original modes ----------
        if args.bh_run_all:
            pairs = runner.blackhole_formation()                          
            runner.blackhole_boundary_search(pairs)                 
            runner.blackhole_unlearning_wrap(pairs)                 

        elif args.bh_precompute:
            runner.blackhole_formation()

        elif args.bh_boundary:
            # call formation if you don't already have pairs in memory
            pairs = runner.blackhole_formation()
            runner.blackhole_boundary_search(pairs)

        elif args.bh_unlearn:
            # expects boundaries already saved by previous step
            pairs = runner.blackhole_formation()
            runner.blackhole_unlearning_wrap(pairs)

        else:
            print('Choose one mode!')
            raise ValueError

    except Exception:
        logging.error(traceback.format_exc())

    return 0

    

if __name__ == '__main__':
    sys.exit(main())
