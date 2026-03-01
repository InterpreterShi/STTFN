import argparse
import os
import time
import torch

from utils.tools import get_cfg, setup_seed
from models.sttfn.sttfn import STTFN
from trainer import trainer_dict


torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

DATASET = 'pems08'   # pems04,pmes08
DEVICE = 'cuda:0'
MODEL = 'STTFN'
ROOT_PATH = './dataset/'

cfg_data_file = f'cfg/datasets/{DATASET}.yaml'
_, _, cfg_defalut, cfg_all = get_cfg(cfg_data_file)
EXPERIMENT_NAME = f"{MODEL}_{cfg_defalut['batch_size']}_{cfg_defalut['lr']}"
CHECK_POINT = os.path.join('logs', MODEL, DATASET, EXPERIMENT_NAME, 'checkpoint.pth')
cfg_all['DATASET'] = DATASET
cfg_all['DEVICE'] = DEVICE
cfg_all['MODEL'] = MODEL
cfg_all['ROOT_PATH'] = ROOT_PATH
cfg_all['EXPERIMENT_NAME'] = EXPERIMENT_NAME
cfg_all['CHECK_POINT'] = CHECK_POINT


def parse_args(cfg):
    parser = argparse.ArgumentParser(description=f"{cfg['MODEL'].upper()} Train Parser")
    parser.add_argument("-expn", "--experiment_name", type=str, default=cfg['EXPERIMENT_NAME'])
    parser.add_argument("-n", "--name", type=str, default=cfg['MODEL'], help="model name")
    parser.add_argument("--checkpoint", type=str, default=cfg['CHECK_POINT'], help="path of trained model")

    parser.add_argument("--root_path", type=str, default=cfg['ROOT_PATH'], help="dataset root path")
    parser.add_argument('--dataset', default=cfg['DATASET'], type=str, help="dataset name")
    parser.add_argument('--device', default=cfg['DEVICE'], type=str, help="device for training")
    parser.add_argument('--num_nodes', default=cfg['num_nodes'], type=int, help="num nodes of dataset")
    parser.add_argument('-l', '--length', default=cfg['total_length'], type=int,
                        help="total time series length of dataset")
    parser.add_argument('--channel', default=cfg['channel'], type=int, help="feature num of dataset")
    parser.add_argument('--features', default=cfg['features'], type=list, help="features of dataset")
    parser.add_argument('--mode', default=cfg['mode'], type=str, help="mode of dataloader")

    parser.add_argument('--in_len', default=cfg['in_len'], type=int, help="input length of dataset")
    parser.add_argument('--out_len', default=12, type=int, help="output length of dataset")
    parser.add_argument('--normalizer', default=cfg['normalizer'], type=str,
                        help="normalizer type of dataset")

    parser.add_argument('--epochs', default=cfg['epochs'], type=int, help="training epochs")
    parser.add_argument('--patience', default=cfg['patience'], type=int, help="early stopping patience")
    parser.add_argument('--seed', default=cfg['seed'], type=int, help="random seed")
    parser.add_argument('--batch_size', default=cfg['batch_size'], type=int,
                        help="batch size of datasets when training/evaluation")
    parser.add_argument('--lr', default=cfg['lr'], type=float, help="init learning rate of training")
    parser.add_argument('--num_workers', default=cfg['num_workers'], type=int,
                        help="number of worker threads for data loading")
    parser.add_argument('--resume', default=cfg['resume'], type=bool,
                        help="resume training from last checkpoint")

    parser.add_argument('--in_dim', default=cfg['in_dim'], type=int, help="input dimension")
    parser.add_argument('--out_dim', default=cfg['out_dim'], type=int, help="output dimension")

    sp_cfg = cfg['spatial_plane']
    parser.add_argument('--embed_dim', default=sp_cfg['embed_dim'], type=int,
                        help="embedding dimension of SRGCN")
    parser.add_argument('--spatial_attention', default=sp_cfg['spatial_attention'], type=bool,
                        help="whether to output spatial attention")

    tp_cfg = cfg['temporal_plane']
    parser.add_argument('--d_model', default=tp_cfg['d_model'], type=int, help="d_model of AutoTRT")
    parser.add_argument('--n_heads', default=tp_cfg['n_heads'], type=int, help="head number of AutoTRT")
    parser.add_argument('--dropout', default=tp_cfg['dropout'], type=float, help="dropout probability")
    parser.add_argument('--num_layers', default=tp_cfg['num_layers'], type=int, help="layer number of AutoTRT")
    parser.add_argument('--factor', default=tp_cfg['factor'], type=int, help="hyperparameter of Auto-correlation")
    parser.add_argument('--temporal_attention', default=tp_cfg['temporal_attention'], type=bool,
                        help="whether to output temporal attention")
    parser.add_argument('--full_attention', default=tp_cfg['full_attention'], type=bool, help="choose full attention")
    parser.add_argument('--grid_ed', default=sp_cfg['grid_ed'], type=list, help="grid parameters of embed_dim")
    parser.add_argument('--grid_factor', default=tp_cfg['grid_factor'], type=list, help="grid parameters of factor")

    return parser.parse_args()


def build_model(args):
    return STTFN(
        args.num_nodes,
        args.in_len,
        args.out_len,
        args.channel,
        args.embed_dim,
        args.d_model,
        args.n_heads,
        args.num_layers,
        args.dropout,
        args.factor,
        args.spatial_attention,
        args.temporal_attention,
        args.full_attention,
    )


args = parse_args(cfg_all)
model = build_model(args)
args.model = model
setup_seed(args.seed)


def main():
    for mode in ['train', 'val', 'test']:
        args.mode = mode
        trainer = trainer_dict[MODEL](args)
        if mode == 'train':
            trainer.train()
        else:
            trainer.evaluate(save_pred=True, inverse=True, checkpoint=args.checkpoint)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()