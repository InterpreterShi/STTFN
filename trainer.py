import os
import time
import glob
import json
import sys
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from utils.logger import setup_logger
from utils.tools import EarlyStopping, get_all_result
from utils.data_loader import PeMSDataset, get_data_loader
from utils.weight_load import WeightProcess


class BaseTrainer(metaclass=ABCMeta):
    """Abstract base class for trainers."""

    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def before_train(self):
        pass

    @abstractmethod
    def train_in_epochs(self):
        pass

    @abstractmethod
    def train_one_epoch(self, epoch):
        pass

    @abstractmethod
    def vali_one_epoch(self, data_loader):
        pass

    @abstractmethod
    def evaluate(self, save_pred=False, inverse=False, checkpoint=None):
        pass

    @abstractmethod
    def after_train(self):
        pass


class STTFN_Trainer(BaseTrainer):
    """Trainer for the STTFN model."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.output_dir = './logs'
        self.val_best_loss = np.inf
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.grad_clip = True
        self.real_value = False
        self.in_len = self.args.in_len
        self.out_len = self.args.out_len
        self.log_interval = 10
        self.patience = 5
        self.train_time = 0
        self.inference_time = 0
        self.total_params = 0
        self.lambda_s = 0
        self.lambda_t = 0
        self.file_name = os.path.join(self.output_dir, args.name, args.dataset, args.experiment_name)
        self.logger = setup_logger(self.args.mode, self.file_name)
        self.logger.level("INFO")

    def train(self):
        self.before_train()
        try:
            self.train_in_epochs()
        except Exception as e:
            self.logger.error(f"An exception occurred: {e}")
        finally:
            self.after_train()

    def before_train(self):
        self.logger.info(f'args:{self.args}')
        self.model = self.args.model.to(self.device)
        if self.args.mode == 'train':
            self.model.train()
        else:
            self.model.eval()
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f'Model Summary:{self.model}')
        self.logger.info("Model Total Prameters:%.2fM" % (self.total_params / 1e6))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 3, gamma=0.96)
        self.criterion = nn.MSELoss()
        train_dataset = PeMSDataset(self.args.dataset, self.args.root_path, self.args.in_len, self.args.out_len,
                                    split_size_1=0.6, split_size_2=0.2, mode='train', normalizer=self.args.normalizer)
        self.train_loader, self.y_scaler = get_data_loader(train_dataset, self.args.batch_size, self.args.num_workers,
                                                           mode='train')
        val_dataset = PeMSDataset(self.args.dataset, self.args.root_path, self.args.in_len, self.args.out_len,
                                  split_size_1=0.6, split_size_2=0.2, mode='val', normalizer=self.args.normalizer)
        self.val_loader, _ = get_data_loader(val_dataset, self.args.batch_size, self.args.num_workers, mode='val')
        test_dataset = PeMSDataset(self.args.dataset, self.args.root_path, self.args.in_len, self.args.out_len,
                                   split_size_1=0.6, split_size_2=0.2, mode='test', normalizer=self.args.normalizer)
        self.test_loader, _ = get_data_loader(test_dataset, self.args.batch_size, self.args.num_workers, mode='test')
        self.s_w = torch.as_tensor(WeightProcess(self.args.root_path, self.args.num_nodes, self.args.dataset).s_w)
        self.s_w = self.s_w.to(self.device)
        self.tb_logger = self.delete_and_create_tb_logger()
        self.logger.add(os.path.join(self.file_name, '%s_log.log' % self.args.mode), rotation="10 MB")
        self.logger.add(sys.stdout, colorize=True,
                        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> "
                               "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                               "<level>{message}</level>")
        self.logger.info('Trainging start......')

    def delete_and_create_tb_logger(self):
        tf_dir = os.path.join(self.file_name, "tensorboard")
        log_file = os.path.join(tf_dir, "events.out.tfevents.*")
        files = glob.glob(log_file)
        for file in files:
            os.remove(file)
        tb_logger = SummaryWriter(tf_dir)
        return tb_logger

    def train_in_epochs(self):
        self.early_stopping = EarlyStopping(self.patience, verbose=True, delta=0)
        count = 0
        for epoch in range(self.args.epochs):
            self.train_one_epoch(epoch)
            count += 1
            if self.early_stopping.early_stop:
                self.val_best_loss = self.early_stopping.val_loss_min
                self.logger.info('Early stopping')
                break
        if count != 0:
            self.run_time = self.train_time / count
            self.logger.info(f'model run time: {self.run_time}')
        else:
            self.run_time = 0
        self.total_params = self.total_params / 1e6
        train_metrics = {'totoal_params': self.total_params,
                         'train time': self.run_time}
        with open(os.path.join(self.file_name, 'train_metrics.json'), 'w') as f:
            json.dump(train_metrics, f)

    def train_one_epoch(self, epoch):
        self.logger.info(f'epoch {epoch} start training')
        total_loss = 0
        epoch_loss = 0
        start_time = time.time()
        epoch_time = time.time()
        for batch_idx, (x, y) in enumerate(self.train_loader):
            x = x.float().to(self.device)
            y = y.float().to(self.device)
            st_outputs, s_out, t_out = self.model(self.s_w, x)
            if self.real_value:
                y_batch = []
                for i in range(y.shape[0]):
                    y_b = self.y_scaler.inverse_transform(y[i].cpu().detach())
                    y_batch.append(y_b)
                y = torch.as_tensor(np.stack(y_batch)).float().to(self.device)
            loss = self.criterion(st_outputs, y)
            loss += self.lambda_s * torch.mean(torch.norm(s_out, p=1)) + self.lambda_t * torch.mean(torch.norm(t_out, p=1))
            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            self.scheduler.step()
            loss1, _, _ = self.compute_order_loss(st_outputs, y)
            total_loss += loss1.item()
            epoch_loss += loss1.item()
            if (batch_idx + 1) % self.log_interval == 0 and batch_idx > 0:
                cur_loss = total_loss / self.log_interval
                elapsed_time = time.time() - start_time
                self.logger.info(f"| epoch {epoch:3d} | {batch_idx + 1:5d}/{len(self.train_loader):5d} batches | "
                                 f"lr {self.scheduler.get_last_lr()[0]:02.9f} |"
                                 f"iter time {elapsed_time / self.log_interval:5.2f} s | loss {cur_loss:5.5f}")
                total_loss = 0
                start_time = time.time()
        each_epoch_time = time.time() - epoch_time
        self.train_time += each_epoch_time
        self.logger.info(f" Epoch:{epoch} training end, cost time: {each_epoch_time} s")
        train_loss = epoch_loss / len(self.train_loader)
        val_loss = self.vali_one_epoch(self.val_loader)
        test_loss = self.vali_one_epoch(self.test_loader)
        self.tb_logger.add_scalar('train_loss', train_loss, epoch)
        self.tb_logger.add_scalar('val_loss', val_loss, epoch)
        self.tb_logger.add_scalar('test_loss', test_loss, epoch)
        self.logger.info(f'Epoch: {epoch + 1}, Steps:{len(self.train_loader)} | '
                         f'Train Loss:{train_loss} | Val Loss:{val_loss} | Test Loss:{test_loss}')
        self.early_stopping(val_loss, self.model, self.file_name)
        self.val_best_loss = self.early_stopping.val_loss_min

    def vali_one_epoch(self, data_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data_loader):
                x = x.float().to(self.device)
                y = y.float().to(self.device)
                st_outputs, s_out, t_out = self.model(self.s_w, x)
                if self.real_value:
                    y_batch = []
                    for i in range(y.shape[0]):
                        y_b = self.y_scaler.inverse_transform(y[i].cpu().detach())
                        y_batch.append(y_b)
                    y = torch.as_tensor(np.stack(y_batch)).float().to(self.device)
                loss1, _, _ = self.compute_order_loss(st_outputs, y)
                total_loss += loss1.item()
        self.model.train()
        return total_loss / len(self.val_loader)

    def compute_order_loss(self, outputs, y):
        output1 = outputs[0, :, :]
        target1 = y[0, :, :]
        for i in range(outputs.shape[0]):
            if i == 0:
                continue
            else:
                output1 = torch.cat((output1, outputs[i, -1, :].reshape(1, -1)), dim=0)
                target1 = torch.cat((target1, y[i, -1, :].reshape(1, -1)), dim=0)
        loss = self.criterion(output1, target1)
        return loss, output1.cpu().detach(), target1.cpu().detach()

    def evaluate(self, save_pred=False, inverse=False, checkpoint=None):
        logger.remove()
        logger.add(os.path.join(self.file_name, '%s_log.log' % self.args.mode), rotation="10 MB", level="INFO")
        logger.add(sys.stdout, colorize=True,
                   format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> "
                          "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                          "<level>{message}</level>")
        self.model = self.args.model.to(self.device)
        if checkpoint is not None and os.path.exists(checkpoint):
            model_dict = torch.load(self.args.checkpoint, map_location='cpu')
            self.model.load_state_dict(model_dict)
        else:
            error_message = f"checkpoint file not found: {checkpoint} or checkpoint is None"
            logger.error(error_message)
            raise FileNotFoundError(error_message)
        
        test_dataset = PeMSDataset(self.args.dataset, self.args.root_path, self.args.in_len, self.args.out_len,
                                   split_size_1=0.6, split_size_2=0.2, mode=self.args.mode,
                                   normalizer=self.args.normalizer)
        test_loader, self.y_scaler = get_data_loader(test_dataset, self.args.batch_size,
                                                     self.args.num_workers, mode=self.args.mode)
        self.s_w = torch.as_tensor(WeightProcess(self.args.root_path, self.args.num_nodes, self.args.dataset).s_w)
        self.s_w = self.s_w.to(self.device)
        self.criterion = nn.MSELoss()
        self.model.eval()

        # Lists to store full predictions and targets
        preds_list = []
        targets_list = []
        evaluate_time = 0
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(test_loader):
                start = time.time()
                x = x.float().to(self.device)
                y = y.float().to(self.device) # [B, T, N]
                
                st_outputs, s_out, t_out = self.model(self.s_w, x) # [B, T, N]
                
                batch_time = time.time() - start
                evaluate_time += batch_time
                
                # Inverse transform if needed
                if inverse and self.real_value == False: # real_value logic in original code was weird, assuming false usually
                    # Process each sample in batch
                    # st_outputs: [B, T, N]
                    # y: [B, T, N]
                    pred_batch = st_outputs.cpu().numpy()
                    target_batch = y.cpu().numpy()
                    
                    batch_preds = []
                    batch_targets = []
                    
                    for i in range(pred_batch.shape[0]):
                        # Inverse transform expects [Samples, Features] usually
                        # Here each sample is [T, N]. 
                        # Assuming scaler was fit on data shape compatible with [..., C] or [..., N]
                        # In PeMSDataset, data is [T, N, C]. Scaler fits on that.
                        # If C=1, scaling is usually on the value.
                        # The original code did: self.y_scaler.inverse_transform(y[i].cpu().detach())
                        
                        p_i = self.y_scaler.inverse_transform(pred_batch[i])
                        t_i = self.y_scaler.inverse_transform(target_batch[i])
                        batch_preds.append(p_i)
                        batch_targets.append(t_i)
                        
                    preds_list.append(np.stack(batch_preds))
                    targets_list.append(np.stack(batch_targets))
                else:
                    preds_list.append(st_outputs.cpu().numpy())
                    targets_list.append(y.cpu().numpy())

        self.inference_time = evaluate_time / len(test_loader)
        
        # Concatenate all batches: [Total_Samples, Out_Len, Num_Nodes]
        e_output = np.concatenate(preds_list, axis=0) 
        e_target = np.concatenate(targets_list, axis=0)

        # Multi-Horizon Evaluation
        horizons = [3, 6, 9, 12]
        horizon_metrics = {}
        
        logger.info(f"{'='*20} Multi-Horizon Evaluation {'='*20}")
        headers = ["Horizon", "MAE", "RMSE", "MAPE"]
        logger.info(f"{headers[0]:<10} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10}")
        logger.info("-" * 45)
        
        for h in horizons:
            if h > self.out_len:
                continue
            
            # Slice: [Total, h, Nodes]
            pred_h = e_output[:, :h, :]
            target_h = e_target[:, :h, :]
            
            # Reshape to flatten for metrics: [Total*h*Nodes, 1]
            mse_h, rmse_h, mae_h, mape_h, _ = get_all_result(
                pred_h.reshape(-1, 1), 
                target_h.reshape(-1, 1), 
                multiple=False
            )
            
            logger.info(f"{h:<10} {mae_h:<10.4f} {rmse_h:<10.4f} {mape_h:<10.4%}")
            horizon_metrics[f'horizon_{h}'] = {'mae': mae_h, 'rmse': rmse_h, 'mape': mape_h}

        # Overall metrics (using full out_len)
        mse, rmse, mae, mape, r2 = get_all_result(e_output.reshape(-1, 1), e_target.reshape(-1, 1), multiple=False)
        SSE = np.sum((e_target.reshape(-1, 1) - e_output.reshape(-1, 1)) ** 2)
        SST = np.sum((e_target.reshape(-1, 1) - e_target.reshape(-1, 1).mean()) ** 2)
        Rsq = 1 - SSE / SST
        
        metrics = {
            'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2, 'Rsq': Rsq,
            'inference_time': self.inference_time,
            'horizon_metrics': horizon_metrics
        }
        
        logger.info(f"Overall Result --> mse:{mse:.4f} | rmse:{rmse:.4f} | mae:{mae:.4f} | mape:{mape:.4f} | r2:{r2:.4f}")
        
        with open(os.path.join(self.file_name, 'evaluate_metrics.json'), 'w') as f:
            # Convert numpy types/floats for JSON serialization
            import json
            def convert(o):
                if isinstance(o, np.float32): return float(o)
                if isinstance(o, np.float64): return float(o)
                return o
            json.dump(metrics, f, default=convert, indent=2)
            
        if save_pred:
            np.save(os.path.join(self.file_name, 'true_pred_dict.npy'), {'truth': e_target, 'pred': e_output})
            
        return metrics

    def process_batch(self, e_output, e_target, outputs, y, batch_idx, inverse=False):
        # Deprecated
        pass

    def after_train(self):
        self.logger.info(f'training is done, best val loss:{self.val_best_loss}')


trainer_dict = {'STTFN': STTFN_Trainer}

__all__ = ['STTFN_Trainer', 'trainer_dict']