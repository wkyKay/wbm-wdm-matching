from fileinput import filename
from tkinter import NO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from layers.core import Flatten
from torch.utils.tensorboard import SummaryWriter
import shutil
import os
import sys
import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from utils.logging import make_epoch_description

# ============================================================
# 2. 跨模态检索模型
# ============================================================

class CrossModalRetrievalModel(nn.Module):
    """
    跨模态检索模型
    
    使用预训练的 WaPIRL encoders + 可学习的投影头
    """
    
    def __init__(self, 
                 encoder_wbm,      # 预训练的 WBM encoder
                 encoder_wdm,      # 预训练的 WDM encoder
                 embedding_dim=128,
                 freeze_encoder=True):
        super().__init__()
        
        # 冻结预训练的 encoders（只训练投影头）
        self.encoder_wbm = encoder_wbm
        self.encoder_wdm = encoder_wdm
        
        if freeze_encoder:
            for param in self.encoder_wbm.parameters():
                param.requires_grad = False
            for param in self.encoder_wdm.parameters():
                param.requires_grad = False
            self.encoder_wbm.eval()
            self.encoder_wdm.eval()
        
        # 可学习的投影头 (包含 GAP + Flatten + Linear)
        self.proj_wbm = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(encoder_wbm.out_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, embedding_dim)
        )
        
        self.proj_wdm = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(encoder_wdm.out_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, embedding_dim)
        )
    
    def encode_wbm(self, wbm):
        """编码 WBM"""
        with torch.no_grad() if not self.encoder_wbm.training else torch.enable_grad():
            feat = self.encoder_wbm(wbm)
        emb = F.normalize(self.proj_wbm(feat), p=2, dim=1)
        return emb

    def encode_wdm(self, wdm):
        """编码 WDM"""
        with torch.no_grad() if not self.encoder_wdm.training else torch.enable_grad():
            feat = self.encoder_wdm(wdm)
        emb = F.normalize(self.proj_wdm(feat), p=2, dim=1)
        return emb
    
    def forward(self, wbm, wdm):
        """前向传播"""
        wbm_emb = self.encode_wbm(wbm)
        wdm_emb = self.encode_wdm(wdm)
        return wbm_emb, wdm_emb
    
    def retrieval(self, query_wbm, gallery_wdm):
        """
        检索: 给定WBM，检索匹配的WDM
        
        Args:
            query_wbm: (B, ...) - 查询的WBM
            gallery_wdm: (N, ...) - WDM图库
        
        Returns:
            similarities: (B, N) - 相似度矩阵
            indices: (B, N) - 排序后的索引
        """
        self.eval()
        with torch.no_grad():
            q_emb = self.encode_wbm(query_wbm)
            g_emb = self.encode_wdm(gallery_wdm)
            
            sim = torch.mm(q_emb, g_emb.T)
            sorted_sim, indices = torch.sort(sim, dim=1, descending=True)
        
        return sorted_sim.cpu().numpy(), indices.cpu().numpy()




def collate_paired_data(batch):
    """自定义batch整理函数"""
    wbm = torch.stack([item['wbm'] for item in batch])
    wdm = torch.stack([item['wdm'] for item in batch])
    idx = torch.tensor([item['idx'] for item in batch])
    return wbm, wdm, idx


# ============================================================
# 4. 训练器
# ============================================================

class CrossModalTrainer:
    """
    跨模态训练器
    """
    
    def __init__(self, 
                 model,
                 triplet_loss,
                 optimizer,
                 device='cuda',
                 scheduler=None,
                 logger=None,
                 num_negatives: int = 100,
                 metrics: dict = None,
                 write_summary: bool = True,
                 local_rank: int = 0,
                 checkpoint_dir: str = None,
                 **kwargs):
        self.model = model.to(device)
        self.triplet_loss = triplet_loss.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.logger = logger
        self.checkpoint_dir = checkpoint_dir
        self.num_negatives = num_negatives
        self.checkpoint_dir = checkpoint_dir
        self.local_rank = local_rank
        self.train_set = None
        self.valid_set = None
        self.test_set = None
        self.metrics = metrics if isinstance(metrics, dict) else None

        if write_summary:
            self.writer = SummaryWriter(self.checkpoint_dir)
        else:
            self.writer = None

    
    def sample_negatives(self, batch_idxs, num_negatives, data):
        """
        Args:
            batch_idxs: list or 1D tensor of indices to exclude (当前 batch 的 idx)
            num_negatives: 需要采样的负样本数量
            data: dataset object
        Returns:
            indices: Tensor of shape (num_negatives,) 采样的负样本索引
            neg_samples: Tensor of shape (num_negatives, C, H, W)
        """
        N = len(data)
        device = next(self.model.parameters()).device

        # 构建候选索引列表（排除 batch 内的样本）
        all_indices = torch.arange(N, device=device)
        mask = torch.ones(N, dtype=torch.bool, device=device)
        mask[batch_idxs] = False
        candidate_indices = all_indices[mask]   # 可选的负样本索引

        num_candidates = candidate_indices.size(0)
        if num_candidates == 0:
            raise RuntimeError("No negative samples available after excluding batch indices.")

        # 如果 num_negatives 大于候选数量，可以采取两种策略：
        # 1) 允许重复（回退到有放回） 2) 减少采样数量。这里采用第2种并给出警告。
        if num_negatives > num_candidates:
            print(f"Warning: num_negatives ({num_negatives}) exceeds available candidates ({num_candidates}). "
                f"Sampling without replacement only {num_candidates} negatives.")
            num_negatives = num_candidates

        # 无放回采样
        sampled_indices_in_candidate = torch.multinomial(
            torch.ones(num_candidates, device=device),  # 均匀概率权重
            num_negatives,
            replacement=False
        )
        indices = candidate_indices[sampled_indices_in_candidate]   # 原始数据集的索引

        # 获取样本
        neg_samples = torch.stack([data[idx.item()]['wdm'] for idx in indices])
        return indices, neg_samples
    
    def evaluate(self, test_set, batch_size=32, num_workers=4, save_images=False):
        """
        评估检索性能
        
        Returns:
            metrics: dict, 包含 Recall@K
        """
        self.model.eval()
        self.test_set = test_set
        
        test_loader = DataLoader(test_set, batch_size, num_workers=num_workers, shuffle=True, pin_memory=False)
        out = {'loss': 0.}
        steps_per_epoch = len(test_loader)

        with torch.no_grad():
            for batch in test_loader:
                idxs = batch['idx']
                wbm = batch['wbm']
                wdm = batch['wdm']
                neg_idxs, neg_wdm = self.sample_negatives(idxs, num_negatives=self.num_negatives, data=test_set)
                
                wbm_emb = self.model.encode_wbm(wbm.to(self.device))
                wdm_emb = self.model.encode_wdm(wdm.to(self.device))
                neg_wdm_emb = self.model.encode_wdm(neg_wdm.to(self.device))
                
                wbm_paths = batch['wbm_path']

                loss, logits = self.triplet_loss(wbm_emb, wdm_emb, neg_wdm_emb)
                _, topk_all_col_idx = torch.topk(logits, k=3, dim=1)
                if save_images:
                    wdm_paths_pos = batch['wdm_path']          # 正样本路径列表，长度 batch_size
                    wdm_paths_neg = [test_set[idx.item()]['wdm_path'] for idx in neg_idxs]  # 负样本路径列表，长度 num_negatives

                    for i in range(len(topk_all_col_idx)):
                        query_path = wbm_paths[i]
                        file_name = os.path.basename(query_path)
                        topk_cols = topk_all_col_idx[i]        # 长度为 k 的 tensor

                        topk_paths = []
                        for col in topk_cols.tolist():
                            if col == 0:
                                # 正样本
                                topk_paths.append(wdm_paths_pos[i])
                            else:
                                # 负样本，col 范围 1..num_negatives，对应 wdm_paths_neg 的索引 col-1
                                topk_paths.append(wdm_paths_neg[col - 1])

                        # 后续保存代码不变，但注意目录命名问题（见下文）
                        wbm_dir = os.path.join(self.checkpoint_dir, 'eval', file_name, 'wbm')
                        wdm_dir = os.path.join(self.checkpoint_dir, 'eval', file_name, 'wdm')
                        os.makedirs(wbm_dir, exist_ok=True)
                        os.makedirs(wdm_dir, exist_ok=True)

                        new_wbm_path = os.path.join(wbm_dir, os.path.basename(query_path))
                        shutil.copy(query_path, new_wbm_path)
                        for p in topk_paths:
                            new_wdm_path = os.path.join(wdm_dir, os.path.basename(p))
                            shutil.copy(p, new_wdm_path)

                # Accumulate loss & metrics
                out['loss'] += loss.item()
                if self.metrics is not None:
                    assert isinstance(self.metrics, dict)
                    for metric_name, metric_function in self.metrics.items():
                        if metric_name not in out.keys():
                            out[metric_name] = 0.
                        logits = logits.detach()                                     # (N, 1+ num_negatives)
                        targets = torch.zeros(logits.size(0), device=logits.device)  # (N, )
                        out[metric_name] += metric_function(logits, targets).item()
        return {k: v / steps_per_epoch for k, v in out.items()}
        
    def valid(self, dataloader):
        """
        评估检索性能
        """
        self.model.eval()
        out = {'loss': 0.}
        steps_per_epoch = len(dataloader)
        with torch.no_grad():
            for batch in dataloader:
                idxs = batch['idx']
                wbm = batch['wbm']
                wdm = batch['wdm']
                neg_idxs, neg_wdm = self.sample_negatives(idxs, num_negatives=self.num_negatives, data=self.valid_set)
                
                wbm_emb = self.model.encode_wbm(wbm.to(self.device))
                wdm_emb = self.model.encode_wdm(wdm.to(self.device))
                neg_wdm_emb = self.model.encode_wdm(neg_wdm.to(self.device))
                
                loss, logits = self.triplet_loss(wbm_emb, wdm_emb, neg_wdm_emb)
                out['loss'] += loss.item()
                if self.metrics is not None:
                    assert isinstance(self.metrics, dict)
                    for metric_name, metric_function in self.metrics.items():
                        if metric_name not in out.keys():
                            out[metric_name] = 0.
                        logits = logits.detach()                                     # (N, 1+ num_negatives)
                        targets = torch.zeros(logits.size(0), device=logits.device)  # (N, )
                        out[metric_name] += metric_function(logits, targets).item()        
        return {k: v / steps_per_epoch for k, v in out.items()}
    

    def train(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        out = {'loss': 0.0}
        steps_per_epoch = len(dataloader)
        with tqdm.tqdm(steps_per_epoch, desc=f'Train:', leave=False) as pbar:
            for i, batch in enumerate(dataloader):
                idxs = batch['idx']
                wbm = batch['wbm']
                wdm = batch['wdm']
                neg_idxs, neg_wdm = self.sample_negatives(idxs, num_negatives=self.num_negatives, data=self.train_set) 
                # 编码
                wbm_emb = self.model.encode_wbm(wbm.to(self.device))
                wdm_emb = self.model.encode_wdm(wdm.to(self.device))  
                neg_wdm_emb = self.model.encode_wdm(neg_wdm.to(self.device))
                
                # 计算损失
                loss, logits = self.triplet_loss(wbm_emb, wdm_emb, neg_wdm_emb)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
                out['loss'] += loss.item()
                # print(f'metric信息{self.metrics}')
                if self.metrics is not None:
                    assert isinstance(self.metrics, dict)
                    for metric_name, metric_function in self.metrics.items():
                        if metric_name not in out.keys():
                            out[metric_name] = 0.
                        with torch.no_grad():
                            logits = logits.detach()
                            targets = torch.zeros(logits.size(0), device=logits.device)
                            out[metric_name] += metric_function(logits, targets).item()

                desc = f" Batch - [{i+1:>4}/{steps_per_epoch:>4}]: "
                desc += " | ".join ([f"{k}: {v/(i+1):.4f}" for k, v in out.items()])
                pbar.set_description_str(desc)
                pbar.update(1)
        # return total_loss / num_batches
        return {k: v / steps_per_epoch for k, v in out.items()}


    def run(self,
            train_set: torch.utils.data.Dataset,
            valid_set: torch.utils.data.Dataset,
            epochs: int,
            batch_size: int,
            num_workers: int = 0, 
            **kwargs):
        """完整训练流程"""
        # 假设你的 dataset 是训练用的完整数据集（实现了 __getitem__ 返回 dict）
        self.train_set = train_set
        self.valid_set = valid_set
        train_loader = DataLoader(train_set, batch_size, num_workers=num_workers, shuffle=True, pin_memory=False)
        valid_loader = DataLoader(valid_set, batch_size, num_workers=num_workers, shuffle=True, pin_memory=False)
        save_every = kwargs.get('save_every', epochs)

        with tqdm.tqdm(total=epochs, leave=True) as pbar:
            best_valid_loss = float('inf')
            best_epoch = 0
            for epoch in range(1, epochs+1):
                # 训练&验证
                train_history = self.train(train_loader)  
                valid_history = self.valid(valid_loader)
                
                epoch_history = {
                    'loss': {
                        'train': train_history.get('loss'),
                        'valid': valid_history.get('loss')
                    },
                }
                if self.metrics is not None:
                    assert isinstance(self.metrics, dict)
                    for metric_name, _ in self.metrics.items():
                        epoch_history[metric_name] = {
                            'train': train_history.get(metric_name),
                            'valid': valid_history.get(metric_name),
                        }
                
                # Tensorboard
                if self.writer is not None:
                    for metric_name, metric_dict in epoch_history.items():
                            self.writer.add_scalars(
                                main_tag=metric_name,
                                tag_scalar_dict=metric_dict,
                                global_step=epoch
                            )
                    if self.scheduler is not None:
                            self.writer.add_scalar(
                                tag='lr',
                                scalar_value=self.scheduler.get_last_lr()[0],
                                global_step=epoch
                    )

                # # Save model if it is the current best
                valid_loss = epoch_history['loss']['valid']
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_epoch = epoch
                    if self.local_rank == 0:
                        best_ckpt = os.path.join(self.checkpoint_dir, "best_model.pt")
                        self.save_checkpoint(best_ckpt, epoch=epoch, **epoch_history)
                    # self.memory.save(os.path.join(os.path.dirname(self.best_ckpt), 'best_memory.pt'), epoch=epoch)

                # Save intermediate models
                # print(f'epoch:{epoch}, save_every:{save_every}')
                if epoch % save_every == 0:
                    if self.local_rank == 0:
                        new_ckpt = os.path.join(self.checkpoint_dir, f'epoch_{epoch:04d}.loss_{valid_loss:.4f}.pt')
                        self.save_checkpoint(new_ckpt, epoch=epoch, **epoch_history)

                # 学习率调度
                if self.scheduler:
                    self.scheduler.step()   
                
                # 6. Logging
                desc = make_epoch_description(
                        history=epoch_history,
                        current=epoch,
                        total=epochs,
                        best=best_epoch
                    )
                # pbar.set_description_str(desc)
                pbar.update(1)

                if self.logger is not None:
                    self.logger.info(desc)
        
        # # 恢复最佳模型
        # self.model.load_state_dict(self.best_model_state)
        # self.logger.info(f"\nBest Recall@1: {best_recall:.4f}")

        if self.local_rank == 0:
            last_ckpt = os.path.join(self.checkpoint_dir, "last_model.pt")
            self.save_checkpoint(last_ckpt, epoch=epoch, **epoch_history)
            # self.memory.save(os.path.join(os.path.dirname(self.last_ckpt), 'last_memory.pt'), epoch=epoch)
    
    def _set_learning_phase(self, train: bool = False):
        if train:
            self.model.encoder_wbm.train()
            self.model.encoder_wdm.train()
        else:
            self.model.encoder_wbm.eval()
            self.model.encoder_wdm.eval()

    def save_checkpoint(self, path: str, **kwargs):
        ckpt = {
            'encoder_wbm': self.model.encoder_wbm.state_dict(),
            'encoder_wdm': self.model.encoder_wdm.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if \
                self.scheduler is not None else None
        }
        if kwargs:
            ckpt.update(kwargs)
        torch.save(ckpt, path)

    def load_model_from_checkpoint(self, path: str):
        ckpt = torch.load(path)
        self.model.encoder_wbm.load_state_dict(ckpt['encoder_wbm'])
        self.model.encoder_wdm.load_state_dict(ckpt['encoder_wdm'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt['scheduler'])

    def load_history_from_checkpoint(self, path: str):
        ckpt = torch.load(path)
        del ckpt['encoder_wbm']
        del ckpt['encoder_wdm']
        del ckpt['optimizer']
        del ckpt['scheduler']
        return ckpt
