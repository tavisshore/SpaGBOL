import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
from torchvision.transforms import v2
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import lightning.pytorch as pl
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GraphSAGE
from pytorch_metric_learning import losses
from sklearn.metrics.pairwise import haversine_distances
torchvision.disable_beta_transforms_warning()

from src.utils.data import GraphData, GraphDataset
from src.utils.metric import recall_accuracy



class ConvNextExtractor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.map_conv = torch.load(f'{self.args.path}/pretrained/convnextv2_tiny_22k_384_ema.pt', map_location=self.device)
        self.map_conv = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.map_conv.classifier[2] = nn.Identity()

        # self.pov_conv = torch.load(f'{self.args.path}/pretrained/convnextv2_tiny_22k_384_ema.pt', map_location=self.device)
        self.pov_conv = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        self.pov_conv.classifier[2] = nn.Identity()
        
    def embed_map(self, map_tile: torch.Tensor) -> torch.Tensor: 
        return self.map_conv(map_tile)
    
    def embed_pov(self, pov_tile: torch.Tensor): 
        image_features = self.pov_conv(pov_tile)
        return image_features


class FullModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        
        self.feat_extractor = ConvNextExtractor()
        self.encoder = GraphSAGE(in_channels=768, hidden_channels=256, num_layers=2, out_channels=64)

        self.augmentor = v2.Compose([#v2.RandomResizedCrop(size=(224, 224), antialias=True), # v2.RandomHorizontalFlip(p=0.1), # Only add these once models are working
                                    v2.ToDtype(torch.float32), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.val_process = v2.Compose([#v2.Resize(size=(224, 224), antialias=True), 
                                    v2.ToDtype(torch.float32), v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.loss_function = losses.TripletMarginLoss(margin=0.1) # NTXent?

        self.batch_size = self.hparams['args'].batch_size
        self.lr = self.hparams['args'].lr
        self.prepare_data()  

        self.current_val_loss = 1000000
        
        self.train_loss, self.val_loss = [], []
        self.train_a, self.train_b, self.val_a, self.val_b = [], [], [], []
        self.train_pointers, self.val_pointers, self.test_pointers = [], [], []
        self.level_of_distance = -1
        self.gt_ori_train, self.est_ori_train = [], []
        self.gt_ori_val, self.est_ori_val = [], []

        self.test_loss = []
        self.test_a, self.test_b = [], []
        self.gt_ori_test, self.est_ori_test = [], []
        

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, img='sat') -> torch.Tensor:
        if img == 'sat': x_feat = self.feat_extractor.embed_map(map_tile=x)
        elif img == 'pov': x_feat = self.feat_extractor.embed_pov(pov_tile=x)

        x = self.encoder(x=x_feat, edge_index=edge_index) # GNN
        x = F.normalize(x, p=2, dim=1)
        return x

    def prepare_data(self):
        data = GraphData(self.hparams['args'])
        self.train_dataset = GraphDataset(self.hparams['args'], data, stage='train')
        # self.val_dataset = GraphDataset(self.hparams, data, stage='val')
        self.test_dataset = GraphDataset(self.hparams['args'], data, stage='test')

    def train_dataloader(self): 
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True if not self.hparams['args'].debug else False)#, sampler=self.train_sampler)

    def val_dataloader(self): 
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)#, sampler=self.val_sampler)

    def test_dataloader(self): 
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)#, sampler=self.test_sampler)

    def triplet_mining(self, batch, z_a, z_b):
        embeddings = torch.cat((z_a.float(), z_b.float()), dim=0)
        emb_length = z_a.shape[0]
        anchors = torch.arange(0, emb_length)
        positives = torch.arange(emb_length, emb_length*2)

        if self.hparams['args'].triplet_mine:
            start_points = batch['pos'].cpu().numpy()
            start_points = np.radians(start_points[batch['ptr'].cpu().numpy()[:-1]])
            matrix = haversine_distances(start_points)
            ordered_indices = np.argsort(matrix, axis=1)
            negatives = torch.tensor(ordered_indices[:, self.level_of_distance])
            negatives = torch.add(negatives, emb_length)
            negatives = torch.repeat_interleave(negatives, self.hparams['args'].walk)
        else:
            negatives = torch.add(torch.randint(0, emb_length, (emb_length,)), emb_length)#.repeat_interleave(self.hparams['args'].walk)
            while torch.any(negatives == torch.arange(emb_length, emb_length*2)):
                negatives = torch.add(torch.randint(0, emb_length, (emb_length,)), emb_length)

        return embeddings, anchors, positives, negatives
    
    def walk_step(self, batch, batch_idx, stage='train'):
        x_sat, ei_sat = batch['sat_image'], batch['edge_index']
        x_pov, ei_pov = batch['pov_image'], batch['edge_index']
        yaws = batch['yaws_image']

        if stage == 'train': x_pov, x_sat = self.augmentor(x_pov), self.augmentor(x_sat)
        else: x_pov, x_sat = self.val_process(x_pov), self.val_process(x_sat)

        z_a = self.forward(x=x_pov, edge_index=ei_pov, img='pov')
        z_b = self.forward(x=x_sat, edge_index=ei_sat, img='sat')
        embeddings, anchors, positives, negatives = self.triplet_mining(batch=batch, z_a=z_a, z_b=z_b) 
        loss = self.loss_function(embeddings=embeddings.float(), indices_tuple=(anchors, positives, negatives))

        self.log(f'{stage}_loss', loss.item(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)

        with torch.no_grad():
            pointers = batch['ptr'].cpu().numpy()[:-1]
            yaws = batch['yaws_image'].cpu().numpy()[pointers]
            batch = batch.to_data_list()
            z_a = z_a.cpu().numpy()[pointers]
            z_b = z_b.cpu().numpy()[pointers]
            
            if stage == 'test':
                self.test_a.append(z_a)
                self.test_b.append(z_b)
                self.gt_ori_test.append(yaws)
                self.test_loss.append(loss.item())
            else:
                self.gt_ori_train.append(yaws) if stage == 'train' else self.gt_ori_val.append(yaws)
                self.train_a.append(z_a) if stage == 'train' else self.val_a.append(z_a)
                self.train_b.append(z_b) if stage == 'train' else self.val_b.append(z_b)
                self.train_loss.append(loss.item()) if stage == 'train' else self.val_loss.append(loss.item())
        return loss

    def training_step(self, batch, batch_idx): 
        loss = self.walk_step(batch=batch, batch_idx=batch_idx, stage='train')
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.walk_step(batch=batch, batch_idx=batch_idx, stage='val')
        return {'loss': loss}
    
    def test_step(self, batch, batch_idx):
        loss = self.walk_step(batch=batch, batch_idx=batch_idx, stage='test')
        return {'loss': loss}

    def on_train_epoch_end(self): 
        epoch_loss = sum(self.train_loss)/len(self.train_loss)
        self.log(f'train_epoch_loss', epoch_loss, prog_bar=True, sync_dist=True)
        if self.current_epoch % self.hparams['args'].acc_interval == 0:    
            # self.log_metrics(stage='train') # Too slow with such large arrays
            self.level_of_distance = min((self.current_epoch // 10) + 1, self.batch_size-1) * -1
        self.train_loss.clear(), self.train_a.clear(), self.train_b.clear(), self.est_ori_train.clear(), self.gt_ori_train.clear()

    def on_validation_epoch_end(self):
        epoch_loss = sum(self.val_loss)/len(self.val_loss)
        if epoch_loss < self.current_val_loss: self.current_val_loss = epoch_loss
        self.log(f'val_epoch_loss', epoch_loss, prog_bar=True, sync_dist=True)
        self.log_metrics(stage='val')
        self.val_loss.clear(), self.val_a.clear(), self.val_b.clear(), self.est_ori_val.clear(), self.gt_ori_val.clear()

    def on_test_epoch_end(self):
        epoch_loss = sum(self.test_loss)/len(self.test_loss)
        self.log(f'test_epoch_loss', epoch_loss, prog_bar=True, sync_dist=True)
        self.log_metrics(stage='test')
        self.test_loss.clear(), self.test_a.clear(), self.test_b.clear(), self.est_ori_test.clear(), self.gt_ori_test.clear()

    def log_metrics(self, stage='train'):
        if stage == 'train': accs = recall_accuracy(emb_a=self.train_a, emb_b=self.train_b, gt_ori=self.gt_ori_train)
        elif stage == 'val': accs = recall_accuracy(emb_a=self.val_a, emb_b=self.val_b, gt_ori=self.gt_ori_val)
        else: accs = recall_accuracy(emb_a=self.test_a, emb_b=self.test_b, gt_ori=self.gt_ori_test)

        if not self.hparams['args'].debug:
            for metric in accs.keys():
                print(f'metric: {metric}, acc: {accs[metric]}')
                self.log(f'{stage}_top_{metric}', accs[metric], prog_bar=True, logger=True, sync_dist=True)
        else:
            print(f'{stage} metrics')
            for k, v in accs.items():
                print(f'{k}: {v}')

    def configure_optimizers(self): 
        opt = torch.optim.AdamW(params=self.parameters(), lr=self.lr) #if self.hparams.gnn else torch.optim.AdamW(params=self.further_encoder.parameters(), lr=self.args.lr)
        sch = ReduceLROnPlateau(optimizer=opt, mode='min', factor=0.5, patience=4, verbose=True)
        return [opt], [{"scheduler": sch, "interval": "epoch", "monitor": "train_epoch_loss"}]

    
