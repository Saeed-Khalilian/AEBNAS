import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils import get_correlation

import warnings
warnings.filterwarnings("ignore", message=".*nested tensors is in prototype stage.*")

class TransformerSurrogate_Split(nn.Module):
    def __init__(self, input_dim=18, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, max_seq_len=45):
        super(TransformerSurrogate_Split, self).__init__()

        if max_seq_len < 1:
            raise ValueError("max_seq_len must be at least 1")
        
       
        self.embedding = nn.Linear(input_dim, d_model)
        
        #[CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        #Positional Encoding Learned
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len + 1, d_model))
        
        #transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # LayerNorm on CLS output before heads (ViT-style)
        self.cls_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

        # input_dim + 1 because we concatenate Input Resolution
        predictor_input_dim = d_model + 1  # +1 for resolution scalar

        self.final_mlp = nn.Sequential(
            nn.Linear(predictor_input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1)
        )
        
        # self.mlp_macs = nn.Sequential(
        #     nn.Linear(predictor_input_dim, 64),
        #     nn.LayerNorm(64),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(64, 1)
        # )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, resolution, src_key_padding_mask=None):
        """
        x: (batch_size, seq_len, 18) - Node features
        resolution: (batch_size, 1) - Normalized image resolution
        src_key_padding_mask: (batch_size, seq_len + 1) - Mask for padded nodes
        """
        batch_size = x.size(0)

        if resolution.dim() == 1:
            resolution = resolution.unsqueeze(1)
        
        x = self.embedding(x) # (batch_size, seq_len, d_model)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) #  (batch_size, seq_len+1, d_model)
        
        # Add Positional Embedding
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # Output shape:  (batch_size, seq_len+1, d_model)
        features = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Extract [CLS] representation
        cls_out = features[:, 0, :] # (batch_size, d_model)
        cls_out = self.cls_norm(cls_out)
        cls_out = self.dropout(cls_out)
        
        # Concatenate resolution
        combined = torch.cat([cls_out, resolution], dim=1) # (batch_size, d_model + 1)
        
        # Predict metrics
        pred = self.final_mlp(combined)
        # macs_pred = self.mlp_macs(combined)
        
        return pred
    
from acc_predictor.architecture_transformer import ArchitectureToGraphEncoder
class Transformer_Split:
    """ Transformer """
    def __init__(self, arch_encoder_kwargs=None, max_sequence_length=None, **kwargs):
        arch_encoder_kwargs = arch_encoder_kwargs or {}
        self.arch_encoder = ArchitectureToGraphEncoder(**arch_encoder_kwargs)
        sequence_length = max_sequence_length or self.arch_encoder.max_sequence_length()
        self.model = TransformerSurrogate_Split(max_seq_len=sequence_length, **kwargs)
        self.name = 'transformer_split'

    def fit(self, x, y, **kwargs):
        train_sequences, input_resolutions, padding_masks, targets = self.arch_encoder.build_sequence_dataset(x, y)
        device = kwargs.pop('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = train(
            self.model,
            train_sequences,
            input_resolutions,
            padding_masks,
            targets,
            device=device,
            **kwargs,
        )

    def predict(self, test_data, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        query_sequences, input_resolutions, padding_masks, _ = self.arch_encoder.build_sequence_dataset(test_data)
        preds = predict(self.model, query_sequences, input_resolutions, padding_masks, device=device)
        # if hasattr(self, 'target_mean') and self.target_mean is not None:
        #     preds = preds * self.target_std + self.target_mean
        return preds


def train(net, sequences, input_resolutions, padding_masks, targets, trn_split=0.8, pretrained=None, device='cpu',
          lr=1e-3, epochs=300, verbose=False):
    n_samples = len(sequences)
    if n_samples == 0:
        raise ValueError("Training set is empty")

    perm = torch.randperm(n_samples)
    trn_idx = perm[:int(n_samples * trn_split)]
    vld_idx = perm[int(n_samples * trn_split):]

    if len(trn_idx) == 0:
        trn_idx = perm
    if len(vld_idx) == 0:
        vld_idx = trn_idx

    # target_mean = targets[trn_idx].mean(dim=0)
    # target_std = targets[trn_idx].std(dim=0) + 1e-8

    # targets_scaled = (targets - target_mean) / target_std

    trn_data = TensorDataset(sequences[trn_idx], input_resolutions[trn_idx], padding_masks[trn_idx], targets[trn_idx])
    vld_data = TensorDataset(sequences[vld_idx], input_resolutions[vld_idx], padding_masks[vld_idx], targets[vld_idx])

    trn_loader = DataLoader(trn_data, batch_size=min(16, len(trn_data)), shuffle=True)
    vld_loader = DataLoader(vld_data, batch_size=min(16, len(vld_data)), shuffle=False)

    if pretrained is not None:
        print("Constructing Transformer surrogate model with pre-trained weights")
        init = torch.load(pretrained, map_location='cpu')
        net.load_state_dict(init)
        net = net.to(device)
        best_net = copy.deepcopy(net)
    else:
        net = net.to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
        criterion = nn.SmoothL1Loss()
        # Warmup for first 10% epochs, then cosine decay
        warmup_epochs = epochs // 10
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 1.0
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-5)

        best_loss = 1e33
        for epoch in range(epochs):
            loss_trn = train_one_epoch(net, trn_loader, criterion, optimizer, device)
            loss_vld = infer(net, vld_loader, criterion, device)
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                scheduler.step()

            if loss_vld < best_loss:
                best_loss = loss_vld
                best_net = copy.deepcopy(net)

    validate(best_net, vld_loader, device=device)

    return best_net.to('cpu')


def train_one_epoch(net, loader, criterion, optimizer, device):
    net.train()
    running_loss = 0.0
    n_batches = 0

    for batch in loader:
        optimizer.zero_grad()

        batch = [item.to(device) for item in batch]
        seq, resolution, mask, target = batch
        pred= net(seq, resolution, src_key_padding_mask=mask)
        # pred = torch.cat((pred_acc, pred_complex), dim=1)
        target = target.view_as(pred)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


def infer(net, loader, criterion, device):
    net.eval()
    running_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = [item.to(device) for item in batch]
            seq, resolution, mask, target = batch
            pred= net(seq, resolution, src_key_padding_mask=mask)
            # pred = torch.cat((pred_acc, pred_complex), dim=1)
            
            loss = criterion(pred, target.view_as(pred))
            running_loss += loss.item()
            n_batches += 1

    return running_loss / max(n_batches, 1)


def validate(net, loader, device):
    net.eval()

    with torch.no_grad():
        pred_list, target_list = [], []
        for batch in loader:
            batch = [item.to(device) for item in batch]
            seq, resolution, mask, target = batch
            pred = net(seq, resolution, src_key_padding_mask=mask)
            target = target.view_as(pred)
            
            pred_list.append(pred.cpu())
            target_list.append(target.cpu())

        pred = torch.cat(pred_list, dim=0).detach().numpy()
        target = torch.cat(target_list, dim=0).detach().numpy()

        rmse_acc, rho_acc, tau_acc = get_correlation(pred[:, 0], target[:, 0])

    return rmse_acc, rho_acc, tau_acc, pred, target


def predict(net, sequences, input_resolutions, padding_masks, device):
    if len(sequences) == 0:
        return np.empty((0, 1), dtype=np.float32)

    loader = DataLoader(TensorDataset(sequences, input_resolutions, padding_masks), batch_size=min(64, len(sequences)), shuffle=False)

    net = net.to(device)
    net.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            seq, resolution, mask = [item.to(device) for item in batch]
            pred = net(seq, resolution, src_key_padding_mask=mask)
            preds.append(pred.cpu())

    return torch.cat(preds, dim=0).detach().numpy()
