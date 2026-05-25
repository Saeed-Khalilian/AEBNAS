from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader
import numpy as np
import torch
import torch.nn as nn
import copy
from utils import get_correlation

class GNN_Surrogate(nn.Module):
    def __init__(self, num_node_features=18, hidden_dim=100, output_dimension=100):
        super(GNN_Surrogate, self).__init__()
        self.mlp1 = nn.Sequential(
                nn.Linear(num_node_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU())
        self.conv1 = GINConv(self.mlp1)

        self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU())
        self.conv2 = GINConv(self.mlp2)

        self.readout = nn.Linear(hidden_dim, output_dimension)
        # Concatenate pooled GIN representation with input resolution scalar.
        predictor_input_dim = output_dimension + 1
        
        self.accuracy_predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.complexity_predictor = nn.Sequential(
            nn.Linear(predictor_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, batch, input_resolution):
        """Assumes that hte input resolution is normalized to [0,1]"""
        x = self.conv1(x=x, edge_index=edge_index)
        x = self.conv2(x=x, edge_index=edge_index)
        x = global_add_pool(x, batch)
        x = self.readout(x)

        if input_resolution.dim() == 1:
            input_resolution = input_resolution.unsqueeze(1)

        x = torch.cat((x, input_resolution), dim=1)
        predicted_accuracy = self.accuracy_predictor(x)
        predicted_complexity = self.complexity_predictor(x)
        return predicted_accuracy, predicted_complexity

from acc_predictor.architecture_transformer import ArchitectureToGraphEncoder
class GIN:
    """ GIN """
    def __init__(self, arch_encoder_kwargs=None, **kwargs):
        arch_encoder_kwargs = arch_encoder_kwargs or {}
        self.model = GNN_Surrogate(**kwargs)
        self.name = 'gin'
        self.arch_encoder = ArchitectureToGraphEncoder(**arch_encoder_kwargs)

    def fit(self, x, y, **kwargs):
        train_graphs, input_resolutions = self.arch_encoder.build_graph_dataset(x, y)
        for graph, input_resolution in zip(train_graphs, input_resolutions):
            graph.input_resolution = torch.tensor([[input_resolution]], dtype=torch.float32)
        self.model, self.target_mean, self.target_std = train(self.model, train_graphs, **kwargs)

    def predict(self, test_data, device='cpu'):
        query_graphs, input_resolutions = self.arch_encoder.build_graph_dataset(test_data)
        for graph, input_resolution in zip(query_graphs, input_resolutions):
            graph.input_resolution = torch.tensor([[input_resolution]], dtype=torch.float32)
        preds = predict(self.model, query_graphs, device=device)
        if hasattr(self, 'target_mean') and self.target_mean is not None:
            preds = preds * self.target_std + self.target_mean
        return preds


def train(net, graph_data, trn_split=0.8, pretrained=None, device='cpu',
          lr=3e-4, epochs=500, verbose=False):
    n_samples = len(graph_data)
    if n_samples == 0:
        raise ValueError("Training set is empty")

    perm = torch.randperm(n_samples)
    trn_idx = perm[:int(n_samples * trn_split)]
    vld_idx = perm[int(n_samples * trn_split):]

    if len(trn_idx) == 0:
        trn_idx = perm
    if len(vld_idx) == 0:
        vld_idx = trn_idx

    all_targets = torch.stack([data.y for data in graph_data])
    target_mean = all_targets[trn_idx].mean(dim=0)
    target_std = all_targets[trn_idx].std(dim=0) + 1e-8

    for i in trn_idx.tolist():
        graph_data[i].y = (graph_data[i].y - target_mean) / target_std

    trn_data = [graph_data[i] for i in trn_idx.tolist()]
    vld_data = [graph_data[i] for i in vld_idx.tolist()]

    trn_loader = DataLoader(trn_data, batch_size=min(16, len(trn_data)), shuffle=True)
    vld_loader = DataLoader(vld_data, batch_size=min(16, len(vld_data)), shuffle=False)

    # back-propagation training of a NN
    if pretrained is not None:
        print("Constructing MLP surrogate model with pre-trained weights")
        init = torch.load(pretrained, map_location='cpu')
        net.load_state_dict(init)
        best_net = copy.deepcopy(net)
    else:
        # print("Constructing MLP surrogate model with "
        #       "sample size = {}, epochs = {}".format(x.shape[0], epochs))

        # initialize the weights
        # net.apply(Net.init_weights)
        net = net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        criterion = nn.SmoothL1Loss()
        # criterion = nn.MSELoss()

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epochs), eta_min=0)

        best_loss = 1e33
        for epoch in range(epochs):
            loss_trn = train_one_epoch(net, trn_loader, criterion, optimizer, device)
            loss_vld = infer(net, vld_loader, criterion, device, target_mean, target_std)
            scheduler.step()
            #print("loop")
            # if epoch % 500 == 0 and verbose:
            #     print("Epoch {:4d}: trn loss = {:.4E}, vld loss = {:.4E}".format(epoch, loss_trn, loss_vld))

            if loss_vld < best_loss:
                best_loss = loss_vld
                best_net = copy.deepcopy(net)

    validate(best_net, vld_loader, device=device, target_mean=target_mean, target_std=target_std)

    return best_net.to('cpu'), target_mean.cpu().numpy(), target_std.cpu().numpy()

def train_one_epoch(net, loader, criterion, optimizer, device):
    net.train()
    running_loss = 0.0
    n_batches = 0

    for batch in loader:
        optimizer.zero_grad()

        batch = batch.to(device)
        pred_acc, pred_complex = net(batch.x, batch.edge_index, batch.batch, batch.input_resolution)
        pred = torch.cat((pred_acc, pred_complex), dim=1)
        target = batch.y.view_as(pred)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)

def infer(net, loader, criterion, device, target_mean, target_std):
    net.eval()
    running_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_acc, pred_complex = net(batch.x, batch.edge_index, batch.batch, batch.input_resolution)
            scaled_pred = torch.cat((pred_acc, pred_complex), dim=1)
            
            unscaled_pred = scaled_pred * target_std.to(device) + target_mean.to(device)
            unscaled_target = batch.y.view_as(unscaled_pred)
            
            loss = criterion(unscaled_pred, unscaled_target)
            running_loss += loss.item()
            n_batches += 1

    return running_loss / max(n_batches, 1)

def validate(net, loader, device, target_mean, target_std):
    net.eval()

    with torch.no_grad():
        pred_list, target_list = [], []
        for batch in loader:
            batch = batch.to(device)
            pred_acc, pred_complex = net(batch.x, batch.edge_index, batch.batch, batch.input_resolution)
            scaled_pred = torch.cat((pred_acc, pred_complex), dim=1)
            
            unscaled_pred = scaled_pred * target_std.to(device) + target_mean.to(device)
            unscaled_target = batch.y.view_as(unscaled_pred)
            
            pred_list.append(unscaled_pred.cpu())
            target_list.append(unscaled_target.cpu())

        pred = torch.cat(pred_list, dim=0).detach().numpy()
        target = torch.cat(target_list, dim=0).detach().numpy()

        rmse, rho, tau = get_correlation(pred, target)

    # print("Validation RMSE = {:.4f}, Spearman's Rho = {:.4f}, Kendall’s Tau = {:.4f}".format(rmse, rho, tau))
    return rmse, rho, tau, pred, target

def predict(net, query, device):
    if len(query) == 0:
        return np.empty((0, 2), dtype=np.float32)

    loader = DataLoader(query, batch_size=min(64, len(query)), shuffle=False)

    net = net.to(device)
    net.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_acc, pred_complex = net(batch.x, batch.edge_index, batch.batch, batch.input_resolution)
            pred = torch.cat((pred_acc, pred_complex), dim=1)
            preds.append(pred.cpu())

    return torch.cat(preds, dim=0).detach().numpy()