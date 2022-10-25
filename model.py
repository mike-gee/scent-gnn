from typing import Tuple

import torch
import torch.nn.functional as F
from torch.nn import SELU, BCELoss, Linear, Sigmoid
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.nn import TransformerConv, global_max_pool


class GCN(torch.nn.Module):
    """Initialize the selected Graph Convolutional Network architecture."""

    def __init__(
        self, num_node_features: int, hidden_channels: int, out_size: int
    ) -> None:
        """Initialize GCN Class

        Args:
            num_node_features: number of input features.
            hidden_channels: number of hidden channels.
            out_size: number of output classes.
        """
        super(GCN, self).__init__()
        self.conv1 = TransformerConv(num_node_features, hidden_channels)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels)
        self.conv3 = TransformerConv(hidden_channels, hidden_channels)
        self.conv4 = TransformerConv(hidden_channels, hidden_channels)
        self.conv5 = TransformerConv(hidden_channels, hidden_channels)
        self.conv6 = TransformerConv(hidden_channels, hidden_channels)
        self.conv7 = TransformerConv(hidden_channels, hidden_channels)
        self.conv8 = TransformerConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_size)
        self.activation = SELU()
        self.out_activation = Sigmoid()

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """Forward operation for Module."""
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.conv3(x, edge_index)
        x = self.activation(x)
        x = self.conv4(x, edge_index)
        x = self.activation(x)
        x = self.conv5(x, edge_index)
        x = self.activation(x)
        x = self.conv6(x, edge_index)
        x = self.activation(x)
        x = self.conv7(x, edge_index)
        x = self.activation(x)
        x = self.conv8(x, edge_index)
        x = global_max_pool(x, batch)
        x = self.lin(x)
        x = self.out_activation(x)
        return x


def train(
    model: GCN,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: Adam,
    scheduler: StepLR,
) -> None:
    """Run one epoch of training for model.

    Args:
        model: GCN model.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Scheduler for learning rate optimization.
    """
    model.train()
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()


def evaluate(model: GCN, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate model performance using dataloader.

    Args:
        model: GCN model.
        loader: Evaluation DataLoader.

    Returns:
        (Predicted class, ground truth class)
    """
    pred_y = []
    actual_y = []
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            out = out.clone().detach()
            out[out >= 0.5] = 1
            out[out < 0.5] = 0
            pred_y.append(out)
            actual_y.append(data.y)
    pred_y = torch.cat(pred_y)
    actual_y = torch.cat(actual_y)
    return pred_y, actual_y
