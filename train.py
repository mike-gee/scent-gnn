import pyrfume
import torch
from torch_geometric.loader import DataLoader

from dataset import construct_dataset, get_balanced_sampler
from model import GCN, evaluate, test, train


def train_model():
    torch.manual_seed(888)

    molecules = pyrfume.load_data("leffingwell/molecules.csv", remote=True)
    behavior = pyrfume.load_data("leffingwell/behavior.csv", remote=True)
    behavior = behavior.set_index("IsomericSMILES")

    # Filter out top 20 classes by frequency
    top_scents = behavior.sum(axis=0).sort_values(ascending=False).head(20)
    behav_sub = behavior[top_scents.index]
    train_dataset, test_dataset = construct_dataset(behav_sub)
    train_sampler = get_balanced_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=32)

    num_node_features = len(train_dataset[0].x[0])
    out_size = train_dataset[0].y.shape[1]
    model = GCN(
        num_node_features=num_node_features, hidden_channels=128, out_size=out_size
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    for epoch in range(0, 301):
        train(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if epoch % 5 == 0:
            pred_y, actual_y = evaluate(model, train_loader)
            train_df = get_eval_metrics(actual_y, pred_y, top_scents.index)
            train_recall = train_df.recall.train()
            train_prec = train_df.precision.train()
            pred_y, actual_y = evaluate(model, test_loader)
            test_df = get_eval_metrics(actual_y, pred_y, top_scents.index)
            test_recall = test_df.recall.test()
            test_prec = test_df.precision.test()
            print(
                f"Epoch: {epoch:03d}, Train Recall {train_recall:.4f}, Test Recall {test_recall:.4f},  Train Prec: {train_prec:.4f}, Test Prec: {test_prec:.4f}"
            )

            torch.save(model.state_dict(), f"model_{epoch}.pt")
