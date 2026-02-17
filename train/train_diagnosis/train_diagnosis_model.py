import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset, random_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopper:
    def __init__(self, num_trials: int, save_path: str) -> None:
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_loss = float("inf")
        self.save_path = save_path

    def is_continuable(self, model: nn.Module, loss: float) -> bool:
        if loss < self.best_loss:
            self.best_loss = loss
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            return True
        if self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        return False


class FeedForward(nn.Module):
    def __init__(self, d_model: int, inner_size: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class MambaLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model * 4, dropout=dropout)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.mamba(input_tensor)
        if self.num_layers == 1:
            hidden_states = self.layer_norm(self.dropout(hidden_states))
        else:
            hidden_states = self.layer_norm(self.dropout(hidden_states) + input_tensor)
        hidden_states = self.ffn(hidden_states)
        return hidden_states


class Net(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.cnn = nn.Conv1d(12, 12, 3, 1)
        self.mamba_layer = MambaLayer(
            d_model=12,
            d_state=16,
            d_conv=4,
            expand=2,
            dropout=0.2,
            num_layers=1,
        )
        self.linear1 = nn.Linear(24, 12)
        self.linear2 = nn.Linear(12, 20)
        self.linear3 = nn.Linear(20, 1)
        self.sig = nn.Sigmoid()

        nn.init.kaiming_uniform_(self.linear1.weight, nonlinearity="leaky_relu")
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.constant_(self.linear2.bias, 0)
        nn.init.constant_(self.linear3.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fft = torch.fft.fft(x, dim=1)
        x_fft_abs = torch.sqrt(x_fft.real ** 2 + x_fft.imag ** 2)

        x_mamba = self.mamba_layer(x)

        x_cnn = self.cnn(x_fft_abs.transpose(1, 2))
        x_cnn = x_cnn.transpose(1, 2)

        x_combined = torch.cat((x_mamba, x_cnn), dim=2)
        x = self.linear1(x_combined)
        x = F.elu(x, alpha=1.0)
        x = self.linear2(x)
        x = self.linear3(x)
        x = x.mean(dim=1)
        x = self.sig(x).squeeze()
        return x


def build_loaders(
    abnormal_path: str,
    normal_path: str,
    max_len: int,
    split_ratio: float,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    abnormal = torch.load(abnormal_path)
    abnormal = abnormal[:, :max_len, :]
    label0 = torch.zeros([abnormal.shape[0]])

    normal = torch.load(normal_path)
    normal = normal[:, :max_len, :]
    label1 = torch.ones([normal.shape[0]])

    raw_data = torch.cat((normal, abnormal), dim=0)
    targets = torch.cat((label1, label0))

    dataset = TensorDataset(raw_data, targets)
    len_train = len(dataset)
    split_num = [int(len_train * split_ratio), len_train - int(len_train * split_ratio)]

    train_data, val_data = random_split(
        dataset=dataset,
        lengths=split_num,
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def cal_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float, float]:
    model.eval()
    pres = []
    labels = []
    with torch.no_grad():
        for field, label in loader:
            field = field.float().to(device)
            label = label.float().to(device)
            prediction = model(field)
            pres.append(prediction)
            labels.append(label)

    pres_t = torch.cat(pres, dim=0)
    labels_t = torch.cat(labels, dim=0)
    loss = criterion(pres_t, labels_t).item()

    pres_bin = (pres_t > 0.5).float()
    labels_np = labels_t.cpu().numpy()
    pres_np = pres_bin.cpu().numpy()

    score1 = recall_score(labels_np, pres_np)
    score2 = f1_score(labels_np, pres_np)
    score3 = precision_score(labels_np, pres_np)
    score4 = accuracy_score(labels_np, pres_np)
    model.train()
    return loss, score1, score2, score3, score4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FFT Mamba+CNN classifier")
    parser.add_argument("--abnormal", default="abnormal_2000.dat", help="Path to abnormal tensor")
    parser.add_argument("--normal", default="normal_2000.dat", help="Path to normal tensor")
    parser.add_argument("--max-len", type=int, default=2500, help="Max time length")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--save", default="model-mamba-cnn.pt", help="Path to save best model")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--resume", action="store_true", help="Resume from --save if exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_loaders(
        abnormal_path=args.abnormal,
        normal_path=args.normal,
        max_len=args.max_len,
        split_ratio=args.split_ratio,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model = Net(12).to(device)
    if args.resume and os.path.exists(args.save):
        model.load_state_dict(torch.load(args.save, map_location=device))

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    early_stopper = EarlyStopper(args.num_trials, args.save)

    for epoch in range(args.epochs):
        for field, label in train_loader:
            field = field.float().to(device)
            label = label.float().to(device)

            prediction = model(field)
            loss = criterion(prediction, label)

            model.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        val_loss, score1, score2, score3, score4 = cal_loss(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}: val_loss={val_loss:.6f}")
        print(f"recall: {score1:.6f}")
        print(f"f1: {score2:.6f}")
        print(f"precision: {score3:.6f}")
        print(f"accuracy: {score4:.6f}")
        print("---" * 10)

        if not early_stopper.is_continuable(model, val_loss):
            break


if __name__ == "__main__":
    main()
