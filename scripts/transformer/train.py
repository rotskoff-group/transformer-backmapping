import torch
import torch.nn as nn
from utils import SequenceDataset, Transformer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
import argparse


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--dim_model", type=int, default=512)
parser.add_argument("--kt_cutoff", type=int, default=10)
config = parser.parse_args()

dropout_p = config.dropout
dim_model = config.dim_model
kt_cutoff = config.kt_cutoff


load_folder_name = f"./ChignolinGMMTransformerDataset/prop_temp_300.0_dt_0.001_num_steps_5_cutoff_to_use_kt_{kt_cutoff}/"

all_src_cont_train = torch.tensor(np.load(f"{load_folder_name}/train_all_src_cont.npy"),
                                  device=device).float()

all_src_cat_train = torch.tensor(np.load(f"{load_folder_name}/train_all_src_cat.npy"),
                                 device=device).long()

all_target_train = torch.tensor(np.load(f"{load_folder_name}/train_all_target.npy"),
                                device=device).long()


all_src_cont_val = torch.tensor(np.load(f"{load_folder_name}/val_all_src_cont.npy"),
                                device=device).float()
all_src_cat_val = torch.tensor(np.load(f"{load_folder_name}/val_all_src_cat.npy"),
                                 device=device).long()
all_target_val = torch.tensor(np.load(f"{load_folder_name}/val_all_target.npy"),
                                device=device).long()



train_dataset = SequenceDataset(all_src_cont=all_src_cont_train,
                                all_src_cat=all_src_cat_train,
                                all_tgt=all_target_train)
train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)


val_dataset = SequenceDataset(all_src_cont=all_src_cont_val,
                              all_src_cat = all_src_cat_val,
                              all_tgt=all_target_val)
val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=False)

t = Transformer(num_tokens_src=7, num_tokens_tgt=67,
                dim_model=dim_model, num_heads=8, num_encoder_layers=6,
                num_decoder_layers=6, dropout_p=dropout_p).to(device)
t_opt = torch.optim.Adam(t.parameters(), lr=1e-6)

loss_fn = nn.CrossEntropyLoss()

tag = f"dropout_p_{dropout_p}_dim_model_{dim_model}_kt_cutoff_{kt_cutoff}/"
writer = SummaryWriter(tag)
num_train_batch = len(train_dataloader)
num_val_batch = len(val_dataloader)

for epoch in range(0, 10000):
    epoch_train_loss = 0

    epoch_train_loss_energy = 0
    epoch_train_loss_sequence = 0

    epoch_val_loss = 0

    epoch_val_loss_energy = 0
    epoch_val_loss_sequence = 0
    t.train()
    for (train_src_cont, train_src_cat, train_y) in train_dataloader:

        y_input = train_y[:, :-1]
        y_expected = train_y[:, 1:]

        sequence_length = y_input.shape[1]
        tgt_mask = t.get_tgt_mask(sequence_length).to(device)
        pred_sequence = t(src_cont=train_src_cont,
                          src_cat = train_src_cat,
                          tgt=y_input, tgt_mask=tgt_mask)

        pred_sequence = pred_sequence.transpose(1, 2)
        loss_sequence = loss_fn(pred_sequence, y_expected)


        loss = loss_sequence

        t_opt.zero_grad()
        loss.backward()
        t_opt.step()
        epoch_train_loss += loss.item()
        epoch_train_loss_energy += 0
        epoch_train_loss_sequence += loss_sequence.item()
    t.eval()
    for (val_src_cont, val_src_cat, val_y) in val_dataloader:

        y_input = val_y[:, :-1]
        y_expected = val_y[:, 1:]

        sequence_length = y_input.shape[1]
        tgt_mask = t.get_tgt_mask(sequence_length).to(device)

        pred_sequence = t(src_cont=val_src_cont,
                          src_cat = val_src_cat,
                          tgt=y_input, tgt_mask=tgt_mask)

        pred_sequence = pred_sequence.transpose(1, 2)
        loss_sequence = loss_fn(pred_sequence, y_expected)

        loss = loss_sequence
        epoch_val_loss += loss.item()
        epoch_val_loss_energy += 0
        epoch_val_loss_sequence += loss_sequence.item()

    writer.add_scalar("Train Loss", epoch_train_loss/num_train_batch, epoch)
    writer.add_scalar("Train Loss Energy",
                      epoch_train_loss_energy/num_train_batch, epoch)
    writer.add_scalar("Train Loss Sequence",
                      epoch_train_loss_sequence/num_train_batch, epoch)

    writer.add_scalar("Val Loss", epoch_val_loss/num_val_batch, epoch)
    writer.add_scalar("Val Loss Energy",
                      epoch_val_loss_energy/num_val_batch, epoch)
    writer.add_scalar("Val Loss Sequence",
                      epoch_val_loss_sequence/num_val_batch, epoch)

    if epoch % 2 == 0:
        torch.save(t.state_dict(), f"{tag}/t_{epoch}.pt")
        torch.save(t_opt.state_dict(), f"{tag}/t_opt_{epoch}.pt")
