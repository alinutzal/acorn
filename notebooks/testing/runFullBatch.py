# %% [markdown]
# # Edge Classifier Debug

# %% [markdown]
# **Goal**: Test the weighting and hard cut config of the data loading process

# %%

import os
import yaml

import numpy as np
import pandas as pd
import seaborn as sns
import yaml
import torch
import matplotlib.pyplot as plt
from time import time as tt
from torch_geometric.data import Data, HeteroData
import torchmetrics

# %% [markdown]
# ## GNN Debug

# %%
from gnn4itk_cf.stages.edge_classifier.models.interaction_gnn import InteractionGNN
import wandb
from sklearn.metrics import roc_auc_score

# %%
config = yaml.load(open('gnn_train.yaml'), Loader=yaml.FullLoader)
model = InteractionGNN(config)
model.setup('fit')

# %%

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(device)

# %%
optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=(model.hparams["lr"]),
		betas=(0.9, 0.999),
		eps=1e-08,
		amsgrad=True,
	)

scheduler = {
			"scheduler": torch.optim.lr_scheduler.StepLR(
				optimizer,
				step_size=model.hparams["patience"],
				gamma=model.hparams["factor"],
			),
			"interval": "epoch",
			"frequency": 1,
		}
	

# %%
num_epochs=model.hparams["max_epochs"]
start = tt()
run = wandb.init(project=model.hparams["project"], entity='gnnproject')

for epoch in range(num_epochs):
    

    model.train()
    
    if (model.hparams["warmup"] is not None) and (epoch < model.hparams["warmup"]):
        lr_scale = min(1.0, float(epoch + 1) / model.hparams["warmup"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr_scale * model.hparams["lr"]
    
    for batch_idx, data in enumerate(model.trainset):
        model.train()

        
        
        
        data = data.to(device)  # Move the batch of graph data to the device
        
        
        ### FORWARD AND BACK PROP
        
        logits = model(data)
        loss = model.loss_function(logits, data)
        
        # update params
        
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        
        
        ### LOGGING
        #if not batch_idx % 300:
        print(f"Epoch: {epoch+1:04d}/{num_epochs:04d} | Batch {batch_idx:04d}/{len(model.trainset):04d} | Loss: {loss:.4f}")

        if (batch_idx+1 % len(model.trainset) == 0):
            with torch.no_grad():
                ### W&B LOGGING ###
                all_truth = data.y.bool()
                target_truth = (data.weights > 0) & all_truth

                preds = torch.sigmoid(logits) > model.hparams["edge_cut"]

                # Positives
                edge_positive = preds.sum().float()

                # Signal true & signal tp
                target_true = target_truth.sum().float()
                target_true_positive = (target_truth.bool() & preds).sum().float()
                all_true_positive = (all_truth.bool() & preds).sum().float()
                target_auc = roc_auc_score(
                    target_truth.bool().cpu().detach(), torch.sigmoid(logits).cpu().detach()
                )

                # Eff, pur, auc
                target_eff = target_true_positive / target_true
                target_pur = target_true_positive / edge_positive
                total_pur = all_true_positive / edge_positive
                current_lr = optimizer.param_groups[0]['lr']

            
    ### MORE LOGGING
    model.eval()
    
    with torch.no_grad():
        val_loss = []
        for data in model.valset:
            data = data.to(device)  # Move the batch of graph data to the device
            outputs = model(data)
            val_loss.append(model.loss_function(outputs, data).item())
            
        avg_loss = sum(val_loss) / len(val_loss)
        run.log({
            "train_loss": loss,
            "current_lr": current_lr,
            "eff": target_eff,
            "target_pur": target_pur,
            "total_pur": total_pur,
            "auc": target_auc,
            "val_loss": avg_loss,
            "epoch": epoch,
            "trainer/global_step": len(model.trainset) + epoch*len(model.trainset)
        }, step=len(model.trainset) + epoch*len(model.trainset))

        print(f"Epoch: {epoch+1:04d}/{num_epochs:04d}")
    
        
    scheduler['scheduler'].step()
    
        
run.finish()
end = tt()
elapsed = end-start
print(f"Time elapsed {elapsed/60:.2f} min")
print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

# %%
model.eval()
with torch.no_grad():
    test_acc = torchmetrics.Accuracy(task="binary").to(device)

    for data in model.testset:
        data = data.to(device)  # Move the batch of graph data to the device
        outputs = model(data)
        test_acc.update(outputs, data.y.int())

    print(f"Test acc.: {test_acc.compute()*100:.2f}%")
    test_acc.reset()
