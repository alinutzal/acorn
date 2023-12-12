# # Edge Classifier Debug

# **Goal**: Test the weighting and hard cut config of the data loading process

import os
import yaml

import lightning as L
from lightning import Fabric
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
import torch
import matplotlib.pyplot as plt
from time import time as tt
from torch_geometric.data import Data, HeteroData
from torchmetrics.classification import Precision, Recall, PrecisionRecallCurve, ROC, AUROC, Accuracy
from torch_geometric.loader import DataLoader

from gnn4itk_cf.stages.edge_classifier.models.interaction_gnn import InteractionGNN
import wandb
from sklearn.metrics import roc_auc_score

num_workers = 0

def train(num_epochs, model, optimizer, scheduler, train_loader, val_loader, fabric, device="cuda"):

    for epoch in range(num_epochs):
        train_auc = AUROC(task="binary").to(fabric.device)
        train_precision = Precision(task="binary").to(fabric.device)
        train_recall = Recall(task="binary").to(fabric.device)    
        
        model.train()
        
        if (model.hparams["warmup"] is not None) and (epoch < model.hparams["warmup"]):
            lr_scale = min(1.0, float(epoch + 1) / model.hparams["warmup"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * model.hparams["lr"]
        current_lr = optimizer.param_groups[0]['lr']
        
        for batch_idx, data in enumerate(train_loader):
            #data = data.to(device)  # Move the batch of graph data to the device
            
            ### FORWARD AND BACK PROP
            logits = model(data)
            loss = model.loss_function(logits, data)
            # update params
            optimizer.zero_grad()
            #loss.backward()
            fabric.backward(loss)
            optimizer.step()
            
            ### LOGGING
            if ((batch_idx+1) % len(model.trainset) == 0):
                fabric.print(f"Epoch: {epoch+1:04d}/{num_epochs:04d} | Batch {batch_idx:04d}/{len(model.trainset):04d} | Loss: {loss:.4f}")


            # with torch.no_grad():
            #     ### W&B LOGGING ###
            #     all_truth = data.y.bool()
            #     target_truth = (data.weights > 0) & all_truth

            #     preds = torch.sigmoid(logits) > model.hparams["edge_cut"]

                # # Positives
                # edge_positive = preds.sum().float()

                # # Signal true & signal tp
                # target_true = target_truth.sum().float()
                # target_true_positive = (target_truth.bool() & preds).sum().float()
                # all_true_positive = (all_truth.bool() & preds).sum().float()
                # target_auc = roc_auc_score(
                #     target_truth.bool().cpu().detach(), torch.sigmoid(logits).cpu().detach()
                # )
                # # Eff, pur, auc
                # target_eff = target_true_positive / target_true
                # target_pur = target_true_positive / edge_positive
                # total_pur = all_true_positive / edge_positive
            #     current_lr = optimizer.param_groups[0]['lr']
                
            #     train_precision.update(preds, target_truth)
            #     train_recall.update(preds, target_truth)
            #     train_auc.update(preds,target_truth)
        #scheduler['scheduler'].step()  
        scheduler.step()           
    
        ### MORE LOGGING
        model.eval()    
        with torch.no_grad():
            val_auc = AUROC(task="binary").to(fabric.device)
            val_precision = Precision(task="binary").to(fabric.device)
            val_recall = Recall(task="binary").to(fabric.device)    
            val_loss = []
            eff = []
            tar_pur = []
            tot_pur = []
            for data in model.valset:
                data = data.to(device)  # Move the batch of graph data to the device

                outputs = model(data)

                all_truth = data.y.bool()
                target_truth = (data.weights > 0) & all_truth


                preds = torch.sigmoid(outputs) > model.hparams["edge_cut"]
                
                # Positives
                edge_positive = preds.sum().float()

                # Signal true & signal tp
                target_true = target_truth.sum().float()
                target_true_positive = (target_truth.bool() & preds).sum().float()
                all_true_positive = (all_truth.bool() & preds).sum().float()
                target_auc = roc_auc_score(
                    target_truth.bool().cpu().detach(), torch.sigmoid(outputs).cpu().detach()
                )
                # Eff, pur, auc
                target_eff = target_true_positive / target_true
                target_pur = target_true_positive / edge_positive
                total_pur = all_true_positive / edge_positive
                
                val_loss.append(model.loss_function(outputs, data).item())
                eff.append(target_eff.item())
                tar_pur.append(target_pur.item())
                tot_pur.append(total_pur.item())
                val_precision.update(preds, target_truth)
                val_recall.update(preds, target_truth)
                val_auc.update(outputs,target_truth)
                
            avg_loss = sum(val_loss) / len(val_loss)
            target_eff = sum(eff) / len(eff)
            avg_tarpur = sum(tar_pur) / len(tar_pur)
            avg_totpur = sum(tot_pur) / len(tot_pur)
            
            efficiency = val_precision.compute()
            purity = val_recall.compute()
            auc = val_auc.compute()
            run.log({
                "train_loss": loss,
                "current_lr": current_lr,
                "eff": target_eff,
                "target_pur": avg_tarpur,
                "total_pur": avg_totpur,
                "auc": target_auc,
                "val_loss": avg_loss,
                "epoch": epoch,
                "trainer/global_step": len(model.trainset) + epoch*len(model.trainset)
            }, step=len(model.trainset) + epoch*len(model.trainset))

            fabric.print(f"Epoch: {epoch+1:04d}/{num_epochs:04d},{auc:.4f}, {efficiency:.4f}, {purity:.4f}")
            #print(val_auc.compute(), val_precision.compute(), val_recall.compute())
            val_auc.reset(), val_precision.reset(), val_recall.reset()

    
    
if __name__ == "__main__":
    print("Torch CUDA available?", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    
    L.seed_everything(123)

    #########################################
    ### 2 Initializing the Model
    #########################################
    config = yaml.load(open(os.environ['HOME']+'/acorn/examples/Example_3/gnn_train.yaml'), Loader=yaml.FullLoader)
    model = InteractionGNN(config)
    model.setup('fit')
    #model = torch.compile(model)
    model.to(device)
    
    weight_decay = model.hparams.get("lr_weight_decay", 0.01)
    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=(model.hparams["lr"]),
            betas=(0.9, 0.999),
            eps=1e-08,
            amsgrad=True,
            weight_decay=weight_decay,
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
        

    num_epochs=model.hparams["max_epochs"]

    ##########################
    ### 1 Loading the Dataset
    ##########################
    
    train_loader = DataLoader(
                model.trainset, batch_size=1, num_workers=num_workers) #, pin_memory=True, persistent_workers=True
            #)
    val_loader = DataLoader(
                model.valset, batch_size=1, num_workers=num_workers
            )
    test_loader = DataLoader(
                model.testset, batch_size=1, num_workers=num_workers
            )

    #########################################
    ### 3 Launch Fabric
    #########################################

    fabric = Fabric(accelerator="cuda", devices=1) #, strategy="ddp") # , precision="bf16-mixed")
    fabric.launch()

    train_loader, val_loader, test_loader = fabric.setup_dataloaders(
        train_loader, val_loader, test_loader)

    model, optimizer = fabric.setup(model, optimizer)

    #########################################
    ### 3 Training the Model
    #########################################
    run = wandb.init(project=model.hparams["project"])
    start = tt()
    train(
        num_epochs=model.hparams["max_epochs"],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler['scheduler'], 
        train_loader=train_loader,
        val_loader=val_loader,
        #device=device,
        fabric=fabric,
    )
    
    end = tt()
    elapsed = end-start
    fabric.print(f"Time elapsed {elapsed/60:.2f} min")
    fabric.print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
    run.finish()
    #########################################
    ### 4 Evaluation
    #########################################
    model.eval()
    with torch.no_grad():
        test_auc = AUROC(task="binary").to(device)
        test_precision = Precision(task="binary").to(device)
        test_recall = Recall(task="binary").to(device)    
        #test_prc = PrecisionRecallCurve(task="binary").to(device)

        for data in model.testset:
            data = data.to(device)  # Move the batch of graph data to the device
            outputs = model(data)
            test_precision.update(outputs, data.y.int())
            test_recall.update(outputs, data.y.int())
            #test_prc.update(outputs, data.y.int())
            test_auc.update(outputs, data.y.int())

        fabric.print(f"Test auc: {test_auc.compute()*100:.2f}")
        fabric.print(f"Test precision: {test_precision.compute()*100:.2f}%")
        fabric.print(f"Test recall: {test_recall.compute()*100:.2f}%")
        #print(f"Test prcurve: {test_prc.compute()}")
        test_precision.reset()
        test_recall.reset()
        #test_prc.reset()
        test_auc.reset()