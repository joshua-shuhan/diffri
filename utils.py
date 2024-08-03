import numpy as np
import torch
from torch.optim import Adam, SGD
from tqdm import tqdm
import pickle
import setting
import os


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def load_checkpoint_train(model, config, train_loader, checkpoint_path, valid_epoch_interval=25, valid_loader=None):

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    model.train()

    output_path = os.path.dirname(checkpoint_path) + "/model.pth"
    foldername = os.path.dirname(checkpoint_path)
    earlystop = EarlyStopper(patience=5, min_delta=0)
    best_valid_loss = 1e10

    for epoch_no in range(epoch+1, config['epochs']):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch, epoch_no)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()
        if epoch_no % 50 == 0 and epoch_no > 0:
            temp_path = f'{foldername}/{epoch_no}_model.pt'
            torch.save({
                'epoch': epoch_no,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'lr_scheduler': lr_scheduler.state_dict(),
            }, temp_path)

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0, epoch_no=None)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )
            if earlystop.early_stop(avg_loss_valid):
                break

    torch.save(model.state_dict(), output_path)


def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=25,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.4 * config["epochs"])
    p2 = int(0.5 * config["epochs"])
    p3 = int(0.75 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2, p3], gamma=0.1
    )
    earlystop = EarlyStopper(patience=5, min_delta=0)
    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch, epoch_no)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()
        if epoch_no % 200 == 0:
            temp_path = f'{foldername}/{epoch_no}_model.pt'
            torch.save({
                'epoch': epoch_no,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'lr_scheduler': lr_scheduler.state_dict(),
            }, temp_path)
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0, epoch_no=None)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )
            if earlystop.early_stop(avg_loss_valid):
                break

    if foldername != "":
        torch.save(model.state_dict(), output_path)

def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername="", num_node=0):

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        num_node = num_node
        eval_list = np.arange(num_node)
        store_return = torch.zeros([len(test_loader), num_node, num_node])
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            # test_batch_list = [1,3,5,7,9,11]
            for batch_no, test_batch in enumerate(it, start=1):
                B = test_batch['observed_data'].shape[0]
                setting.init(num_node)
                for i in eval_list:
                    output = model.evaluate(
                        test_batch, torch.tensor(i), nsample)
                    samples, c_target, eval_points, observed_points, observed_time = output
                    samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                    c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                    eval_points = eval_points.permute(0, 2, 1)
                    observed_points = observed_points.permute(0, 2, 1)

                    samples_median = samples.median(dim=1)
                    all_target.append(c_target)
                    all_evalpoint.append(eval_points)
                    all_observed_point.append(observed_points)
                    all_observed_time.append(observed_time)
                    all_generated_samples.append(samples)

                    mse_current = (
                        ((samples_median.values - c_target) * eval_points) ** 2
                    ) * (scaler ** 2)
                    mae_current = (
                        torch.abs((samples_median.values - c_target)
                                  * eval_points)
                    ) * scaler

                    mse_total += mse_current.sum().item()
                    mae_total += mae_current.sum().item()
                    evalpoints_total += eval_points.sum().item()

                    it.set_postfix(
                        ordered_dict={
                            "rmse_total": np.sqrt(mse_total / evalpoints_total),
                            "mae_total": mae_total / evalpoints_total,
                            "batch_no": batch_no,
                        },
                        refresh=True,
                    )

                store_return[batch_no-1] = setting.record_mat
    return store_return


def mask(num_nodes, B, target_list):
    # input_batch: [#sims, #nodes, ...]
    sender_mask = torch.zeros([B, num_nodes-1, num_nodes])
    receiver_mask = torch.zeros([B, num_nodes-1, num_nodes])
    for i in range(B):
        target = target_list[i]
        sender_list = [i for i in range(num_nodes) if i != target]
        sender_mask[i, [i for i in range(num_nodes-1)], sender_list] = 1
    receiver_mask[[i for i in range(B)], :, target_list.long()] = 1
    return torch.FloatTensor(sender_mask), torch.FloatTensor(receiver_mask)
