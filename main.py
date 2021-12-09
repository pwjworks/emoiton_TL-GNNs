from dataset_builder import build_dataset, get_dataset
from sklearn.metrics import roc_auc_score, accuracy_score
from torch_geometric.loader import DataLoader
from dataset_builder import get_dataset
from arguments import get_args
import time
import torch
import os
import pandas as pd
import numpy as np


"""
Settings for training
"""
Network = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def recorder(train_config, run_config):
    """Record result in csv file
    """
    print('***'*20)
    version = 1
    dfile = ''
    root = '/home/pwjworks/pytorch/emotion_gnns/result/{:s}/{:s}/{:s}'.format(
        time.strftime('%m.%d'), run_config['dataset'], run_config['feature'])
    if not os.path.exists(root):
        os.makedirs(root)
    while(1):
        # show train settings in result csv filename.
        prefix = ""
        for key, value in train_config.items():
            prefix += key+"="+str(value)+"_"
        dfile = root+'/{:s}{}_LOG_{:.0f}.csv'.format(
            prefix, Network.__name__, version)
        if not os.path.exists(dfile):
            break
        version += 1
    print(dfile)
    df = pd.DataFrame()
    df.to_csv(dfile)
    print('***'*20)
    return dfile, version


def transfer_train(model, train_loader, test_loader, crit, optimizer, classes, dann_loss):
    """transfer learning train

    Args:
        model (GNN model): model
        train_loader (loader): loader
        crit (crit): crit
        optimizer (optimizer): optimizer

    Returns:
        loss: loss
    """
    model.train()
    loss_all = 0

    for train_data in train_loader:
        train_data = train_data.cuda()

        optimizer.zero_grad()

        # Multiple Classes classification Loss function
        t = train_data.y.view(-1, classes)
        label = torch.argmax(t, axis=1)

        output, _, f_s = model(
            train_data.x, train_data.edge_index, train_data.batch)

        loss = crit(output, label)

        transfer_losses = []
        # try:
        #     data, target = next(test_iter)
        # except StopIteration:
        #     test_iter = iter(dataloader)
        #     data, target = next(test_iter)
        for test_data in test_loader:
            test_data = test_data.cuda()

            _, _, f_t = model(
                test_data.x, test_data.edge_index, test_data.batch)
            transfer_losses.append(dann_loss(f_s, f_t))

        loss += torch.mean(torch.tensor(transfer_losses))
        loss.backward()
        loss_all += train_data.num_graphs * loss.item()
        optimizer.step()

    return loss_all


def train(model, train_loader, crit, optimizer, classes):
    """train

    Args:
        model (GNN model): model
        train_loader (loader): loader
        crit (crit): crit
        optimizer (optimizer): optimizer

    Returns:
        loss: loss
    """
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)

        optimizer.zero_grad()

        # Multiple Classes classification Loss function
        t = data.y.view(-1, classes)
        label = torch.argmax(t, axis=1)

        output, _ = model(data.x, data.edge_index, data.batch)

        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all


def evaluate(model, loader, classes, crit):
    model.eval()

    predictions = []
    labels = []
    valid_losses = []
    with torch.no_grad():
        for data in loader:
            label = data.y.view(-1, classes)
            data = data.to(device)

            _, pred = model(data.x, data.edge_index, data.batch)
            pred = pred.detach().cpu()
            loss = crit(pred, label)
            valid_losses.append(loss.item())

            pred = pred.numpy()

            pred = np.squeeze(pred)
            predictions.append(pred)
            labels.append(label)

    valid_loss = np.average(valid_losses)
    predictions = np.vstack(predictions)
    labels = np.vstack(labels)

    # AUC score estimation
    AUC = 0
    for i in range(0, classes):
        AUC += roc_auc_score(labels[:, i], predictions[:, i])
    AUC /= classes

    # Accuracy
    predictions = np.argmax(predictions, axis=-1)
    labels = np.argmax(labels, axis=-1)
    acc = accuracy_score(labels, predictions)

    return AUC, acc, valid_loss


def main(train_config, config):

    lastacc_all = 0.0
    train_config['channels'] = config['channels']
    train_config['save'] = False
    build_dataset(config)  # Build dataset for each fold

    result_data = []

    print('Cross Validation.')
    dfile, _ = recorder(train_config, config)
    for cv_n in range(0, config['subjects']):

        train_dataset, test_dataset = get_dataset(config, cv_n)
        train_loader = DataLoader(
            train_dataset, batch_size=train_config['batch_size'], drop_last=False, shuffle=True)
        test_loader = DataLoader(
            test_dataset, batch_size=train_config['batch_size'], drop_last=False, shuffle=True)

        # get_nodes_graph(train_dataset[0])

        model = Network(train_config).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(train_config['lr']), weight_decay=0.0003)
        crit = torch.nn.CrossEntropyLoss()

        epoch_data = []
        loss = None
        t0 = 0
        t1 = 0
        val_acc = 0
        val_AUC = 0
        train_acc = 0
        train_AUC = 0
        epoch = 0
        lastacc = 0
        for epoch in range(config['epochs']):
            t0 = time.time()

            # transfer learning
            # loss_all = transfer_train(model, train_loader, test_loader,
            #                           crit, optimizer, classes, dann_loss)

            loss_all = train(model, train_loader, crit,
                             optimizer, config['classes'])
            loss = loss_all/len(train_dataset)
            train_AUC, train_acc, _ = evaluate(
                model, train_loader, config['classes'], crit)
            val_AUC, val_acc, valid_loss = evaluate(
                model, test_loader, config['classes'], crit)

            epoch_data.append(
                [str(cv_n), epoch+1, loss, train_AUC, train_acc, val_AUC, valid_loss, val_acc])
            t1 = time.time()
            print('V{:01d}, EP{:03d}, Loss:{:.3f}, AUC:{:.3f}, Acc:{:.3f}, VAUC:{:.2f}, Vacc:{:.2f}, Vloss:{:.2f}, Time: {:.2f}'.format(
                cv_n, epoch+1, loss, train_AUC, train_acc, val_AUC, val_acc, valid_loss, (t1-t0)))
            # if train_AUC > 0.999:
            if loss < 0.15:
                break

        print('Results::::::::::::')
        print('V{:01d}, EP{:03d}, Loss:{:.3f}, AUC:{:.3f}, Acc:{:.3f}, VAUC:{:.2f}, Vacc:{:.2f}, Vloss:{:.2f}, Time: {:.2f}'.
              format(cv_n, epoch+1, loss, train_AUC, train_acc, val_AUC, val_acc, valid_loss, (t1-t0)))

        result_data.append(
            [str(cv_n), epoch+1, loss, train_AUC, train_acc, val_AUC, valid_loss, val_acc])

        df = pd.DataFrame(data=result_data, columns=[
            'Fold', 'Epoch', 'Loss', 'Tra_AUC', 'Tra_acc', 'Val_AUC', 'Val_loss', 'Val_acc'])

        df.to_csv(dfile)

    df = pd.read_csv(dfile)
    lastacc = ['Val_acc', df['Val_acc'].mean()]
    lastauc = ['Val_AUC', df['Val_AUC'].mean()]
    print(lastacc)
    print(lastauc)
    print(dfile)
    print('*****************')

    finalresult = pd.DataFrame(
        data=[lastacc, lastauc], columns=['Fold', 'Epoch'])
    df2 = df.append(finalresult)
    df2.to_csv(dfile)
    print(df2)
    lastacc_all += lastacc[1]


if __name__ == "__main__":
    config = get_args()
    # build_SEED_dataset(config)
    test, train = get_dataset(config, 1, 1)
    train_setting = config['train_setting']
    main(train_setting, config)
    print()
