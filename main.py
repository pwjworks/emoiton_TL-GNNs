from dataset_builder import get_dataset, build_SEED_dataset
from utils.process_SEED import get_data_label_from_mat
from sklearn.metrics import roc_auc_score, accuracy_score
from torch_geometric.loader import DataLoader
from dataset_builder import get_dataset
from arguments import get_args
import time
import torch
import os
import pandas as pd
import numpy as np
import math

# see https://github.com/thuml/Transfer-Learning-Library
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.dan import MultipleKernelMaximumMeanDiscrepancy
from dalib.modules.kernels import GaussianKernel

from models.SEED.encoder_SOGAT import in_encoder_sogat2
from models.SEED.MSMDAER import MSMDAERNet

"""
Settings for training
"""
Network = MSMDAERNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def recorder(train_config, run_config):
    """Record result in csv file
    """
    print('***'*20)
    version = 1
    dfile = ''
    root = '/home/pwjworks/pytorch/emotion_TL-GNNs/results/{:s}/{:s}/{:s}'.format(
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


def get_transfer_learning_model():
    domain_discriminator = DomainDiscriminator(
        in_feature=512, hidden_size=4096).cuda()
    kernels = (GaussianKernel(alpha=0.5), GaussianKernel(
        alpha=1.), GaussianKernel(alpha=2.))
    dann_loss = MultipleKernelMaximumMeanDiscrepancy(kernels).cuda()
    return domain_discriminator, dann_loss


def transfer_train(model, train_loader, test_loader, crit, optimizer, classes, dann_loss, iteration, subject_index):
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
    source_iter = iter(train_loader)
    target_iter = iter(test_loader)
    for i in range(1, iteration+1):
        # extract data
        try:
            source_data, source_label = next(source_iter).x.view(-1, 310)
            source_label = torch.argmax(
                next(source_iter).y.view(-1, 3), axis=1).view(-1, 1)
        except Exception as err:
            source_iter = iter(train_loader)
            source_data = next(source_iter).x.view(-1, 310)
            source_label = torch.argmax(
                next(source_iter).y.view(-1, 3), axis=1)
        try:
            target_data = next(target_iter).x.view(-1, 310)
        except Exception as err:
            target_iter = iter(test_loader)
            target_data = next(target_iter).x.view(-1, 310)

        source_data, source_label = source_data.to(
            device), source_label.to(device)
        target_data = target_data.to(device)
        optimizer.zero_grad()
        cls_loss, mmd_loss, l1_loss = model(
            source_data, number_of_source=14, data_tgt=target_data, label_src=source_label, mark=subject_index)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
        beta = gamma/100
        loss = cls_loss + gamma * mmd_loss + beta * l1_loss
        if i % 400 == 0:
            print("loss: "+str(loss.item()))

        loss.backward()
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

            pred = model(data.x, number_of_source=14)
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


def SEED_train_main(session_id, train_setting, config):

    lastacc_all = 0.0
    iteration = math.ceil(config['epochs']*3394/train_setting['batch_size'])
    train_setting['channels'] = config['channels']

    result_data = []

    print('Cross Validation.')
    dfile, _ = recorder(train_setting, config)
    for subject_index in range(0, config['subjects']):

        train_dataset, test_dataset = get_dataset(
            config, session_id, subject_index)
        train_loader = DataLoader(
            train_dataset, batch_size=train_setting['batch_size'], drop_last=True, shuffle=True)
        test_loader = DataLoader(
            test_dataset, batch_size=train_setting['batch_size'], drop_last=True, shuffle=True)

        model = Network(train_setting).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(train_setting['lr']), weight_decay=0.0003)
        crit = torch.nn.CrossEntropyLoss()
        # transfer learning loss function
        kernels = (GaussianKernel(alpha=0.5), GaussianKernel(
            alpha=1.), GaussianKernel(alpha=2.))
        dann_loss = MultipleKernelMaximumMeanDiscrepancy(kernels).cuda()

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
            loss_all = transfer_train(model, train_loader, test_loader,
                                      crit, optimizer, config['classes'], dann_loss, iteration, subject_index)
            # loss_all = train(model, train_loader, crit,
            #                  optimizer, config['classes'])
            loss = loss_all/len(train_dataset)
            train_AUC, train_acc, _ = 0, 0, 0
            val_AUC, val_acc, valid_loss = evaluate(
                model, test_loader, config['classes'], crit)

            epoch_data.append(
                [str(subject_index), epoch+1, loss, train_AUC, train_acc, val_AUC, valid_loss, val_acc])
            t1 = time.time()
            print('V{:01d}, EP{:03d}, Loss:{:.3f}, AUC:{:.3f}, Acc:{:.3f}, VAUC:{:.2f}, Vacc:{:.2f}, Vloss:{:.2f}, Time: {:.2f}'.format(
                subject_index, epoch+1, loss, train_AUC, train_acc, val_AUC, val_acc, valid_loss, (t1-t0)))
            if loss < 0.15:
                break

        print('Results::::::::::::')
        print('V{:01d}, EP{:03d}, Loss:{:.3f}, AUC:{:.3f}, Acc:{:.3f}, VAUC:{:.2f}, Vacc:{:.2f}, Vloss:{:.2f}, Time: {:.2f}'.
              format(subject_index, epoch+1, loss, train_AUC, train_acc, val_AUC, val_acc, valid_loss, (t1-t0)))

        result_data.append(
            [str(subject_index), epoch+1, loss, train_AUC, train_acc, val_AUC, valid_loss, val_acc])

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
    # train_setting = config['train_setting']
    # SEED_train_main(1, train_setting, config)
    data, label = get_data_label_from_mat(config, 1)
    print()
