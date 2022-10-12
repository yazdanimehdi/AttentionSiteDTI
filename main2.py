import gc

import torch as th
import pickle
import random
import numpy as np
import time
from model_abiliation import DTITAG
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from dgl import batch, unbatch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, precision_recall_curve, \
    average_precision_score, f1_score, auc, recall_score, precision_score
gpu = 0
device = th.device(gpu if th.cuda.is_available() else "cpu")
#device = th.device("cpu")


def get_roce(predList, targetList, roceRate):
    p = sum(targetList)
    n = len(targetList) - p
    predList = [[index, x] for index, x in enumerate(predList)]
    predList = sorted(predList, key=lambda x:x[1], reverse=True)
    tp1 = 0
    fp1 = 0
    maxIndexs = []
    for x in predList:
        if(targetList[x[0]] == 1):
            tp1 += 1
        else:
            fp1 += 1
            if(fp1>((roceRate*n)/100)):
                break
    roce = (tp1*n)/(p*fp1)
    return roce

#with open("train_new_dude_balanced_all2_active.pkl", "rb") as fp:
#    active = pickle.load(fp)

#with open("train_new_dude_balanced_all2_decoy.pkl", "rb") as fp:
#    inactive = pickle.load(fp)

print("load_done")
#random_selector = np.random.randint(len(inactive) - len(active))
#a = int(len(inactive) / (len(active)))
#ds = active + inactive
with open("human_part_train.pkl", 'rb') as fp:
    ds = pickle.load(fp)

random.shuffle(ds)

X = [i[0] for i in ds]
y = [i[1][0] for i in ds]
# random.shuffle(active)
# random.shuffle(inactive)
#
# X_inactive = [i[0] for i in inactive]
# y_inactive = [i[1][0] for i in inactive]

#with open("test_new_dude_all_active_none_pdb.pkl", "rb") as fp:
#    active = pickle.load(fp)

#with open("test_new_dude_all_decoy_none_pdb.pkl", "rb") as fp:
#    inactive = pickle.load(fp)
#
#ds_test = active + inactive
#random.shuffle(ds_test)
with open("human_part_test.pkl", 'rb') as fp:
    ds_test = pickle.load(fp)
#
X_test = [i[0] for i in ds_test]
y_test = [i[1][0] for i in ds_test]

with open("human_part_val.pkl", 'rb') as fp:
    ds_val = pickle.load(fp)

X_val = [i[0] for i in ds_val]
y_val = [i[1][0] for i in ds_val]

#ds_all = ds + ds_test + ds_val
#random.shuffle(ds_all)
#ds = ds_all[0:int(len(ds_all)*0.8)]
#ds_val = ds_all[int(len(ds_all)*0.8): int(len(ds_all)*0.9)]
#ds_test = ds_all[int(len(ds_all)*0.9): int(len(ds_all))]
#X = [i[0] for i in ds]
#y = [i[1][0] for i in ds]
#X_val = [i[0] for i in ds_val]
#y_val = [i[1][0] for i in ds_val]
#X_test = [i[0] for i in ds_test]
#y_test = [i[1][0] for i in ds_test]
model = DTITAG()
#model.load_state_dict(th.load('without_batching2021_09_05-20_23_26-26_checkpoint.pt', map_location=th.device('cpu'))['net_state'])
model.to(device)
MODEL_NAME = f"model-{int(time.time())}"
optimizer = th.optim.Adam(model.parameters(), lr=1e-3)
criterion = th.nn.BCELoss()
scheduler = ExponentialLR(optimizer, gamma=0.90)

print("init done")


def fwd_pass(X, y, train=False):
    if train:
        model.zero_grad()
    out = []

    for item in X:
        x = [0, 0]
        x[0] = item[0].to(device)
        x[1] = item[1].to(device)
        out.append(model(x))
        del x

    out = th.stack(out, 0).view(-1, 1).to(device)

    y = th.Tensor(y).view(-1, 1).to(device)

    loss = criterion(out, y)

    matches = [th.round(i) == th.round(j) for i, j in zip(out, y)]
    acc = matches.count(True) / len(matches)

    if train:
        loss.backward()
        optimizer.step()

    return acc, loss, out


def test_func(model_f, y_label, X_test_f):
    y_pred = []
    y_label = th.Tensor(y_label)
    print("Testing:")
    print("-------------------")
    with tqdm(range(0, len(X_test_f), 1)) as tepoch:
        for i in tepoch:
            with th.no_grad():
                x = [0, 0]
                x[0] = X_test_f[i][0].to(device)
                x[1] = X_test_f[i][1].to(device)
                y_pred.append(model_f(x).cpu())

    y_pred = th.cat(y_pred, dim=0)
    y_pred_c = [round(i.item()) for i in y_pred]
    roce1 = get_roce(y_pred, y_label, 0.5)
    roce2 = get_roce(y_pred, y_label, 1)
    roce3 = get_roce(y_pred, y_label, 2)
    roce4 = get_roce(y_pred, y_label, 5)
    print("AUROC: " + str(roc_auc_score(y_label, y_pred)), end=" ")
    print("PRAUC: " + str(average_precision_score(y_label, y_pred)), end=" ")
    print("F1 Score: " + str(f1_score(y_label, y_pred_c)), end=" ")
    print("Precision Score:" + str(precision_score(y_label, y_pred_c)), end=" ")
    print("Recall Score:" + str(recall_score(y_label, y_pred_c)), end=" ")
    print("Balanced Accuracy Score " + str(balanced_accuracy_score(y_label, y_pred_c)), end=" ")
    print("0.5 re Score " + str(roce1), end=" ")
    print("1 re Score " + str(roce2), end=" ")
    print("2 re Score " + str(roce3), end=" ")
    print("5 re Score " + str(roce4), end=" ")
    print("-------------------")


def train(net):
    EPOCHS = 100
    BATCH_SIZE = 80

    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            losses = []
            accs = []
            with tqdm(range(0, len(X), BATCH_SIZE)) as tepoch:
                for i in tepoch:
                    tepoch.set_description(f"Epoch {epoch + 1}")
                    try:

                        batch_X = X[i: i+BATCH_SIZE]
                        batch_y = y[i: i+BATCH_SIZE]
                    except:
                        gc.collect()
                        continue
                    acc, loss, _ = fwd_pass(batch_X, batch_y, train=True)

                    losses.append(loss.item())
                    accs.append(acc)
                    acc_mean = np.array(accs).mean()
                    loss_mean = np.array(losses).mean()
                    tepoch.set_postfix(loss=loss_mean, accuracy=100. * acc_mean)
                    if i % 100000 == 0:
                        test_func(model, y_test, X_test)
                        test_func(model, y_val, X_val)
                        # print(f'Average Loss: {val_loss}')
                        # print(f'Average Accuracy: {val_acc}')
                        f.write(
                            f"{MODEL_NAME},{round(time.time(), 3)},{round(float(acc), 2)},{round(float(loss), 4)}\n")
                scheduler.step()
            print(f'Average Loss: {np.array(losses).mean()}')
            print(f'Average Accuracy: {np.array(accs).mean()}')
            #dt = time.strftime("%Y_%m_%d-%H_%M_%S")
            #fn = "without_batching" + str(dt) + str("-") + \
            #     str(epoch) + "_checkpoint.pt"
            #info_dict = {
            #    'epoch': epoch,
            #    'net_state': model.state_dict(),
            #    'optimizer_state': optimizer.state_dict()
            #}
            #th.save(info_dict, fn)


train(model)
test_func(model, y_test, X_test)
