import sys
sys.path.append('..')
import os
import tqdm
import time
import copy
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils.getdata import Dataset
from utils.parse_cfg import parse_opt
from nets.factory_nets import all_net
from utils.common import show_db, show_lg, show_er, check_save_pth,mkdir
"""
'yolov5_62cls_st':[yolov5n_cls_st,yolov5s_cls_st,yolov5l_cls_st,yolov5m_cls_st],
'yolov5_62cls_dw':[yolov5n_cls_dw,yolov5s_cls_dw,yolov5l_cls_dw,yolov5m_cls_dw],
'yolov5_62cls_dp':[yolov5n_cls_dp,yolov5s_cls_dp,yolov5l_cls_dp,yolov5m_cls_dp],
"""



def run(opts):
    opts.log_pth = check_save_pth(opts.log_name,'train')
    mode_Lst = opts.mode_Lst[:2]
    Net = all_net[opts.net][0]
    dataDataset = {mode: Dataset(mode, opts.data_path, 1) for mode in mode_Lst}
    show_db(opts.batch_size,'')
    dataloader = {mode: DataLoader(dataDataset[mode],batch_size=opts.batch_size, shuffle=False,  \
                                    num_workers=opts.workers,drop_last=True) \
                for mode in mode_Lst}
    show_lg(f'Dataset loaded! length of train set is ', len(dataloader[mode_Lst[0]]))
    data_size = {v: len(dataDataset[v]) for v in mode_Lst}
    show_lg('data_size',data_size)

    opts.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net(num_cls1=opts.n_cls)
  
    model.to(device=opts.device)
    show_lg('model using device ', opts.device)
    show_lg('parallel mulit device ', opts.n_cuda)

    if opts.resume_weights:
        model.load_state_dict(torch.load(opts.resume_weights)['model'].state_dict())

    criterion = nn.CrossEntropyLoss()
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if opts.adam:
        optimizer = torch.optim.Adam(pg0, lr=0.01, betas=(0.937, 0.999))  # adjust beta1 to momentum
    else:
        optimizer = torch.optim.SGD(pg0, lr=0.01, momentum=0.937, nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': 0.0005})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    show_lg('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)),'')
    del pg0, pg1, pg2

    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
   
    show_lg("train cfg:", opts)
    with SummaryWriter(log_dir=opts.log_pth) as writer:
        train_loop(model, dataloader, criterion, optimizer, exp_lr_scheduler, writer, data_size, opts)

def train_loop(model, dataloader, criterion, optimizer,  exp_lr_scheduler, writer, data_size, opts):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for nepoch in range(opts.epochs):
    # for nepoch in range(1):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:  # val
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for data, label in tqdm.tqdm(dataloader[phase], desc=f'{nepoch + 1}/{opts.epochs}'):
                data = data.to(device=opts.device)
                label = label.to(device=opts.device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    out = model(data)
                    _, pre = torch.max(out, 2)
                    loss = criterion(out, label.squeeze())
                    if phase == 'train':
                        loss.backward(retain_graph=True)
                        optimizer.step()
                        exp_lr_scheduler.step()
                running_loss += loss.item() * data.size(0)
                running_corrects += torch.sum(pre.view(opts.batch_size, -1) == label.data)
            epoch_loss = running_loss / data_size[phase]
            epoch_acc = running_corrects.double() / data_size[phase]
            writer.add_scalar(f"{phase}/Loss", epoch_loss, nepoch)
            writer.add_scalar(f"{phase}/Acc", epoch_acc, nepoch)
            show_lg(f'\n{phase} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}', '\n')
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        time_elapsed = time.time() - since
        since = time.time()
        show_lg('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60), '')
    writer.add_graph(model, data)

    show_lg('Best val Acc: {:4f}'.format(best_acc), '')
    weight_pth = os.path.join(opts.log_pth, 'weights')
    mkdir(weight_pth)
    lastpth = f"{weight_pth}/{opts.net}-lastacc_{epoch_acc:.3f}.pt"
    torch.save({"model": model, "acc": best_acc},lastpth)
    bestpth  = f"{weight_pth}/{opts.net}-bestacc_{best_acc:.3f}.pt"
    model.load_state_dict(best_model_wts)
    torch.save({"model": model, "acc": best_acc},bestpth )
    print(f"best-model save to : {bestpth}")
    print(f"last-model save to : {lastpth}")


def test_loop(model, dataloader, data_size):
    model.eval()
    running_corrects = 0
    for data, label in dataloader['test']:
        data = data.to(opts.device)
        label = label.to(opts.device)
        with torch.set_grad_enabled(False):
            out = model(data)
            _, pre = torch.max(out, 2)
        running_corrects += torch.sum(pre.view(opts.batch_size, -1) == label.data)
    epoch_acc = running_corrects.double() / data_size['test']
    show_lg('test  Acc: ', f'{epoch_acc:.4f}')


if __name__ == "__main__":
    opts = parse_opt()
    show_lg("opts",opts)
    run(opts)
    