import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def Get_args():
    parser = argparse.ArgumentParser(description='Pytorch MNIST QUANTIZE Training')
    parser.add_argument('--lr',default=0.1,type = float)
    parser.add_argument('--type',choices=['fp32','PTQ','PAQ'])
    parser.add_argument('--resume','-r',action='store_true',help='use HistogramObserver to quantizate')
    #HistogramObserver是PyTorch中用于量化模型的观察器模块之一。它通过记录输入的张量的值分布(直方图)来计算量化参数
    parser.add_argument('--level',default='L',choices=['L','C'],help="per_channel or per_tensor")
    parser.add_argument('--path',default='./checkpoint/')

    parser.add_argument('--adaround',action='store_true')
    parser.add_argument('--adaround-iter',default=1000)
    parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
    parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')

    args = parser.parse_args()
    return args

def train(epoch,net,TrainDataLoader,device,optimizer,criterion):
    print("\n Epoch:%d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx,(inputs,targets) in enumerate(TrainDataLoader):
        inputs,targets = inputs.to(device)
        optimizer.zero_grad()
        outputs = net(input)
        loss = criterion(outputs,targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _,predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(TrainDataLoader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(args,epoch,net,TestDataLoader,device,optimizer,criterion):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx,(inputs,targets) in enumerate(TestDataLoader):
            inputs,targets = inputs.to(device),targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs,targets)

            test_loss += loss.item()
            _,predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(TestDataLoader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.* correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net':net.state_dict(),
            'acc':acc,
            'epoch':epoch
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if args.type == 'fp32':
            torch.save(state, './checkpoint/ckpt.pth')
        else:
            torch.save(state, './checkpoint/ckpt_q.pth')
        best_acc = acc

def calibrate(net,TrainDataLoader,device,criterion):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx,(inputs,targets) in enumerate(TrainDataLoader):
            if batch_idx == 10: break
            inputs,targets = inputs.to(device),targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs,targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(TrainDataLoader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

def calibrate_ada(net,TrainDataLoader,device,criterion):
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx,(inputs,targets) in enumerate(TrainDataLoader):
        if batch_idx == 10: break
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(TrainDataLoader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return net
def main():
    args = Get_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0#best test acc
    start_epoch = 0
    train_epoches = 20

    #Data
    print("=" * 20)
    print("Preparing Data...")
    print("="*20)

    train_dataset = torchvision.datasets.MNIST(
        root='./data',train=True,download=True,transform=torchvision.transforms.ToTensor()
    )
    TrainDataLoader = torch.utils.data.DataLoader(
        train_dataset,batch_size=128,shuffle=True,num_workers=4
    )

    test_dataset = torchvision.datasets.MNIST(
        root='./data',train=False,download=True,transform=torchvision.transforms.ToTensor()
    )
    TesstDataLoader = torch.utils.data.DataLoader(
        test_dataset,batch_size=100,shuffle=False,num_workers=4
    )

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
    
    print("data loaded")
    #Model
    print("="*20)
    print("Building model..")
    print("="*20)

    net = VGG('VGG_s')
    net = net.to(device)

    if args.resume:
        print("=" * 20)
        print("Resuming from checkpoint..")
        print("=" * 20)
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    
    if args.type == "PTQ":
        checkpoint = torch.load('./checkpoint/ckpt.path')

        new_state_dict = add_module_dict(checkpoint['net'])
        net.load_state_dict(new_state_dict)
    
    if args.type == "PTQ" or args.type == "QAT":
        net = inplace_quantize_layers(
            net,
            len(TrainDataLoader)*train_epoches,
            ptq = True if args.type == "PTQ" else False,
            Histogram = args.Histogram,
            level = args.level,
            adaround = args.adaround
            )
        net = net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(),lr = args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=train_epoches)
    print("=" * 20)
    print("Model built")
    print("=" * 20)
    
    #Train
    print("Start Train!")
    print("=" * 20)

    for epoch in range(start_epoch,start_epoch + train_epoches):
        if epoch == start_epoch:
            enable_calibrate(net)
            calibrate()
            disable_calibrate(net)
            if args.adaround:
                calibrate_adaround(net,args.adaround_iter,args.b_start, args.b_end, args.warmup,TrainDataLoader, device)
            test(args,epoch,net,TesstDataLoader,device)
            if args.type == "PTQ":
                break
        else:
            train(epoch,net,TrainDataLoader,device)
            test(args,epoch,net,TesstDataLoader,device)
            scheduler.step()
            