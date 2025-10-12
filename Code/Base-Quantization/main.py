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
from uttils import progress_bar,add_module_dict,enable_calibrate,disable_calibrate,calibrate_adaround,inplace_quantize_layers
from models.VGG import VGG


best_acc = 0  # best test accuracy
def Get_args():
    parser = argparse.ArgumentParser(description='Pytorch MNIST Training with Quantization Support')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate (default: 0.01 for fp32, 0.1 for quantization)')
    parser.add_argument('--type', default='fp32', choices=['fp32','PTQ','QAT'], help='training type: fp32 (full precision), PTQ (post-training quantization), QAT (quantization-aware training)')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--epochs', default=20, type=int, help='number of training epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')

    # 量化相关参数
    parser.add_argument('--Histogram', action='store_true', help='use HistogramObserver for quantization')
    parser.add_argument('--level', default='L', choices=['L','C'], help="quantization level: L (per-tensor), C (per-channel)")
    parser.add_argument('--adaround', action='store_true', help='use AdaRound quantization')
    parser.add_argument('--adaround-iter', default=1000, type=int, help='AdaRound iterations')
    parser.add_argument('--b_start', default=20, type=int, help='AdaRound temperature start')
    parser.add_argument('--b_end', default=2, type=int, help='AdaRound temperature end')
    parser.add_argument('--warmup', default=0.2, type=float, help='AdaRound warmup ratio')

    args = parser.parse_args()
    return args

def train(epoch,net,TrainDataLoader,device,optimizer,criterion):
    print("\n Epoch:%d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx,(inputs,targets) in enumerate(TrainDataLoader):
        inputs,targets = inputs.to(device),targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
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
    start_epoch = 0

    #Data
    print("=" * 20)
    print("Preparing Data...")
    print("="*20)

    # 使用标准MNIST归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data',train=True,download=True,transform=transform
    )
    TrainDataLoader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    TestDataLoader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False, num_workers=2
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
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        print(f"Loaded checkpoint with {len(checkpoint['net'])} parameters")

        # 检查键名格式 - 如果已经有正确的格式则直接使用
        sample_key = list(checkpoint['net'].keys())[0]
        if sample_key.startswith('features.') or sample_key.startswith('classifier.'):
            print("Using checkpoint directly (correct key format)")
            net.load_state_dict(checkpoint['net'])
        else:
            print("Processing checkpoint key format with add_module_dict")
            new_state_dict = add_module_dict(checkpoint['net'])
            net.load_state_dict(new_state_dict)
    
    if args.type == "PTQ" or args.type == "QAT":
        net = inplace_quantize_layers(
            net,
            len(TrainDataLoader) * args.epochs,
            ptq = True if args.type == "PTQ" else False,
            level = args.level,
            adaround = args.adaround
            )
        net = net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(),lr = args.lr)


    if args.type == "fp32":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print("=" * 20)
    print("Model built")
    print("=" * 20)
    
    #Train
    print("Start Train!")
    print("=" * 20)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        if epoch == start_epoch:
            # 对于量化训练，需要校准
            if args.type == "PTQ" or args.type == "QAT":
                enable_calibrate(net)
                calibrate(net,TrainDataLoader,device,criterion)
                disable_calibrate(net)
                if args.adaround:
                   calibrate_adaround(net,args.adaround_iter,args.b_start, args.b_end, args.warmup,TrainDataLoader,device)

                test(args,epoch,net,TestDataLoader,device,optimizer,criterion)
            if args.type == "PTQ":
                break
        else:
            train(epoch,net,TrainDataLoader,device,optimizer,criterion)
            test(args,epoch,net,TestDataLoader,device,optimizer,criterion)
            scheduler.step()
if __name__ == "__main__":
    main()
            