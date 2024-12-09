import os
import time
import yaml
import torch
import pickle
import argparse
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from model_man import Model
from torch.utils.data import DataLoader
from dataset_reader import DataSetReader
from torch.utils.tensorboard import SummaryWriter
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def init_parse():
    parser = argparse.ArgumentParser(description="model arg")
    parser.add_argument('--mode', type=str, default='train', help='want to train or val')
    parser.add_argument('--exp_dir', type=str, default='', help='')
    parser.add_argument('--output', type=str, default='slowfast_traslow_new_mm', help='')
    parser.add_argument('--batch_size', type=int, default=8, help='number of batches for train')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs for train')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for train')
    parser.add_argument('--weights',
                  default=None,
                  help='the weights for network initialization')

    # data
    parser.add_argument('--label_path', type=str, default='label', help='')
    parser.add_argument('--benchmark', type=str, default='cs1',
                        choices=['cs1', 'cv1'])
    # cs1: cross-scene benchmark; cv1: cross-video benchmark

    parser.add_argument('--feature_path', type=str, default='/home/user/data', help='')
    # modify the file path to include the two folders 'vr' and 'rr'
    
    parser.add_argument('--modality', type=str, default='tra', choices=['vr', 'rr'], help='')
    # if modality == vr, slowpath input: vr, fastpath input:rr
    # if modality == rr, slowpath input: rr, fastpath input:vr
    
    parser.add_argument('--frame_number', type=int, default='32', choices=['4', '8', '16', '32'], help='')
    # if frame == 32, then the fastpath input contains 32 frames
    
    args = parser.parse_args()
    return args


def print_log(exp_dir, str):
    print(str)
    with open('{}/log.txt'.format(exp_dir), 'a') as f:
        print(str, file=f)


if __name__ == '__main__':
    args = init_parse()

    running_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    exp_dir = os.path.join(args.output, args.benchmark, running_date)
    os.makedirs(exp_dir, exist_ok=True)
    args.exp_dir = exp_dir

    arg_dict = vars(args)
    with open('{}/config.yaml'.format(exp_dir), 'w') as f:
        yaml.dump(arg_dict, f)

    log_writer = SummaryWriter(log_dir=exp_dir)

    net = Model()
    net.cuda()
    if args.weights:
        net.load_state_dict(torch.load(args.weights))
        print_log(exp_dir, "Load model weights from {}".format(args.weights))
    else:
        print_log(exp_dir, "Training model from scratch...")

    dataset_test = DataSetReader(args, 'val')
    dataset_test_loader = DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers,
                                     shuffle=False, pin_memory=True)
    print_log(exp_dir, "Load test data Successfully ！")
    if args.mode == 'train':
        dataset_train = DataSetReader(args)
        dataset_train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                          shuffle=True, pin_memory=True, drop_last=True)
        print_log(exp_dir, "Load train data Successfully ！")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=1e-4)

    max_acc = 0
    max_epoch = 0
    for epoch in range(args.num_epochs):
        if args.mode == 'train':

            net.train()
            pbar = tqdm(dataset_train_loader)
            for itern, data in enumerate(pbar):
                slow_path = data[0].cuda()
                fast_path = data[1].cuda()               
                labels = data[2].cuda()
                # print('slow', slow_path.shape)
                # print('fast', fast_path.shape)
                outputs = net(slow_path, fast_path)
                
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                value, predict_label = torch.max(outputs.data, 1)
                acc = torch.mean((predict_label == labels.data).float())
                pbar.set_description("[{}/{}]  Acc:  {:2f}  Loss: {:2f}".format(epoch + 1, args.num_epochs, acc, loss.data.item()))
                log_writer.add_scalar('Acc', acc, epoch * len(dataset_train_loader) + itern)
                log_writer.add_scalar('NLL Loss', loss.data.item(), epoch * len(dataset_train_loader) + itern)

        labels_list = torch.empty(0).cuda()
        outputs_list = torch.empty(0).cuda()
        net.eval()
        ppbar = tqdm(dataset_test_loader)
        for itern, data in enumerate(ppbar):
            slow_path = data[0].cuda()
            fast_path = data[1].cuda()           
            labels = data[2].cuda()
            with torch.no_grad():
                outputs = net(slow_path, fast_path)

            labels_list = torch.cat((labels_list, labels), 0)
            outputs_list = torch.cat((outputs_list, outputs), 0)

        value, predict_label = torch.max(outputs_list.data, 1)
        acc = torch.mean((predict_label == labels_list.data).float())

        log_writer.add_scalar('Test/Acc', acc, epoch)

        output_scores = outputs_list.data.cpu().numpy()
        name = dataset_test.sample_name[:len(output_scores)]
        score_dict = dict(zip(name, output_scores))
        with open('{}/epoch{}_{:2f}_score.pkl'.format(exp_dir, epoch + 1, acc.data.item()), 'wb') as f:
            pickle.dump(score_dict, f)

        if acc.data.item() > max_acc:
            max_acc = acc.data.item()
            max_epoch = epoch + 1
            path = os.path.join(exp_dir, 'Max_Acc.pt')
            torch.save(net.state_dict(), path)
            print_log(exp_dir, "\nMax_Acc model saved to: {}".format(path))

        print_log(exp_dir, "\n-------------------------------------------------------")
        print_log(exp_dir, "\033[92m Epoch:  [{}/{}]  Done with {:2f}% Accuracy for {} samples\033[0m".format(epoch + 1, args.num_epochs, acc * 100, len(labels_list)))
        print_log(exp_dir, "\033[92m Epoch:  [{}/{}]  Max_Acc with {:2f}% Accuracy \033[0m".format(max_epoch, args.num_epochs, max_acc * 100, ))
        print_log(exp_dir, "-------------------------------------------------------\n\n")