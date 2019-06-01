import argparse
import torch
import torch.nn as nn
import os
from models.bulid_model import create_model
from utils.MyDataset import MyDataset, MyDataset1, MyDataset2
import numpy as np
import time
from utils.config import cls_test
from shutil import copyfile


def train():
    '''cfg load'''

    '''parameters which will be optimized  '''
    cfg = cls_test

    save_model_dir = './weight'

    gpu_id = cfg['gpu_id']
    iteration = 0
    max_iter, save_iter, log_iter = cfg['max_iter'], cfg['save_iter'], cfg['log_iter']
    num_epoch = max_iter
    device = 'cuda:{}'.format(cfg['gpu_id']) if cfg['use_gpu'] else 'cpu'
    print(device)
    # device = 'cuda:1'

    '''load the train model'''
    print('Model {}'.format(cfg['base_model']))
    print('Set {}'.format(cfg['train_images_set']))
    
    model = create_model(cfg)

    ''' load pretrained model'''
    if (cfg['is_pretrained']):
        model.load_state_dict(torch.load(cfg['pretrained_path']))
        print('load pretrained end')
    else:
        print('train from beginning')

    ''' dataset load'''
    train_list_path = './all_list.txt'
    # train_set = MyDataset(train_list_path)
    train_set = MyDataset1(cfg['trainset_root'], cfg['train_images_set'],channels=cfg['channels'])
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=cfg['train_batch_size'], shuffle=True)
    test_set = MyDataset1(cfg['testset_root'], cfg['test_images_set'], channels=cfg['channels'])
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=cfg['test_batch_size'], shuffle=False)
    print('trainloader end')

    
    '''    write train log '''
    trainlogfp = open(os.path.join(cfg['trainlogpath'],'{}_{}.txt'.format(cfg['base_model'],time.asctime())),'w')
    trainlogfp.write('Model {}\nSet {}\nlr {} , init_lr {}\n'.format(cfg['base_model'],cfg['train_images_set'],cfg['lr_change'],cfg['learningrate']))
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    now_lr = cfg['learningrate']
    print('lr {} , init_lr {}'.format(cfg['lr_change'],now_lr))
#     optimizer = torch.optim.SGD(model.parameters(), lr=now_lr , momentum = 0.9)
    optimizer = torch.optim.Adam(model.parameters(),lr = now_lr)
    if (cfg['lr_change'] == 'lr_steps'):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg['lr_steps'], gamma=cfg['gamma'])
    elif (cfg['lr_change'] == 'cosine'):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter)
    t0 = 0
    Loss_t = np.zeros([max_iter])
    acc_train = np.zeros([max_iter])
    acc_test = np.zeros([max_iter])
    for epoch in range(num_epoch):

        for idx, (images, labels) in enumerate(train_loader):
            t1 = time.time()
            total_train, current_train = 0, 0
            total_eval, current_eval = 0, 0
            model.train()
            images = images.float().to(device)
            labels = labels.long().to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t2 = time.time()
            t0 += t2 - t1
            if(iteration == 0 or (iteration % cfg['acc_iter'] == 0)):
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                current_train += (predicted == labels).sum().item()
                acc_train[iteration] = 1.0 * current_train / total_train
                
                for timages , tlabels in test_loader:
                    timages = timages.float().to(device)
                    tlabels = tlabels.long().to(device)
                    eval_out = model(timages,'eval')
                    _,predicted= torch.max(eval_out.data,1)
                    total_eval += tlabels.size(0)
                    current_eval += (predicted == tlabels).sum().item()
                acc_test[iteration] = 1.0 * current_eval / total_eval

            if (iteration == 0 or (iteration % log_iter == 0)):
                print('iteration {} lr {}  loss {:.4f}'.format(iteration, optimizer.param_groups[0]['lr'], loss.item()))
                trainlogfp.write('iteration {} lr {:.6f}  loss {:.4f} time {:.6f}\n'.format(iteration, optimizer.param_groups[0]['lr'], loss.item(),t2-t1))
            if (iteration == 0 or (iteration != 0 and iteration % save_iter == 0)):
                save_checkpoint(model, cfg['save_model_dir'], '{}-{}.pth'.format(cfg['base_model'], iteration))
                
            if (cfg['lr_change'] == 'lr_steps'):
                scheduler.step()
                # if (iteration in cfg['lr_steps']):
                #     now_lr *= cfg['gamma']
                #     adjust_learning_rate(optimizer, now_lr)
            elif (cfg['lr_change'] == 'cosine'):
                scheduler.step()
            else:
                pass
            Loss_t[iteration] = loss.item()

            iteration += 1
            if (iteration >= max_iter):
                save_checkpoint(model, cfg['save_model_dir'], '{}-{}.pth'.format(cfg['base_model'], iteration))
                
                print(t0)
                trainlogfp.write('train time {:.6f}\n'.format(t0))
                trainlogfp.close()
                print(os.path.join(cfg['log_path'], '{}.txt'.format(cfg['base_model'])))
                np.savetxt(os.path.join(cfg['log_path'], '{}_{}_{}.txt'.format(cfg['base_model'],cfg['lr_change'],max_iter)), Loss_t)
                np.savetxt(os.path.join(cfg['log_acc'] , '{}_{}_{}_trainacc.txt'.format(cfg['base_model'],max_iter,cfg['acc_iter'])),acc_train)
                np.savetxt(os.path.join(cfg['log_acc'], '{}_{}_{}_testacc.txt'.format(cfg['base_model'], max_iter,cfg['acc_iter'])),acc_test)
                raise SystemExit('train is done')


''' adjust the learning in trianing '''


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


'''save model '''


def save_checkpoint(model, path, name):
    final_path = os.path.join(path, name)
    if (os.path.exists(path)):
        torch.save(model.state_dict(), final_path)
        print('{} save end'.format(final_path))
    else:
        print('file {} is not exist'.format(path))


if __name__ == '__main__':
    train()
