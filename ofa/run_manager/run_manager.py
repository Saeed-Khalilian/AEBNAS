import os
import time
import json
import math
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision
from itertools import product
import copy
import utils
from ..imagenet_codebase.utils import get_net_info, cross_entropy_loss_with_soft_target, cross_entropy_with_label_smoothing
from ofa.utils import  AverageMeter, accuracy


class RunConfig:

    def __init__(self, n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
                 dataset, train_batch_size, test_batch_size, valid_size,
                 opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
                 mixup_alpha,
                 model_init, validation_frequency, print_frequency):
        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.lr_schedule_type = lr_schedule_type
        self.lr_schedule_param = lr_schedule_param

        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.valid_size = valid_size

        self.opt_type = opt_type
        self.opt_param = opt_param
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.no_decay_keys = no_decay_keys

        self.mixup_alpha = mixup_alpha

        self.model_init = model_init
        self.validation_frequency = validation_frequency
        self.print_frequency = print_frequency

    @property
    def config(self):
        config = {}
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    def copy(self):
        return RunConfig(**self.config)

    """ learning rate """

    def calc_learning_rate(self, epoch, batch=0, nBatch=None):
        if self.lr_schedule_type == 'cosine':
            T_total = self.n_epochs * nBatch
            T_cur = epoch * nBatch + batch
            lr = 0.5 * self.init_lr * (1 + math.cos(math.pi * T_cur / T_total))
        elif self.lr_schedule_type is None:
            lr = self.init_lr
        else:
            raise ValueError('do not support: %s' % self.lr_schedule_type)
        return lr

    def adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None):
        """ adjust learning of a given optimizer and return the new learning rate """
        new_lr = self.calc_learning_rate(epoch, batch, nBatch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def warmup_adjust_learning_rate(self, optimizer, T_total, nBatch, epoch, batch=0, warmup_lr=0):
        T_cur = epoch * nBatch + batch + 1
        new_lr = T_cur / T_total * (self.init_lr - warmup_lr) + warmup_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    """ data provider """

    @property
    def data_provider(self):
        raise NotImplementedError

    @property
    def train_loader(self):
        return self.data_provider.train

    @property
    def valid_loader(self):
        return self.data_provider.valid

    @property
    def test_loader(self):
        return self.data_provider.test

    def random_sub_train_loader(self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None):
        return self.data_provider.build_sub_train_loader(n_images, batch_size, num_worker, num_replicas, rank)

    """ optimizer """

    def build_optimizer(self, net_params):
        if self.no_decay_keys is not None:
            assert isinstance(net_params, list) and len(net_params) == 2
            net_params = [
                {'params': net_params[0], 'weight_decay': self.weight_decay},
                {'params': net_params[1], 'weight_decay': 0},
            ]
        else:
            net_params = [{'params': net_params, 'weight_decay': self.weight_decay}]

        if self.opt_type == 'sgd':
            opt_param = {} if self.opt_param is None else self.opt_param
            momentum, nesterov = opt_param.get('momentum', 0.9), opt_param.get('nesterov', True)
            optimizer = torch.optim.SGD(net_params, self.init_lr, momentum=momentum, nesterov=nesterov)
        elif self.opt_type == 'adam':
            optimizer = torch.optim.Adam(net_params, self.init_lr)
        else:
            raise NotImplementedError
        return optimizer

class RunManager:

    def __init__(self, path, net, run_config: RunConfig, init=True, measure_latency=None, no_gpu=False, mix_prec=None):
        self.path = path
        self.net = net
        self.run_config = run_config
        self.mix_prec = mix_prec

        self.best_acc = 0
        self.start_epoch = 0
        self.info = None
        self.input_shape = None

        self.threhold_range=None
        self.target_macs = None
        self.alpha_macs = None

        os.makedirs(self.path, exist_ok=True)
        # move network to GPU if available
        if torch.cuda.is_available() and (not no_gpu):
            self.device = torch.device('cuda:0')
            self.net = self.net.to(self.device)
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        # initialize model (default)
        if init:
            self.network.init_model(run_config.model_init)

        # net info
        '''
        net_info = get_net_info(self.net, self.run_config.data_provider.data_shape, measure_latency, True)
        with open('%s/net_info.txt' % self.path, 'w') as fout:
            fout.write(json.dumps(net_info, indent=4) + '\n')
            try:
                fout.write(self.network.module_str)
            except Exception:
                pass
        '''

        # criterion
        if isinstance(self.run_config.mixup_alpha, float):
            self.train_criterion = cross_entropy_loss_with_soft_target
        elif self.run_config.label_smoothing > 0:
            self.train_criterion = lambda pred, target: \
                cross_entropy_with_label_smoothing(pred, target, self.run_config.label_smoothing)
        else:
            self.train_criterion = nn.CrossEntropyLoss()
        self.test_criterion = nn.CrossEntropyLoss()

        # optimizer
        if self.run_config.no_decay_keys:
            keys = self.run_config.no_decay_keys.split('#')
            net_params = [
                self.network.get_parameters(keys, mode='exclude'),  # parameters with weight decay
                self.network.get_parameters(keys, mode='include'),  # parameters without weight decay
            ]
        else:
            try:
                net_params = self.network.weight_parameters()
            except Exception:
                net_params = self.network.parameters()
        self.optimizer = self.run_config.build_optimizer(net_params)

        if mix_prec is not None:
            from apex import amp
            self.network, self.optimizer = amp.initialize(self.network, self.optimizer, opt_level=mix_prec)

        self.net = torch.nn.DataParallel(self.net)

    """ save path and log path """

    @property
    def save_path(self):
        if self.__dict__.get('_save_path', None) is None:
            save_path = os.path.join(self.path, 'checkpoint')
            os.makedirs(save_path, exist_ok=True)
            self.__dict__['_save_path'] = save_path
        return self.__dict__['_save_path']

    @property
    def logs_path(self):
        if self.__dict__.get('_logs_path', None) is None:
            logs_path = os.path.join(self.path, 'logs')
            os.makedirs(logs_path, exist_ok=True)
            self.__dict__['_logs_path'] = logs_path
        return self.__dict__['_logs_path']

    @property
    def network(self):
        if isinstance(self.net, nn.DataParallel):
            return self.net.module
        else:
            return self.net

    @network.setter
    def network(self, new_val):
        if isinstance(self.net, nn.DataParallel):
            self.net.module = new_val
        else:
            self.net = new_val

    def write_log(self, log_str, prefix='valid', should_print=True):
        """ prefix: valid, train, test """
        if prefix in ['valid', 'test']:
            with open(os.path.join(self.logs_path, 'valid_console.txt'), 'a') as fout:
                fout.write(log_str + '\n')
                fout.flush()
        if prefix in ['valid', 'test', 'train']:
            with open(os.path.join(self.logs_path, 'train_console.txt'), 'a') as fout:
                if prefix in ['valid', 'test']:
                    fout.write('=' * 10)
                fout.write(log_str + '\n')
                fout.flush()
        else:
            with open(os.path.join(self.logs_path, '%s.txt' % prefix), 'a') as fout:
                fout.write(log_str + '\n')
                fout.flush()
        if should_print:
            print(log_str)

    """ save and load models """

    def save_model(self, checkpoint=None, is_best=False, model_name=None):
        if checkpoint is None:
            checkpoint = {'state_dict': self.network.state_dict()}

        if model_name is None:
            model_name = 'checkpoint.pth.tar'

        if self.mix_prec is not None:
            from apex import amp
            checkpoint['amp'] = amp.state_dict()

        checkpoint['dataset'] = self.run_config.dataset  # add `dataset` info to the checkpoint
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        model_path = os.path.join(self.save_path, model_name)
        with open(latest_fname, 'w') as fout:
            fout.write(model_path + '\n')
        torch.save(checkpoint, model_path)

        if is_best:
            best_path = os.path.join(self.save_path, 'model_best.pth.tar')
            torch.save({'state_dict': checkpoint['state_dict']}, best_path)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                model_fname = fin.readline()
                if model_fname[-1] == '\n':
                    model_fname = model_fname[:-1]
        try:
            if model_fname is None or not os.path.exists(model_fname):
                model_fname = '%s/checkpoint.pth.tar' % self.save_path
                with open(latest_fname, 'w') as fout:
                    fout.write(model_fname + '\n')
            print("=> loading checkpoint '{}'".format(model_fname))

            if torch.cuda.is_available():
                checkpoint = torch.load(model_fname)
            else:
                checkpoint = torch.load(model_fname, map_location='cpu')

            self.network.load_state_dict(checkpoint['state_dict'])

            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
            if 'best_acc' in checkpoint:
                self.best_acc = checkpoint['best_acc']
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.mix_prec is not None and 'amp' in checkpoint:
                from apex import amp
                amp.load_state_dict(checkpoint['amp'])

            print("=> loaded checkpoint '{}'".format(model_fname))
        except Exception:
            print('fail to load checkpoint from %s' % self.save_path)

    def save_config(self):
        """ dump run_config and net_config to the model_folder """
        net_save_path = os.path.join(self.path, 'net.config')
        json.dump(self.network.config, open(net_save_path, 'w'), indent=4)
        print('Network configs dump to %s' % net_save_path)

        run_save_path = os.path.join(self.path, 'run.config')
        json.dump(self.run_config.config, open(run_save_path, 'w'), indent=4)
        print('Run configs dump to %s' % run_save_path)

    """ train and test """

    def validate(self, epoch=0, is_test=True, run_str='', net=None, data_loader=None, no_logs=False):
        if net is None:
            net = self.net
        if not isinstance(net, nn.DataParallel):
            net = nn.DataParallel(net)

        if data_loader is None:
            if is_test:
                data_loader = self.run_config.test_loader
            else:
                data_loader = self.run_config.valid_loader

        net.eval()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        with torch.no_grad():
            with tqdm(total=len(data_loader),
                      desc='Validate Epoch #{} {}'.format(epoch + 1, run_str), disable=no_logs) as t:
                for i, (images, labels) in enumerate(data_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    # compute output
                    output = net(images)
                    loss = self.test_criterion(output, labels)
                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(output, labels, topk=(1, 5))

                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0].item(), images.size(0))
                    top5.update(acc5[0].item(), images.size(0))
                    t.set_postfix({
                        'loss': losses.avg,
                        'top1': top1.avg,
                        'top5': top5.avg,
                        'img_size': images.size(2),
                    })
                    t.update(1)
        return losses.avg, top1.avg, top5.avg

    def train_one_epoch(self, args, epoch, warmup_epochs=0, warmup_lr=0):
        # switch to train mode
        self.net.train()

        nBatch = len(self.run_config.train_loader)

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        data_time = AverageMeter()

        with tqdm(total=nBatch,
                  desc='Train Epoch #{}'.format(epoch + 1)) as t:
            end = time.time()
            for i, (images, labels) in enumerate(self.run_config.train_loader):
                data_time.update(time.time() - end)
                if epoch < warmup_epochs:
                    new_lr = self.run_config.warmup_adjust_learning_rate(
                        self.optimizer, warmup_epochs * nBatch, nBatch, epoch, i, warmup_lr,
                    )
                else:
                    new_lr = self.run_config.adjust_learning_rate(self.optimizer, epoch - warmup_epochs, i, nBatch)

                images, labels = images.to(self.device), labels.to(self.device)
                target = labels

                # soft target
                if args.teacher_model is not None:
                    args.teacher_model.train()
                    with torch.no_grad():
                        soft_logits = args.teacher_model(images).detach()
                        soft_label = F.softmax(soft_logits, dim=1)

                # compute output
                if isinstance(self.network, torchvision.models.Inception3):
                    output, aux_outputs = self.net(images)
                    loss1 = self.train_criterion(output, labels)
                    loss2 = self.train_criterion(aux_outputs, labels)
                    loss = loss1 + 0.4 * loss2
                else:
                    output = self.net(images)
                    loss = self.train_criterion(output, labels)

                if args.teacher_model is None:
                    loss_type = 'ce'
                else:
                    if args.kd_type == 'ce':
                        kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)
                    else:
                        kd_loss = F.mse_loss(output, soft_logits)
                    loss = args.kd_ratio * kd_loss + loss
                    loss_type = '%.1fkd-%s & ce' % (args.kd_ratio, args.kd_type)

                # compute gradient and do SGD step
                self.net.zero_grad()  # or self.optimizer.zero_grad()
                if self.mix_prec is not None:
                    from apex import amp
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)

                self.optimizer.step()

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))

                t.set_postfix({
                    'loss': losses.avg,
                    'top1': top1.avg,
                    'top5': top5.avg,
                    'img_size': images.size(2),
                    'lr': new_lr,
                    'loss_type': loss_type,
                    'data_time': data_time.avg,
                })
                t.update(1)
                end = time.time()
        return losses.avg, top1.avg, top5.avg

    def adaptive_train_one_epoch(self, args, epoch, warmup_epochs=0, warmup_lr=0):
        # switch to train mode
        self.net.train()

        nBatch = len(self.run_config.train_loader)

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        data_time = AverageMeter()

        with tqdm(total=nBatch,
                  desc='Train Epoch #{}'.format(epoch + 1)) as t:
            end = time.time()
            for i, (images, labels) in enumerate(self.run_config.train_loader):
                data_time.update(time.time() - end)
                if epoch < warmup_epochs:
                    new_lr = self.run_config.warmup_adjust_learning_rate(
                        self.optimizer, warmup_epochs * nBatch, nBatch, epoch, i, warmup_lr,
                    )
                else:
                    new_lr = self.run_config.adjust_learning_rate(self.optimizer, epoch - warmup_epochs, i, nBatch)

                images, labels = images.to(self.device), labels.to(self.device)
                target = labels

                # soft target
                if args.teacher_model is not None:
                    args.teacher_model.train()
                    with torch.no_grad():
                        soft_logits = args.teacher_model(images).detach()
                        soft_label = F.softmax(soft_logits, dim=1)

                # compute output
                if isinstance(self.network, torchvision.models.Inception3):
                    output, aux_outputs = self.net(images)
                    loss1 = self.train_criterion(output, labels)
                    loss2 = self.train_criterion(aux_outputs, labels)
                    loss = loss1 + 0.4 * loss2
                else:
                    #weighted loss 
                    preds = self.net(images)
            
                    weights = np.ones(len(preds)) #all weigths are set to 1

                    loss = 0
                    acc1 = 0
                    acc5 = 0
                    w_tot = 0

                    for p,w in zip(preds,weights):
                        w_tot += w
                        t_loss = self.train_criterion(p, labels) 
                        loss += t_loss * w
                        acc1_exit, acc5_exit = accuracy(p, target, topk=(1, 5))
                        acc1 += acc1_exit[0].item() * w
                        acc5 += acc5_exit[0].item() * w
                    
                    acc1 = acc1 / w_tot #average acc of all the exits
                    acc5 = acc5 / w_tot

                if args.teacher_model is None:
                    loss_type = 'ce'
                else:
                    if args.kd_type == 'ce':
                        kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)
                    else:
                        kd_loss = F.mse_loss(output, soft_logits)
                    loss = args.kd_ratio * kd_loss + loss
                    loss_type = '%.1fkd-%s & ce' % (args.kd_ratio, args.kd_type)

                # compute gradient and do SGD step
                self.net.zero_grad()  # or self.optimizer.zero_grad()
                if self.mix_prec is not None:
                    from apex import amp
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)

                self.optimizer.step()

                # measure accuracy and record loss

                #acc1, acc5 = accuracy(aux_outputs, target, topk=(1, 5))
                
                losses.update(loss.item(), images.size(0))
                top1.update(acc1, images.size(0))
                top5.update(acc5, images.size(0))

                t.set_postfix({
                    'loss': losses.avg,
                    'top1': top1.avg,
                    'top5': top5.avg,
                    'img_size': images.size(2),
                    'lr': new_lr,
                    'loss_type': loss_type,
                    'data_time': data_time.avg,
                })
                t.update(1)
                end = time.time()
        return losses.avg, top1.avg, top5.avg

    def adaptive_train(self, args, warmup_epoch=0, warmup_lr=0):

        for epoch in range(self.start_epoch, self.run_config.n_epochs + warmup_epoch):
        
            train_loss, train_top1, train_top5 = self.adaptive_train_one_epoch(args, epoch, warmup_epoch, warmup_lr)

            ## Early Stopping setup ##
            patience = 10
            counter = 0
            ##

            if (epoch + 1) % self.run_config.validation_frequency == 0:
                img_size, val_loss, val_acc, val_acc5 = self.validate_all_resolution(epoch=epoch, is_test=False, is_adaptive = True)

                is_best = np.mean(val_acc) > self.best_acc
                self.best_acc = max(self.best_acc, np.mean(val_acc))
                val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f} ({4:.3f})'. \
                    format(epoch + 1 - warmup_epoch, self.run_config.n_epochs,
                           np.mean(val_loss), np.mean(val_acc), self.best_acc)
                val_log += '\ttop-5 acc {0:.3f}\tTrain top-1 {top1:.3f}\tloss {train_loss:.3f}\t'. \
                    format(np.mean(val_acc5), top1=train_top1, train_loss=train_loss)

                for i_s, v_a in zip(img_size, val_acc):
                      val_log += '(%d, %.3f), ' % (i_s, v_a)

                self.write_log(val_log, prefix='valid', should_print=False)
                
                ## Early Stopping check##
                if(is_best):
                    counter = 0
                else:
                    counter = counter + 1
                    if (counter >= patience):
                        print("Early Stopping")
                        self.save_model({
                            'epoch': epoch,
                            'best_acc': self.best_acc,
                            'optimizer': self.optimizer.state_dict(),
                            'state_dict': self.network.state_dict(),
                        }, is_best=is_best)
                        return self.net
                        #break
                ##

            else:
                is_best = False
            

            self.save_model({
                'epoch': epoch,
                'best_acc': self.best_acc,
                'optimizer': self.optimizer.state_dict(),
                'state_dict': self.network.state_dict(),
            }, is_best=is_best)

        

        return self.net

    def train(self, args, warmup_epoch=0, warmup_lr=0):

        for epoch in range(self.start_epoch, self.run_config.n_epochs + warmup_epoch):
        
            train_loss, train_top1, train_top5 = self.train_one_epoch(args, epoch, warmup_epoch, warmup_lr)
        

            if (epoch + 1) % self.run_config.validation_frequency == 0:
                img_size, val_loss, val_acc, val_acc5 = self.validate_all_resolution(epoch=epoch, is_test=False, is_adaptive = False)

                is_best = np.mean(val_acc) > self.best_acc
                self.best_acc = max(self.best_acc, np.mean(val_acc))
                val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f} ({4:.3f})'. \
                    format(epoch + 1 - warmup_epoch, self.run_config.n_epochs,
                           np.mean(val_loss), np.mean(val_acc), self.best_acc)
                val_log += '\ttop-5 acc {0:.3f}\tTrain top-1 {top1:.3f}\tloss {train_loss:.3f}\t'. \
                    format(np.mean(val_acc5), top1=train_top1, train_loss=train_loss)

                for i_s, v_a in zip(img_size, val_acc):
                      val_log += '(%d, %.3f), ' % (i_s, v_a)

                self.write_log(val_log, prefix='valid', should_print=False)
            else:
                is_best = False

            self.save_model({
                'epoch': epoch,
                'best_acc': self.best_acc,
                'optimizer': self.optimizer.state_dict(),
                'state_dict': self.network.state_dict(),
            }, is_best=is_best)

        return self.net

    def reset_running_statistics(self, net=None):
        from ofa.elastic_nn.utils import set_running_statistics
        if net is None:
            net = self.network
        sub_train_loader = self.run_config.random_sub_train_loader(2000, 100)
        set_running_statistics(net, sub_train_loader)

    def adaptive_validate(self, epoch=0, is_test=True, run_str='', net=None, data_loader=None, no_logs=False):

        if net is None:
            net = self.net

        if isinstance(net, nn.DataParallel):
            net.module.set_inference(False)
        else:
            net.set_inference(False)


        #torch.device('cpu')
        #picks this one
        if not isinstance(net, nn.DataParallel):
            net = nn.DataParallel(net)
            
            
            #print("second")

        if data_loader is None:
            if is_test:
                data_loader = self.run_config.test_loader
            else:
                data_loader = self.run_config.valid_loader
        
        
        #print(type(net))
        #print("at line 378")
        net.eval()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        utils = []      

        with torch.no_grad():
            with tqdm(total=len(data_loader),
                      desc='Validate Epoch #{} {}'.format(epoch + 1, run_str), disable=no_logs) as t:
                #print("loop progress")
                for i, (images, labels) in enumerate(data_loader):
                    images, labels = images.to(self.device), labels.to(self.device)

                    output, counts = net(images)
                    

                    loss = self.test_criterion(output, labels)

                    # utils preparation
                    if len(utils)==0:
                        n_exit = len(counts)
                        for i in range(n_exit):
                          utils.append(AverageMeter())
                    #print("line 403")
                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(output, labels, topk=(1, 5))

                    for i,c in enumerate(counts):
                       utils[i].update(c/images.size(0), images.size(0))
                   # print("line 409")
                    avg_utils = []
                    for i,u in enumerate(utils):
                        avg_utils.append(u.avg)

                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0].item(), images.size(0))
                    top5.update(acc5[0].item(), images.size(0))
                    #print("line 417")
                    t.set_postfix({
                        'loss': losses.avg,
                        'top1': top1.avg,
                        'top5': top5.avg,
                        'utils': avg_utils,
                        'img_size': images.size(2),
                    })
                    t.update(1)
                    #print("line 426")
        
        avg_utils = []
        for u in utils:
            avg_utils.append(u.avg)

        return losses.avg, top1.avg, top5.avg, avg_utils

    def validate_all_resolution(self, epoch=0, is_test=True, is_adaptive = False, net=None):
        if net is None:
            net = self.network
        if isinstance(self.run_config.data_provider.image_size, list):
            img_size_list, loss_list, top1_list, top5_list = [], [], [], []
            for img_size in self.run_config.data_provider.image_size:
                img_size_list.append(img_size)
                self.run_config.data_provider.assign_active_img_size(img_size)
                self.reset_running_statistics(net=net)
                loss, top1, top5 = self.adaptive_validate(epoch, is_test, net=net)
                loss_list.append(loss)
                top1_list.append(top1)
                top5_list.append(top5)
            return img_size_list, loss_list, top1_list, top5_list
        else:
            if(is_adaptive):
                # loss, top1, top5, util = self.adaptive_validate(epoch, is_test, net=net)
                loss, top1, top5, _ = self.adaptive_validate_thresholds(epoch, is_test, net=net)
            else:
                loss, top1, top5 = self.validate(epoch, is_test, net=net)
            return [self.run_config.data_provider.active_img_size], [loss], [top1], [top5]

    def adaptive_validate_thresholds(self, epoch=0, is_test=True, run_str='', net=None, data_loader=None, no_logs=False):
        if net is None:
            net = self.net
        net.set_inference(True) 

        thresholds= net.threshold 
        initialized_thresholds = net.initialized_thresholds

        if data_loader is None:
            if is_test:
                data_loader = self.run_config.test_loader
            else:
                data_loader = self.run_config.valid_loader

        if (epoch== self.run_config.n_epochs - 1): 
            thresholds, best_acc,best_avg,utils = self.parallel_find_best_thresholds(net, data_loader)  # find the best thresholds for early exits
            net.set_thresholds(thresholds)  # set the thresholds in the network
            self.info['top1'] = best_acc
            self.info['avg_macs'] = best_avg
            self.info['util'] = utils
    
        if not isinstance(net, nn.DataParallel):
            net = nn.DataParallel(net)

        net.eval()

        losses = AverageMeter()
        top1 = AverageMeter()
        utils = np.zeros(len(thresholds)+1)  # to store utils for each exi

        correct_with_thresholds = 0
        total_samples = 0

        with torch.no_grad():
            with tqdm(total=len(data_loader),desc='Validate Epoch #{} {}'.format(epoch + 1, run_str), disable=no_logs) as t:

                for i, (images, labels) in enumerate(data_loader):
                    images, labels = images.to(self.device), labels.to(self.device)

                    preds = net(images)
            
                    loss = 0
                    acc1 = 0

                    for p in preds:
                        loss += self.train_criterion(p, labels)
                        # Calculate top-1 accuracy
                        pred_labels = p.argmax(dim=1)
                        correct = pred_labels.eq(labels).sum().item()
                        acc1 += 100* (correct / labels.size(0)) 

                    acc1 = acc1 / len(preds)  # average acc of all the exits
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1, images.size(0))

                    for image in range(len(images)):
                        exited = False
                        total_samples += 1
                        disabled_exit = 0
                        for exit_id in range(len(thresholds)):  
                            if initialized_thresholds[exit_id] == 1.0:  # if the threshold is 1.0, skip this exit
                                disabled_exit += 1
                                continue
                            logits = preds[exit_id-disabled_exit][image]
                            # conf = torch.softmax(logits, dim=0).max().item()
                            conf = self.confidence(logits) 
                            if conf >= thresholds[exit_id]:
                                exited=True
                                utils[exit_id] += 1  # increment the exit usage
                                correct_with_thresholds += (logits.argmax() == labels[image]).item()
                                break  # early exit
                        if not exited:
                            utils[-1] += 1  # increment the last exit usage
                            logits = preds[-1][image]
                            correct_with_thresholds += (logits.argmax() == labels[image]).item()    

                    t.set_postfix({'loss': losses.avg,'top1': top1.avg,'top1__th': (correct_with_thresholds / total_samples * 100) ,'utils': (utils/total_samples * 100)})
                    t.update(1)

                                        
                correct_with_thresholds = correct_with_thresholds / total_samples * 100  # calculate accuracy with thresholds
                utils = utils / total_samples * 100  # calculate utils for each exit
        
        net.module.set_inference(False) 
        return losses.avg, top1.avg,0 , utils

    def get_score_margin(outputs):
        prob = F.softmax(outputs)
        top2_prob, top2_index = torch.topk(prob,2) 
        score_margin = torch.diff(top2_prob,dim=1)*(-1)
        return score_margin
    
    def confidence(self,x):
        # conf = torch.softmax(x, dim=0).max().item()
        # return conf
        prob = F.softmax(x)
        top2_prob, top2_index = torch.topk(prob,2) 
        sm = torch.diff(top2_prob)*(-1)
        return sm 

    def parallel_find_best_thresholds(self, net, val_loader, step=0.15, lamda=0.1):
          
            net.eval()  # set the network to evaluation mode
            net.set_inference(True)  # set inference mode for early exits
            print("Updating  thethresholds for early exit branches...")
            initialized_thresholds = net.initialized_thresholds
            num_exits = net.total_number_of_exits
            
            print(f"DEBUG: num_exits = {num_exits}")
            print(f"DEBUG: initialized_thresholds = {initialized_thresholds}")
            print(f"DEBUG: len(initialized_thresholds) = {len(initialized_thresholds)}")

            threshold_macs={}
            def threshold_to_macs(candidate_thresholds=None,initialized_thresholds=None):
                th= [0.1,0.1,0.1,0.1]
                for i in range(len(initialized_thresholds)):
                    if initialized_thresholds[i] == 1.0:
                            th[i] = 1.0
                if len(threshold_macs) == 0:
                    str_th= str(th)
                    threshold_macs[str_th]=self.info['macs']
                    return self.info['macs']
                else:
                    for i in range(len(candidate_thresholds)):
                        if candidate_thresholds[i]==1:
                            th[i] = 1.0
                    str_th= str(th)
                    if str_th in threshold_macs:
                        return threshold_macs[str_th]
                    else:   
                        # print("Calculating macs for thresholds:", th)
                        # get_adapt_net_info(subnet, (input_channel, resolution, resolution), measure_latency=measure_latency,print_info=False, clean=True, lut=lut, pmax = pmax, fmax = fmax, amax = amax, wp = wp, wf = wf, wa = wa, penalty = penalty)
                        net2=copy.deepcopy(net)
                        net2.set_thresholds(th)
                        net2.set_inference(False)
                        net_info = utils.get_adapt_net_info(net2, self.input_shape,print_info=False)
                        macs = []
                        for m in net_info['macs']:
                            macs.append(np.round(m / 1e6, 2))
                        threshold_macs[str_th]=macs
                        return macs
            threshold_to_macs(candidate_thresholds=None, initialized_thresholds=initialized_thresholds)
                    
                

            all_data = []  # (label, [(conf, pred), ... per exit])
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    batch_preds = net(images)

                    for i in range(images.size(0)):
                        sample = []
                        for exit_id, logits in enumerate(batch_preds):
                            output = logits[i].cpu()  
                            conf = self.confidence(output)
                            pred = output.argmax().item()
                            sample.append((conf, pred))
                        all_data.append((labels[i].cpu().item(), sample))

            num_exits = net.total_number_of_exits
            best_score = float('-inf')
            best_thresholds = None
            best_utils = None
            best_acc = 0
            best_avg_macs = 0
            best_norm_avg_macs = 0

            checked_thresholds = set()  
            candidates = []  # to store candidate thresholds for parallel evaluation
            # Grid search over thresholds for early exits (excluding final exit)
            
            # FIX: Define threshold_values first
            step = 0.15
            threshold_values = np.arange(0.1, 1.0 + step, step)
            
            # Use custom range if available, else use default
            if hasattr(self, 'threhold_range') and self.threhold_range is not None:
                threshold_values = self.threhold_range
            
            print(f"DEBUG: threshold_values = {threshold_values}")
            print(f"DEBUG: will generate {len(list(product(threshold_values, repeat=num_exits)))} threshold combinations")
            
            # SAFETY CHECK: If no early exits, skip threshold optimization
            if num_exits == 0:
                print("WARNING: Network has 0 early exits. Skipping threshold optimization.")
                current_thresholds = list(net.threshold) if hasattr(net, 'threshold') else [1.0]
                current_utils = np.zeros(len(current_thresholds) + 1)
                current_utils[-1] = 1.0
                current_avg_macs = self.info.get('macs', [1.0])[0] if 'macs' in self.info else 1.0
                return current_thresholds, self.info.get('top1', 0), current_avg_macs, current_utils
            
            for thresholds in product(threshold_values, repeat=num_exits):
                thresholds = list(thresholds)

                for i in range(num_exits):
                    if initialized_thresholds[i] == 1:
                        thresholds[i] = 1.0
                if tuple(thresholds) in checked_thresholds:
                    continue

                is_static=0
                for exit_id in range(num_exits):
                    if thresholds[exit_id] == 1:  # if the threshold is 1.0, skip this exit
                        is_static+=1
                if is_static == num_exits:  # if all exits are static, return a low score
                    continue
                
                macs_list=threshold_to_macs(candidate_thresholds=thresholds, initialized_thresholds=initialized_thresholds)
                

                checked_thresholds.add(tuple(thresholds))
                candidates.append((thresholds, num_exits, initialized_thresholds,  macs_list, self.target_macs, self.alpha_macs))
            
            global validation_conf_data
            validation_conf_data = all_data

            from multiprocessing import Pool, cpu_count
            print(f"Total configurations to evaluate: {len(checked_thresholds)}")
            print(f"Total valid candidates (after filtering): {len(candidates)}")
            
            if len(candidates) == 0:
                # No valid threshold candidates were generated. Keep the current thresholds
                # and return a conservative fallback rather than crashing.
                print("WARNING: No valid threshold candidates generated. Using current network thresholds.")
                current_thresholds = list(net.threshold) if hasattr(net, 'threshold') else [0.5] * num_exits
                current_utils = np.zeros(num_exits + 1)
                current_utils[-1] = 1.0
                current_avg_macs = threshold_to_macs(candidate_thresholds=current_thresholds, initialized_thresholds=initialized_thresholds)
                return current_thresholds, self.info.get('top1', 0), current_avg_macs, current_utils

            # Try parallel evaluation first, fall back to sequential if it fails
            results = []
            use_sequential = False
            
            try:
                with Pool(processes=min(20, cpu_count())) as pool:
                    results = pool.map(evaluate_thresholds, candidates, chunksize=max(1, len(candidates)//20))
                    
                # Check if results are empty, which indicates a silent failure in workers
                if len(results) == 0:
                    print("WARNING: Multiprocessing pool returned empty results. Switching to sequential evaluation.")
                    use_sequential = True
            except Exception as e:
                print(f"WARNING: Multiprocessing failed with error: {e}")
                print("Switching to sequential evaluation.")
                use_sequential = True
            
            # Fall back to sequential evaluation if parallel failed
            if use_sequential:
                print(f"Evaluating {len(candidates)} threshold configurations sequentially...")
                results = []
                for i, candidate_args in enumerate(candidates):
                    try:
                        result = evaluate_thresholds(candidate_args)
                        results.append(result)
                        if (i + 1) % 100 == 0:
                            print(f"  Completed {i + 1}/{len(candidates)} configurations")
                    except Exception as e:
                        print(f"  Error evaluating configuration {i}: {e}")
                        continue
                
                if len(results) == 0:
                    print("ERROR: Sequential evaluation also failed or produced no results.")
                    print("Using current network thresholds as fallback.")
                    current_thresholds = list(net.threshold) if hasattr(net, 'threshold') else [0.5] * num_exits
                    current_utils = np.zeros(num_exits + 1)
                    current_utils[-1] = 1.0
                    current_avg_macs = threshold_to_macs(candidate_thresholds=current_thresholds, initialized_thresholds=initialized_thresholds)
                    return current_thresholds, self.info.get('top1', 0), current_avg_macs, current_utils

            print(f"results {results}")
            print(f"candidates {candidates}")
            print(f"threshold_values {threshold_values}")
            print(f"checked_thresholds {checked_thresholds}")
            best = max(results, key=lambda x: x[0])
            best_score,best_thresholds, best_acc, best_avg_macs, best_macs_list, best_utils=best
            

                    
            for i in range(len(best_thresholds)):
                best_thresholds[i]=round(best_thresholds[i], 1)
                if best_thresholds[i] > 0.95:
                    best_thresholds[i]=1

            # print("Best score:", best_score)
            # print("Best accuracy:", best_acc)
            # print("Best avg_macs:", best_avg_macs, best_macs_list, 'info macs:', self.info['macs'])
            # print("Exit utils:", best_utils)
            # print("Best thresholds:", best_thresholds)

            return best_thresholds ,best_acc, best_avg_macs, best_utils

validation_conf_data=None

def evaluate_thresholds(args):
                global validation_conf_data
                try:
                    if validation_conf_data is None or len(validation_conf_data) == 0:
                        raise RuntimeError("validation_conf_data is None or empty in worker process")
                    
                    thresholds, num_exits, initialized_thresholds, macs_list, target_macs, alpha_macs = args
                    utils = np.zeros(num_exits+1)  # to store utils for each exit
                    total_samples = 0
                    correct_with_thresholds = 0
                    norm_avg_macs = 0

                    for label, conf_pred_list in validation_conf_data:
                        disabled_exit=0
                        exited=False
                        total_samples+=1
                        for exit_id in range(num_exits):
                            if initialized_thresholds[exit_id] == 1:
                                disabled_exit += 1
                                continue
                            conf, pred = conf_pred_list[exit_id-disabled_exit]
                            if conf >= thresholds[exit_id]:
                                utils[exit_id] += 1
                                exited = True
                                if pred == label:
                                    correct_with_thresholds += 1
                                break
                        if not exited:
                            # Go to final exit
                            utils[-1] += 1
                            conf, pred = conf_pred_list[-1]
                            if pred == label:
                                correct_with_thresholds += 1

                    utils = utils / total_samples 
                    avg_macs=0
                    disabled_exit = 0
                    for i in range(len(utils)):
                        if i < len(initialized_thresholds):
                            if initialized_thresholds[i] == 1:  # if the threshold is 1.0, skip this exit
                                disabled_exit+=1
                                continue
                            if thresholds[i] == 1:
                                disabled_exit+=1
                                continue
                        avg_macs+= macs_list[i-disabled_exit] * utils[i]

                    desired_macs=target_macs
                    MAEP_error = ((avg_macs - desired_macs)/ desired_macs)  
                    norm_avg_macs= np.absolute(MAEP_error)

                    acc = 100.0 * correct_with_thresholds / total_samples
                    utils = utils * 100  # calculate utils for each exit
                    lamda = 0.1
                    
                    score = acc - lamda * (norm_avg_macs*100)  # example: penalize higher normalized macs

                    return score, thresholds, acc, avg_macs, macs_list, utils
                    
                except Exception as e:
                    import traceback
                    error_msg = f"Error in evaluate_thresholds: {str(e)}\n{traceback.format_exc()}"
                    print(error_msg, flush=True)
                    raise
