# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:07:29 2021

@author: chumache
"""
import json
import os
import time
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import transforms
from dataset import get_training_set, get_validation_set, get_test_set
from model import generate_model
from opts import parse_opts
from train import train_epoch
from utils import Logger, adjust_learning_rate, save_checkpoint
from validation import val_epoch
import matplotlib.pyplot as plt
import wandb


class EarlyStopping:
    def __init__(self, patience, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.best_loss = None
        self.epochs_without_improvement = 0

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.epochs_without_improvement = 0
            if self.verbose:
                print(f'Validation loss improved to {validation_loss:.4f}')
        else:
            self.epochs_without_improvement += 1
            if self.verbose:
                print(f'No improvement in validation loss for {self.epochs_without_improvement} epochs')

        return self.epochs_without_improvement >= self.patience


if __name__ == '__main__':
    opt = parse_opts()
    wandb.init(project='emotion-multimodal-recognition', config=opt)
    n_folds = 1
    test_accuracies = []
    train_losses = []
    val_losses = []
    test_losses = []
    train_prec1s = []
    val_prec1s = []
    test_prec1s = []
    early_stopping = EarlyStopping(patience=10, verbose=True)

    if opt.device != 'cpu':
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pretrained = opt.pretrain_path != 'None'

    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)

    opt.arch = '{}'.format(opt.model)
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.sample_duration)])

    for fold in range(n_folds):
        print(opt)
        with open(os.path.join(opt.result_path, 'opts' + str(time.time()) + str(fold) + '.json'), 'w') as opt_file:
            json.dump(vars(opt), opt_file)

        torch.manual_seed(opt.manual_seed)
        model, parameters = generate_model(opt)

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(opt.device)

        if not opt.no_train:
            print("Train 1")
            video_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotate(),
                transforms.ToTensor(opt.video_norm_value)])

            training_data = get_training_set(opt, spatial_transform=video_transform)

            train_loader = DataLoader(
                training_data,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=opt.n_threads,
                pin_memory=False)

            train_logger = Logger(
                os.path.join(opt.result_path, 'train' + str(fold) + '.log'),
                ['epoch', 'loss', 'prec1', 'prec5', 'lr'])
            train_batch_logger = Logger(
                os.path.join(opt.result_path, 'train_batch' + str(fold) + '.log'),
                ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'])

            optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                dampening=opt.dampening,
                weight_decay=opt.weight_decay,
                nesterov=False)
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=opt.lr_patience)

        if not opt.no_val:
            print("Validation 1")
            video_transform = transforms.Compose([
                transforms.ToTensor(opt.video_norm_value)])

            validation_data = get_validation_set(opt, spatial_transform=video_transform)

            val_loader = DataLoader(
                validation_data,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=False)

            val_logger = Logger(
                os.path.join(opt.result_path, 'val' + str(fold) + '.log'), ['epoch', 'loss', 'prec1', 'prec5'])
            test_logger = Logger(
                os.path.join(opt.result_path, 'test' + str(fold) + '.log'), ['epoch', 'loss', 'prec1', 'prec5'])
        best_prec1 = 0
        best_loss = 1e10
        if opt.resume_path:
            print('loading checkpoint {}'.format(opt.resume_path))
            checkpoint = torch.load(opt.resume_path)
            assert opt.arch == checkpoint['arch']
            best_prec1 = checkpoint['best_prec1']
            opt.begin_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])

    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            print("Train 2")
            adjust_learning_rate(optimizer, i, opt)
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger, train_losses, train_prec1s)
            wandb.log({'train_loss': train_losses[-1], 'train_prec1': train_prec1s[-1], 'epoch': i})
            state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
            }
            save_checkpoint(state, False, opt, fold)

        if not opt.no_val:
            print("Validate 2")
            validation_loss, prec1 = val_epoch(i, val_loader, model, criterion, opt, val_logger)
            val_losses.append(validation_loss)
            val_prec1s.append(prec1)
            wandb.log({'val_loss': val_losses[-1], 'val_prec1': val_prec1s[-1], 'epoch': i})
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
            }
            save_checkpoint(state, is_best, opt, fold)
            # Controllo dell'early stopping
            '''
              if early_stopping(validation_loss):
                print("Early stopping triggered")
                break  # Esci dal ciclo se non ci sono miglioramenti
            '''
    if opt.test:
        test_logger = Logger(
            os.path.join(opt.result_path, 'test' + str(fold) + '.log'), ['epoch', 'loss', 'prec1', 'prec5'])

        video_transform = transforms.Compose([
            transforms.ToTensor(opt.video_norm_value)])

        test_data = get_test_set(opt, spatial_transform=video_transform)

        # Load best model
        best_state = torch.load('%s/%s_best' % (opt.result_path, opt.store_name) + str(fold) + '.pth')
        model.load_state_dict(best_state['state_dict'])

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)

        test_loss, test_prec1 = val_epoch(10000, test_loader, model, criterion, opt, test_logger)
        test_losses.append(test_loss)
        test_prec1s.append(test_prec1)

        with open(os.path.join(opt.result_path, 'test_set_bestval' + str(fold) + '.txt'), 'a') as f:
            f.write('Prec1: ' + str(test_prec1) + '; Loss: ' + str(test_loss))

        test_accuracies.append(test_prec1)

        # Log to wandb
        wandb.log({'test_loss': test_losses[-1], 'test_prec1': test_prec1s[-1], 'epoch': 10000})

    # Tracciamento dei grafici
    plt.figure(figsize=(18, 5))

    # Grafico della perdita
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(range(len(train_losses), len(train_losses) + len(test_losses)), test_losses, label='Test Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Grafico dell'accuratezza
    plt.subplot(1, 3, 2)
    plt.plot(train_prec1s, label='Training Accuracy')
    plt.plot(val_prec1s, label='Validation Accuracy')
    plt.plot(range(len(train_prec1s), len(train_prec1s) + len(test_prec1s)), test_prec1s, label='Test Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Grafico della perdita e accuratezza di test
    plt.subplot(1, 3, 3)
    plt.plot(test_losses, label='Test Loss')
    plt.plot(test_prec1s, label='Test Accuracy')
    plt.title('Test Loss and Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()

    # Salva i grafici
    plt.savefig(os.path.join(opt.result_path, 'training_validation_test_graphs.png'))
    plt.show()
    wandb.finish()
