import argparse

import pandas as pd
import numpy as np
from tqdm import tqdm
from utils._utils import make_data_loader
from model import BaseModel

import torch
import torch.nn as nn # edited
from torchvision.models import resnet18, ResNet18_Weights

import optuna
import gc

def acc(pred,label):
    pred = pred.argmax(dim=-1)
    return torch.sum(pred == label).item()

def train(model, optimizer, scheduler, train_loader, valid_loader):
    """
    TODO: Change the training code as you need. (e.g. different optimizer, different loss function, etc.)
            You can add validation code. -> This will increase the accuracy.
    """
    gc.collect()   # cuda memory 부족 방지

    valid_acc_list = []  # 추가: 각 Epoch의 검증 정확도를 저장할 리스트

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optimizer
    scheduler = scheduler

    for epoch in range(args.epochs):
        train_losses = [] 
        train_acc = 0.0
        total=0
        print(f"[Epoch {epoch+1} / {args.epochs}]")
        
        model.train()
        print('training')
        pbar = tqdm(train_loader)
        for i, (x, y) in enumerate(pbar):
            image = x.to(args.device)
            label = y.to(args.device)          
            optimizer.zero_grad()

            output = model(image)
            
            label = label.squeeze()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            total += label.size(0)

            train_acc += acc(output, label)

        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = train_acc/total

        # add valid course.
        model.eval()
        valid_acc = 0.0
        valid_total = 0
        with torch.no_grad():
            print('valid calculating')
            pbar = tqdm(valid_loader)
            for i, (x, y) in enumerate(pbar):
                image = x.to(args.device)
                label = y.to(args.device)
                output = model(image)
                valid_total += label.size(0)
                valid_acc += acc(output, label)
        epoch_valid_acc = valid_acc/valid_total

        valid_acc_list.append(epoch_valid_acc)   # 추가: 리스트에 검증 정확도 추가


        print(f'Epoch {epoch+1}') 
        print(f'train_loss : {epoch_train_loss}')
        print('train_accuracy : {:.3f}'.format(epoch_train_acc*100))
        print('valid_accuracy : {:.3f}'.format(epoch_valid_acc*100))  # Print validation accuracy

        # 검증 세트의 정확도를 반환합니다.

        print("==============================")
        print("Save path:", args.save_path)
        print('Using Device:', device)
        print('Number of usable GPUs:', torch.cuda.device_count())
        
        # Print epoch
        print("Epoch:", epoch)
        print("==============================")
        
    return valid_acc_list

def initializer(model, learning_rate, weight_decay, gamma):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    return model, optimizer, scheduler


def objective(model, trial: optuna.Trial, args):
    # 모델마다 최적의 하이퍼파라미터값은 다르다.
    learning_rate = trial.suggest_float('learning_rate', low=5e-5, high=0.1, log=True)
    weight_decay = trial.suggest_float('weight_decay', low=4e-5, high=0.1, log=True)
    train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [32, 64, 128, 256, 512])
    gamma = trial.suggest_float('gamma', 0, 1)

    # Make Data loader and Model
    args.batch_size = train_batch_size
    train_loader, valid_loader, _ = make_data_loader(args)

    model, optimizer, scheduler = initializer(model, learning_rate, weight_decay, gamma)
    epoch_valid_acc = train(model, optimizer, scheduler, train_loader, valid_loader)

    gc.collect()   # cuda memory 부족 방지
    return max(epoch_valid_acc)   # 검증데이터중 가장 큰 값 return

def save(model, optimizer, scheduler, train_loader):

    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        
        train_losses = [] 
        train_acc = 0.0
        total=0
        print(f"[Epoch {epoch+1} / {args.epochs}]")
        
        model.train()
        pbar = tqdm(train_loader)
        for i, (x, y) in enumerate(pbar):
            image = x.to(args.device)
            label = y.to(args.device)          
            optimizer.zero_grad()

            output = model(image)
            
            label = label.squeeze()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            total += label.size(0)

            train_acc += acc(output, label)

        torch.save(model.state_dict(), f'{args.save_path}/model_epoch{epoch+1}.pth')

def save_intermediate_results(study, trial: optuna.Trial):
    best = study.best_params
    df_best = pd.DataFrame([best])
    df_trials = study.trials_dataframe()
    df_best.to_csv('current_best.csv', index=False)
    df_trials.to_csv('optuna_trials.csv', index=False)

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='2023 DL Term Project')
    parser.add_argument('--save-path', default='checkpoints/', help="Model's state_dict")
    parser.add_argument('--data', default='data/', type=str, help='data folder')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    num_classes = 10 # edited
    
    """
    TODO: You can change the hyperparameters as you wish.
            (e.g. change epochs etc.)
    """
    
    # hyperparameters
    args.epochs = 7

    # custom model
    model = resnet18(weights=ResNet18_Weights)
    
    # you have to change num_classes to 10
    num_features = model.fc.in_features # edited
    model.fc = nn.Linear(num_features, num_classes) # edited
    model.to(device)
    print(model)

    # Training The Model
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(model, trial, args), n_trials=100, callbacks=[save_intermediate_results])

    df = study.trials_dataframe()
    df.to_csv('optuna_trials.csv', index = False)

    # 최적의 하이퍼파라미터 출력
    print(study.best_params)
    best_params = study.best_params
    best_learning_rate = best_params['learning_rate']
    best_weight_decay = best_params['weight_decay']
    best_train_batch_size = best_params['per_device_train_batch_size']
    best_gamma = best_params['gamma']
    args.batch_size = best_train_batch_size

    train_loader, valid_loader, _ = make_data_loader(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_learning_rate, weight_decay=best_weight_decay)  
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=best_gamma)
    save(model, optimizer, scheduler, train_loader)
