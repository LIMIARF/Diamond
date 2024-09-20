import argparse
import os
import torchvision.transforms as transforms
from PIL import Image
import cv2
import pandas as pd
import torch.nn as nn
import numpy as np
import torch
from model import DiamondModel
from LOSS import CB_loss
from torch.utils.data import DataLoader
from data.DiamondDataset import DiamondDataset
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import prepare_data, generate_val_csv, load_float_from_file, augment_data
from torchmetrics.classification import BinaryCalibrationError, BinaryAUROC, BinaryF1Score
import torch.nn.functional as F
 
def main(args):
    # Determine if CUDA (GPU) is available, otherwise use CPU
    if torch.cuda.is_available():
        print("GPU is used")
    else:
        print("CPU is used")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize TensorBoard writer
    log_dir = f'./logs/{args.submission_name}/{args.backbone}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    # Initialize metrics
    AUC = BinaryAUROC(thresholds=None).to(device)
    F1 = BinaryF1Score().to(device)
    ECE = BinaryCalibrationError(n_bins=10, norm='l1').to(device)
    # Initialize the results directory for this submission
    results_dir = f'./{args.submission_name}'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    submission_best_score_file = os.path.join(results_dir, 'best_score.txt')

    # Load dataframes
    df_train = pd.read_csv(args.train_csv)
    df_val = pd.read_csv(args.val_csv)

    # Data Augmentation
    images_path_list=df_train['image_path'].iloc[:]
    df_train_aug=augment_data(images_path_list ,df_train, args.root_dir, "train_augmented_images", augment=True)
    images_path_list=df_val['image_path'].iloc[:]
    df_val_aug=augment_data(images_path_list ,df_val, args.root_dir, "val_augmented_images", augment=False)

    df_train_aug['next_DME_label'] = df_train_aug['next_DME_label'].astype(int)
    df_val_aug['next_DME_label'] = df_val_aug['next_DME_label'].astype(int)
    train_dataset = DiamondDataset(df_train_aug, mode='train', args=args)
    val_dataset = DiamondDataset(df_val_aug, mode='val', args=args)
    print(train_dataset.__len__())
    
    # Data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Model Initialization
    model = DiamondModel(args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    model.to(device)
    best_loss = 999
    # Checkpoint directory
    ckpt_dir = f'./checkpoints/{args.submission_name}/{args.backbone}'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    for epoch in range(args.epochs):
        AUC.reset()
        F1.reset()
        ECE.reset()
        avg_loss_list = []
        beta = 0.9999
        gamma = 2.0
        samples_per_cls = [696,504]
        loss_type = "focal"
        no_of_classes=2
        # Training step
        model.train()
        with torch.enable_grad():
            for batch_idx, data in enumerate(tqdm(train_dataloader)):
                img, labels = prepare_data(data, device)
                logits = model(img)
                cb_loss = CB_loss(labels, logits, samples_per_cls, no_of_classes,loss_type, beta, gamma)
#                 loss = criterion(logits, labels)
                # Update metrics
                probs = torch.softmax(logits, dim=1)
                AUC.update(probs[:, 1], labels)
                F1.update(probs[:, 1], labels)
                ECE.update(probs[:, 1], labels)
                cb_loss.backward()
                optimizer.step()
                avg_loss_list.append(cb_loss.item())
            avg_loss_train = np.mean(avg_loss_list)
            writer.add_scalar('Loss/Train', avg_loss_train, epoch)
            writer.add_scalar('Metric/AUC_Train', AUC.compute(), epoch)
            writer.add_scalar('Metric/F1_Train', F1.compute(), epoch)
            writer.add_scalar('Metric/ECE_Train', ECE.compute(), epoch)
            print(f"[TRAIN] epoch={epoch}/{args.epochs}, Loss = {avg_loss_train}")
            print(f"[TRAIN] epoch={epoch}/{args.epochs}, F1 Score = {F1.compute().item()}, AUC Score = {AUC.compute().item()}, ECE Score = {ECE.compute().item()}")
        # Validation step
        model.eval()
        AUC.reset()
        F1.reset()
        ECE.reset()
        with torch.no_grad():
            for batch_idx, data in enumerate(val_dataloader):
                img, labels = prepare_data(data, device)
                logits = model(img)
                cb_loss = CB_loss(labels, logits, samples_per_cls, no_of_classes,loss_type, beta, gamma)
                avg_loss_list.append(cb_loss.item())
                # Update metrics for validation
                probs = torch.softmax(logits, dim=1)
                AUC.update(probs[:, 1], labels)
                F1.update(probs[:, 1], labels)
                ECE.update(probs[:, 1], labels)
            avg_loss_val = np.mean(avg_loss_list)
            writer.add_scalar('Loss/Validation', avg_loss_val, epoch)
            writer.add_scalar('Metric/AUC_Val', AUC.compute(), epoch)
            writer.add_scalar('Metric/F1_Val', F1.compute(), epoch)
            writer.add_scalar('Metric/ECE_Val', ECE.compute(), epoch)
            print(f"[VAL] epoch={epoch}/{args.epochs}, Loss = {avg_loss_val}")
            print(f"[VAL] epoch={epoch}/{args.epochs}, F1 Score = {F1.compute().item()}, AUC Score = {AUC.compute().item()}, ECE Score = {ECE.compute().item()}")
        # Save best model
        if avg_loss_val < best_loss:
            print('Saving best model based on Loss, in epoch = ', epoch)
            print('best Val_Loss  =', avg_loss_val)
            best_loss = avg_loss_val
            ep_loss = epoch
            torch.save(model.state_dict(), os.path.join(ckpt_dir, '{args.submission_name}.pth'))
            if best_loss < load_float_from_file(submission_best_score_file, 999):
                with open(submission_best_score_file, 'w') as best_score_file:
                    best_score_file.write(str(best_loss))
                print('Generating the validation CSV using the Best Loss Model !')
                generate_val_csv(model, val_dataloader, device, results_dir)
        print(f"[Best checkpoints values] Best_loss = {best_loss}")
        print(f"[Best checkpoints epochs] epoch = {ep_loss}")
    writer.close() 

if __name__ == "__main__":
    """
    Parses command-line arguments and runs the main training, validation, and testing pipeline for the DiamondModel.

    Command-line arguments include learning rate, weight decay, batch size, number of epochs, paths to dataset files, etc.
    """

    parser = argparse.ArgumentParser(description='Training script for predicting the apparition of central diabetic edema.')
    parser.add_argument('--submission_name', type=str, default='Submission_3', help='Used submission name')
    parser.add_argument('--backbone', type=str, default='resnet50', help='Used Backnone name')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--root_dir', type=str, default='./diamond_data', help='Path to the dataset')
    parser.add_argument('--train_csv', type=str, default='training_set.csv', help='Path to the training CSV')
    parser.add_argument('--val_csv', type=str, default='validation_set.csv', help='Path to the validation CSV')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--weights_pth_file', type=str, default='./backbone/resnet152.pth', help='Weights backbone pth file')

    args = parser.parse_args()
    main(args)
