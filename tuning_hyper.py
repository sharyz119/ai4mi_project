import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ImprovedENet import ImprovedENet, DoubleConv
from common import runTraining, setup, datasets_params
import argparse
from pathlib import Path

class CustomImprovedENet(ImprovedENet):
    def __init__(self, in_channels, num_classes, n_channels, dropout_rate):
        super().__init__(in_channels, num_classes)
        self.enc1 = DoubleConv(in_channels, n_channels[0])
        self.enc2 = DoubleConv(n_channels[0], n_channels[1])
        self.enc3 = DoubleConv(n_channels[1], n_channels[2])
        self.enc4 = DoubleConv(n_channels[2], n_channels[3])
        self.bridge = DoubleConv(n_channels[3], n_channels[3]*2)
        self.dec4 = DoubleConv(n_channels[3]*3, n_channels[3])
        self.dec3 = DoubleConv(n_channels[3] + n_channels[2], n_channels[2])
        self.dec2 = DoubleConv(n_channels[2] + n_channels[1], n_channels[1])
        self.dec1 = DoubleConv(n_channels[1] + n_channels[0], n_channels[0])
        self.final_conv = nn.Conv2d(n_channels[0], num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
            # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
            
            # Bridge
        bridge = self.bridge(self.pool(enc4))
        bridge = self.dropout(bridge)
            
            # Decoder
        dec4 = self.dec4(torch.cat([F.interpolate(bridge, size=enc4.shape[2:], mode='bilinear', align_corners=True), enc4], dim=1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, size=enc3.shape[2:], mode='bilinear', align_corners=True), enc3], dim=1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, size=enc2.shape[2:], mode='bilinear', align_corners=True), enc2], dim=1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, size=enc1.shape[2:], mode='bilinear', align_corners=True), enc1], dim=1))
            
            # Final layer
        final = self.final_conv(dec1)
            
        return final
def create_custom_improved_enet(in_channels, num_classes, n_channels, dropout_rate):
    return CustomImprovedENet(in_channels, num_classes, n_channels, dropout_rate)

def objective(trial, args):
    # Define the hyperparameters to tune
    n_channels = [
        trial.suggest_int("n_channels_1", 32, 128),
        trial.suggest_int("n_channels_2", 64, 256),
        trial.suggest_int("n_channels_3", 128, 512),
        trial.suggest_int("n_channels_4", 256, 1024),
    ]
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-2)
    
    # Create a custom ImprovedENet with the suggested hyperparameters
    
    

    # Create args for runTraining
    # parser = argparse.ArgumentParser(description='Hyperparameter tuning')
    # args = parser.parse_args([])
    # args.epochs = 50  # Adjust as needed
    # args.dataset = 'SEGTHOR'
    # args.mode = 'full'
    # args.dest = Path(f'results/optuna_trial_{trial.number}')
    # args.gpu = torch.cuda.is_available()
    # args.debug = False

    args.dest = Path(f'results/optuna_trial_{trial.number}')

    # Modify setup function to use our custom model
    def custom_setup(args):
        _, _, device, train_loader, val_loader, K = setup(args)
        #net = CustomImprovedENet(1, K).to(device)
        net = create_custom_improved_enet(1, K, n_channels, dropout_rate).to(device)
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        return net, optimizer, device, train_loader, val_loader, K


    # Run training with custom setup
    dice_score = runTraining(args, custom_setup)
    
    return dice_score

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)  # Adjust number of trials as needed

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save study results
    import joblib
    joblib.dump(study, "optuna_study.pkl")