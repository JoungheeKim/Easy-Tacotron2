from torch import nn
import torch


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss
    
    
class NewTacotron2Loss(nn.Module):
    def __init__(self):
        super(NewTacotron2Loss, self).__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.bce = nn.BCEWithLogitsLoss()
        self.loss_masking=True
        self.gate_masking=True
        

    def forward(self, model_output, targets):
        mel_target, gate_target, output_lengths = targets[0], targets[1], targets[2]
        mel_out, mel_out_postnet, gate_out, _ = model_output
        
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        
        
        ## generate mel mask : [B, n_mel_channels, T]
        B = mel_out.size(0)
        N = mel_out.size(1) ## n_mel_channels
        max_len = output_lengths.max()
        ids = torch.arange(max_len, device=output_lengths.device).expand(B, N, max_len)
        mel_mask = ~(ids < output_lengths.view(B, 1, 1)).transpose(1, 2).bool()
        
        ## 1. + 2. mel
        pred = mel_out.transpose(1, 2)
        post_pred = mel_out_postnet.transpose(1, 2)
        label = mel_target.transpose(1, 2)
        if self.loss_masking:
            pred.data.masked_fill_(mel_mask, 0.0)
            post_pred.data.masked_fill_(mel_mask, 0.0)
        
        predicted_mse = self.mse(pred, label)
        predicted_mse = torch.nan_to_num(predicted_mse)

        predicted_mae = self.mae(pred, label)
        predicted_mae = torch.nan_to_num(predicted_mae)

        ## 3. + 4. postnet mel
        predicted_postnet_mse = self.mse(post_pred, label)
        predicted_postnet_mse = torch.nan_to_num(predicted_postnet_mse)

        predicted_postnet_mae = self.mae(post_pred, label)
        predicted_postnet_mae = torch.nan_to_num(predicted_postnet_mae)

        ## 5. gate
        g_ids = torch.arange(max_len, device=output_lengths.device).expand(B, max_len)
        gate_mask = ~(g_ids < output_lengths.view(B, 1)).bool()
        gate_pred = gate_out
        if self.gate_masking:
            gate_pred.data.masked_fill_(gate_mask, 1e3)
        
        predicted_gate_bce = self.bce(gate_pred, gate_target)
        predicted_gate_bce = torch.nan_to_num(predicted_gate_bce)

        loss = predicted_mse + predicted_mae + predicted_postnet_mse + predicted_postnet_mae + predicted_gate_bce
        
        return loss
