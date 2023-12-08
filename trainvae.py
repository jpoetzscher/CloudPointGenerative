import os
import argparse
import torch
import torch.nn.functional as F
from torch import nn
#import torch.utils.tensorboard
from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
from dataset import *
from vae import *
from emd import EMD_CD
from emd import distChamfer


dataset_path = 'ShapeNetCore.v2.PC15k 2/'
category = 'cap'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
train_batch_size = 2
val_batch_size = 32
max_grad_norm = 10
lr=0.001
end_lr=0.0001
input_points = 1024
latent_dim = 64
kl_beta = 0.1
start_epoch = 2000
end_epoch = 8000
max_iters = float('inf')
val_frequency = 1000
val_batch = 32

model = VAE(input_points*3, latent_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr)

def get_data_iterator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, data in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

"""
def lr_func(epoch):
    if epoch <= 0:
        return 1.0
    elif epoch <= 100:
        total = end_epoch - start_epoch
        delta = epoch - start_epoch
        frac = delta / total
        return (1-frac) * 1.0 + frac * (end_lr / lr)
    else:
        return end_lr / lr
    #return LambdaLR(optimizer, lr_lambda=lr_func)
"""
def get_linear_scheduler(optimizer, start_epoch, end_epoch, start_lr, end_lr):
    def lr_func(epoch):
        if epoch <= start_epoch:
            return 1.0
        elif epoch <= end_epoch:
            total = end_epoch - start_epoch
            delta = epoch - start_epoch
            frac = delta / total
            return (1-frac) * 1.0 + frac * (end_lr / start_lr)
        else:
            return end_lr / start_lr
    return LambdaLR(optimizer, lr_lambda=lr_func)
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=10)
scheduler = get_linear_scheduler(optimizer,
    start_epoch=start_epoch,
    end_epoch=end_epoch,
    start_lr=lr,
    end_lr=end_lr
)

train_dset = ShapeNetCore(path=dataset_path, cat_name=category, split='train', downsample=input_points, transform=None)

val_dset = ShapeNetCore(path=dataset_path, cat_name=category, split='val', downsample=input_points, transform=None)

train_iter = get_data_iterator(DataLoader(train_dset, batch_size=train_batch_size, shuffle=True, num_workers=0))

#maybe make shuffle false?
val_loader = DataLoader(val_dset, batch_size=val_batch_size, shuffle=True, num_workers=0)

kl_div_loss = nn.KLDivLoss()


#overfit initially

batch = next(train_iter)
print(batch)
def train(it):
    # Load data
    #batch = next(train_iter)
    x = batch['pointcloud'].to(device)
    #print(x.shape)
    # Reset grad and model state
    optimizer.zero_grad()
    model.train()

    # Forward
    reconstruction, mu, log_var = model(x)
    #print(x.shape)
    #print("RECONSTRUCTIONL ", reconstruction.shape)
    dl, dr = distChamfer(reconstruction, x)
    rec_loss = dl.mean(dim=1) + dr.mean(dim=1)
    rec_loss = rec_loss.mean()
    #rec_loss = (reconstruction, x, reduction='mean')
    #print("REC LOSS: ", rec_loss)
    #kl_loss_fn = kl_div_loss(mu, sigma)
    kl_loss = torch.sum((1+log_var-mu.pow(2) - torch.exp(log_var)),axis=1)
    kl_loss = kl_loss.mean()
    #kl_loss = torch.mean(0.5*)
    loss = rec_loss - kl_beta*kl_loss
    #print("LOSS: ", loss)

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()

    print('[Train] Iter %04d | Loss %.6f | Grad %.4f ' % (it, loss.item(), orig_grad_norm))
    #print('train/loss', loss, it)
    #writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    #writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    #writer.flush()

def validate_loss(it):

    all_refs = []
    all_recons = []
    for i, batch in enumerate(tqdm(val_loader, desc='Validate')):
        if val_batch > 0 and i >= val_batch:
            break
        ref = batch['pointcloud'].to(device)
        shift = batch['shift'].to(device)
        scale = batch['scale'].to(device)
        with torch.no_grad():
            model.eval()
            recons, _, _ = model(ref)
        all_refs.append(ref * scale + shift)
        all_recons.append(recons * scale + shift)
    all_refs = torch.cat(all_refs, dim=0)
    all_recons = torch.cat(all_recons, dim=0)
    metrics = EMD_CD(all_recons, all_refs, batch_size=val_batch_size)
    cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
    
    print(('[Val] Iter %04d | CD %.6f | EMD %.6f  ' % (it, cd, emd)))
    #writer.add_scalar('val/cd', cd, it)
    #writer.add_scalar('val/emd', emd, it)
    #writer.flush()

    return cd

def validate_inspect(it):
    sum_n = 0
    sum_chamfer = 0
    for i, batch in enumerate(tqdm(val_loader, desc='Inspect')):
        x = batch['pointcloud'].to(device)
        model.eval()
        code = model.encode(x)
        recons = model.decode(code).detach()
        sum_n += x.size(0)
        if i >= 5:
            break   # Inspect only 5 batch

    #writer.add_mesh('val/pointcloud', recons[:args.num_inspect_pointclouds], global_step=it)
    #writer.flush()

# Main loop
print('Start training...')
try:
    it = 1
    while it <= max_iters:
        train(it)
        if it % val_frequency == 0 or it == max_iters:
            with torch.no_grad():
                cd_loss = validate_loss(it)
                validate_inspect(it)
            opt_states = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            torch.save({'model_state_dict': model.state_dict()}, cd_loss, opt_states, step=it)
        it += 1

except KeyboardInterrupt:
    print('Terminating...')