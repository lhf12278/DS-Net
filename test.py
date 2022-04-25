# -*- coding: utf-8 -*-
import argparse
from torch.utils.data import DataLoader
from val_data import ValData
from dehaze import *

from utils import validation

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for dehaze')
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-val_data_dir', help='Set the test dir', default='data/test/SOTS/outdoor/', type=str)
args = parser.parse_args()

val_batch_size = args.val_batch_size
val_data_dir = args.val_data_dir

print('--- Hyper-parameters for testing ---')
print('val_batch_size: {}\n'.format(val_batch_size))

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Validation data loader --- #
val_data_loader = DataLoader(ValData(val_data_dir), batch_size=val_batch_size, shuffle=False, num_workers=0)


# --- Define the network --- #
haze_net = DeHaze()

# --- Multi-GPU --- #
haze_net = haze_net.to(device)
haze_net = nn.DataParallel(haze_net, device_ids=device_ids)

# --- Load the network weight --- #
haze_net.load_state_dict(torch.load('checkpoint/outdoor_haze_best'))

# --- Use the evaluation model in testing --- #
haze_net.eval()

print('--- Testing starts! ---')
val_psnr, val_ssim = validation(haze_net, val_data_loader, device, save_tag=True)
print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))



