import os
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import random
from pathlib import Path
from tqdm import tqdm
from timeit import default_timer as timer

print(torch.__version__)
torch.manual_seed(42)
np.random.seed(seed=42)
random.seed(42)
c_elegans = True

if c_elegans:
    vRead = iio.imread('data/c_elegans.mp4')
    video = np.array(vRead)
    video = torch.as_tensor(video)
    testFrame = video[0]
    ### Crop letters from video
    croppedVideo = torch.zeros((2484, 270, 344,3)).type(torch.float32)
    for i in range(len(video)):
        croppedVideo[i] = video[i][18:288]
    #Normalize range of RGBs
    croppedVideo = torch.mul(croppedVideo, 1.0/255.0).type(torch.float32)
    video = croppedVideo[10:]
else:
    print("custom file input")
    #other custom file processing for later use
#Set device
print("Cuda available: ", torch.cuda.is_available())
if(torch.cuda.is_available()):
    torch.cuda.set_device("cuda:0")
    print("Is cuDNN version:", torch.backends.cudnn.version())
    print("cuDNN enabled:a", torch.backends.cudnn.enabled)
    print("Device count: ", torch.cuda.device_count())
    print("Current device: ", torch.cuda.current_device())
    print("Device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
#Setup device agnostic code (i.e use GPU if possible)
device = "cuda" if torch.cuda.is_available() else "cpu"
video = video.to(device)
print(device)

#Create Model
class hashNerf(nn.Module):
    def __init__(self,
                input_shape: int,
                hidden_units: int,
                output_shape: int,
                L=16, T=2**18, F=2, N_min=16, N_max=1024, num_output=3):
        self.L = L
        self.T = T
        self.F = F
        self.N_min = N_min
        self.N_max = N_max
        self.num_output = num_output
        super().__init__()

        b = np.exp((np.log(self.N_max) - np.log(self.N_min)) / (self.L - 1)) # scalar : dim 1
        self.N_values = torch.floor(torch.tensor(np.float32(self.N_min * b ** np.arange(self.L)))).type(torch.int64).to(device) # dim : 16,1
        temp_hash_table = torch.FloatTensor(L,T,F).uniform_(-1e-4, 1e-4) # dim : L, T, F
        self.hash_table = nn.Parameter(torch.tensor(temp_hash_table.clone(), requires_grad=True).to(device).type(torch.float32))
        self.vertices = torch.transpose(torch.tensor([
                        [0,0],
                        [0,1],
                        [1,0],
                        [1,1]]), 0,1).type(torch.float32).to(device) # dim : (2,4)
        self.prime_numbers = torch.tensor([1, 2654435761]).type(torch.int64).to(device)

        self.layer_stack = nn.Sequential(
            nn.Linear(input_shape, hidden_units),
            nn.LeakyReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.LeakyReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.LeakyReLU(),
            nn.Linear(hidden_units, output_shape),
            nn.ReLU()
        ).to(device)
    
    def forward(self, X):
        # X : 2D coordinates -> (num_points, 2)
        X = X.to(device)
        x_shape = X.shape
        x_scaled = X[:, :, None] * self.N_values[None, None, :]
        x_floor = torch.floor(x_scaled) # dim : (batch_size, 2, L)
        x_ceil = torch.ceil(x_scaled) # dim : (batch_size, 2, L)
        x_vertices = torch.zeros((len(x_floor), 2, self.L, 4)).to(device) # dim : (batch_size, 2, L, 4)
        x_vertices = x_floor[:, :, :, None] * self.vertices[None, :, None, :]
        x_vertices = x_vertices.type(torch.int64)
        
        primesTemp = torch.tensor([1, 2654435761]).type(torch.int64).to(device)
        x_to_hash_temp = x_vertices * primesTemp[None, :, None, None]

        x_hash_temp = torch.bitwise_xor(x_to_hash_temp[:, 0],
                                        x_to_hash_temp[:, 1]) # shape: num_points, L, 4
        x_hash_temp = torch.remainder(x_hash_temp, self.T) 
        x_hash = x_hash_temp
        x_hash = x_hash.to(device)
        x_hash = x_hash.permute(1, 0, 2)  # shape: L, num_points, 4
        lookup = torch.stack([self.hash_table[i][x_hash[i]] for i in range(self.L)], dim=0).to(device)
        lookup = lookup.permute(1,0,2,3)  # shape: num_points, L, 4, F
        # linear interpolation
        weights = x_scaled - x_floor
        fx = weights[:, 0, :]
        cx = 1 - fx
        fy = weights[:, 1, :]

        cy = 1 - fy
        f11 = fx * fy
        f10 = fx * cy
        f01 = cx * fy
        f00 = cx * cy
        f_stack = torch.stack([f00, f01, f10, f11], dim=2).to(device)  # shape: num_points, L, 4
        x_interp = torch.sum(f_stack[:, :, :, None] * lookup, dim=2)  # shape: num_points, L, F
        x = x_interp.reshape(-1, self.L * self.F) # dim : num_points, L*F
        return self.layer_stack(x.to(device).type(torch.float32))
    
# Data Loader
class SingleImageDataset(Dataset):
    def __init__(self, image, transform=None, target_transform=None):
        self.image = image.type(torch.float32)
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return int(self.image.shape[0]) * int(self.image.shape[1])
    def __getitem__(self, idx):
        row = idx // int(self.image.shape[1])
        col = idx % int(self.image.shape[1])
        pixel = torch.as_tensor(self.image[row][col]).type(torch.float32).to(device)
        #label = pixel 
        row = row / (self.image.shape[0]-1)
        col = col / (self.image.shape[1]-1)
        return torch.as_tensor([row, col]).type(torch.float32).to(device), pixel
'''
#DataLoader Sanity Check!
testGrid = torch.empty(video[currFrame].shape[0], video[currFrame].shape[1], 3).cpu()
testGrid = testGrid.type(torch.float32)
train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True)
for batch in iter(train_dataloader):
    coords = batch[0].cpu()
    pixels = batch[1]
    for i in range(0, len(coords)):
        frameShape = workingFrame.shape
        row = int(coords[i][0] * (workingFrame.shape[0]-3))
        col = int(coords[i][1] * (workingFrame.shape[1]-3))
        # No normalization
     
        pixel = pixels[i]
        testGrid[row][col] = pixel
#testGrid = torch.mul(testGrid, 1)
plt.imshow(testGrid.cpu());
plt.axis(False);
'''
## Utils
import math
from pathlib import Path
def PSNR(MSELoss, max):
    return (20*math.log10(max)) - (10*math.log10(MSELoss))

def saveModel(modelPointer, psnr_note=35, frameNumber=0):
    # 1. Create models directory - won't create if it exists
    MODEL_PATH = Path("c_elegans_models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = "hash_nerf_frame_"+str(frameNumber)+"_psnr_"+str(psnr_note)+".pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    MODEL_SAVE_PATH
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=modelPointer.state_dict(),
            f=MODEL_SAVE_PATH)
    print(MODEL_SAVE_PATH)

#prologue to saveImage


def saveImage(frame_note, psnr_note=39):
    model_0.eval()
    x_test_T = torch.as_tensor(croppedVideo[100])
    reconstruction_input_matrix = torch.zeros(croppedVideo[100].shape[0], croppedVideo[100].shape[1], 2).type(torch.float32)
    #encode coordinates into debug matrix
    for i in range(0, x_test_T.shape[0]):
        for j in range(0, x_test_T.shape[1]):
            reconstruction_input_matrix[i][j] = torch.as_tensor([i/(video.shape[1]-1.0),j/(video.shape[2]-1.0)]).type(torch.float32)
    reconstruction_input_matrix = torch.flatten(reconstruction_input_matrix, 0, 1)
    im_psnr_note = 40
    model_0.eval()
    with torch.inference_mode():
        reconstruction = model_0(reconstruction_input_matrix).cpu()
        reconstruction = reconstruction.reshape((croppedVideo[100].shape[0], croppedVideo[100].shape[1],3))
        reconstruction = torch.mul(reconstruction, 255.0).type(torch.int32)
        plt.imshow(reconstruction)
        plt.axis(False)
        plt.savefig("c_elegans_reconstructions_ex1/hash_nerf_reconstruction_frame_"+str(frame_note)+"_psnr_"+str(psnr_note)+".png", bbox_inches="tight", pad_inches=0.0)
        plt.close()
# initialize model
init = hashNerf(32, 128, 3)
#replace with len of video
for t in range(0,100):
    if t == 0:
        model_0 = hashNerf(32, 128, 3)
    else:
        model_0 = init
    workingFrame = video[t]
    training_data = SingleImageDataset(workingFrame)
    lr1 = 0.01
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=lr1, eps=10e-15)
    # Training Loop
    start = timer()
    PSNR_thresh = 39.00
    train_loader = DataLoader(training_data, batch_size=2**14, shuffle=True)
    batchCount = 0
    psnr_table = []
    savedAt25 = False
    savedAt30 = False
    exit_loop = False
    for epoch in tqdm(range(0,100)):
        model_0.train()
        for batch in iter(train_loader):
            batchCount += 1
            y_train = torch.as_tensor(batch[1]).to(device)
            y_train = torch.squeeze(y_train)
            X = torch.tensor(batch[0]).type(torch.float32)
            y_train = torch.as_tensor(y_train).type(torch.float32)
            # Forward Pass
            y_pred = model_0(X).to(device).type(torch.float32)
            loss = loss_fn(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        psnr = PSNR(loss, 1.0)
        psnr_table.append(psnr)
        if (savedAt25 == False) and (psnr >= 25):
            #saveModel(model_0, 25, 0)
            saveImage(t, 25)
            savedAt25 = True
        elif (savedAt30 == False) and (psnr >= 30):
            lr1 = lr1/10
            #saveModel(model_0, 30, 0)
            saveImage(t, 30)
            savedAt30 = True
        elif (psnr >= PSNR_thresh):
            saveImage(t, PSNR_thresh)
            #saveModel(model_0, PSNR_thresh, t)
            break
        if epoch % 1 == 0:
            print(f"Epoch: {epoch} | LR: {lr1} | Train loss: {loss} | PSNR: {psnr}")
    end = timer()
    time_elapsed = end - start
    print('time: ', str(time_elapsed))
    print('Training Finished')
    #Plot PSNR
    plt.plot(range(0,len(psnr_table)), psnr_table)
    plt.title('Train PSNR - Frame: '+str(t))
    plt.ylabel('PSNR')
    #NOT EPOCHS - THESE ARE BATCHES!!!
    plt.xlabel('Epoch')
    plt.savefig("c_elegans_psnr_plots_ex1/hash_nerf_reconstruction_frame_"+str(t)+".png", bbox_inches="tight")
    plt.close()
    if t == 0:
        init = model_0 # set initial model for later use
    






