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
from pthflops import count_ops
from tqdm import tqdm

print(torch.__version__)
torch.manual_seed(42)
np.random.seed(seed=42)
random.seed(42)

#Load video file - RUN ONCE
vRead = iio.imread('data/c_elegans.mp4')
video = np.array(vRead)
video = torch.as_tensor(video)
testFrame = video[0]
### Crop letters from video
def writeVideoMOV(videoTensor, filename='decoded'):

    w = iio.get_writer(filename + '.mov', format='FFMPEG', mode='I', fps=20,
                        codec='h264_vaapi',
                        output_params=['-vaapi_device',
                                        '/dev/dri/renderD128',
                                        '-vf',
                                        'format=gray|nv12,hwupload'],
                        pixelformat='vaapi_vld')
    for frame in videoTensor:
        w.append_data(frame.numpy())
    w.close()
    print('video saved in local directory as: ' + filename + '.mov')
    return None

croppedVideo = torch.zeros((2484, 270, 344,3)).type(torch.float32)
for i in range(len(video)):
    croppedVideo[i] = video[i][18:288]
#Normalize range of RGBs
croppedVideo = torch.mul(croppedVideo, 1/255.0).type(torch.float32)
plt.imshow(croppedVideo[100])
#writeVideoMOV(croppedVideo, filename="cropped_C_Elegans")
#Set Video to cropped version
video = croppedVideo
print(video)
print("Cuda available: ", torch.cuda.is_available())
if(torch.cuda.is_available()):
    torch.cuda.set_device("cuda:1")

    print("Is cuDNN version:", torch.backends.cudnn.version())

    print("cuDNN enabled:a", torch.backends.cudnn.enabled)

    print("Device count: ", torch.cuda.device_count())

    print("Current device: ", torch.cuda.current_device())

    print("Device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
#Setup device agnostic code (i.e use GPU if possible)
device = "cuda" if torch.cuda.is_available() else "cpu"
gpuNum = 1
print(device)
video = video.to(device)
#Encoding
#INPUT  = L * F
#Output = 3
class hashNerf(nn.Module):
    def __init__(self,
                input_shape: int,
                hidden_units: int,
                output_shape: int,
                L=16, T=2**18, F=2, N_min=16, N_max=64, num_output=3):
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
        self.hash_table = nn.Parameter(torch.tensor(temp_hash_table.clone().detach().requires_grad_(True), requires_grad=True).to(device))
        self.vertices = torch.transpose(torch.tensor([
                        [0,0],
                        [0,1],
                        [1,0],
                        [1,1]]), 0,1).type(torch.float64) # dim : (2,4)
        self.prime_numbers = torch.tensor([1, 2654435761]).type(torch.int64)

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
        x_shape = X.shape
        """
        x_col = X[:,[0]] * self.N_values
        y_col = X[:, [1]] * self.N_values
        outputT = torch.zeros((len(x_col), L, 2)).type(torch.float32) # sca
        for i in range(len(x_col)):
            tempT = torch.zeros((L, 2))
            for j in range(L):
                tempT[j][0] = x_col[i][j]
                tempT[j][1] = y_col[i][j]
            outputT[i] = tempT
        """
        outputT = torch.zeros((len(X), 2, self.L)).type(torch.float32)
        for i in range(len(X)):
            outputT[i][0] = X[i][0] * self.N_values
            outputT[i][1] = X[i][1] * self.N_values
        x_scaled = outputT # dim : (batch_size, 2, L)
        x_floor = torch.floor(x_scaled) # dim : (batch_size, 2, L)
        x_ceil = torch.ceil(x_scaled) # dim : (batch_size, 2, L)

        x_vertices = torch.zeros((len(x_floor), 2, self.L, 4)) # dim : (batch_size, 2, L, 4)
        for i in range(len(x_floor)):
            # make sure to do x and y !
            for j in range(0,self.L):
                x_vertices[i][0][j] = outputT[i][0][j] * self.vertices[0]
                x_vertices[i][1][j] = outputT[i][1][j] * self.vertices[1]
        x_vertices = x_vertices.type(torch.int64)
        
        primesTemp = torch.tensor([1, 2654435761]).type(torch.int64)
        x_to_hash_temp = x_vertices * primesTemp[None, :, None, None]
        #print(x_to_hash_temp)
        #print(x_to_hash_temp.shape)

        x_hash_temp = torch.bitwise_xor(x_to_hash_temp[:, 0],
                                        x_to_hash_temp[:, 1]) # shape: num_points, L, 4
        x_hash_temp = torch.remainder(x_hash_temp, 2) 
        x_hash = x_hash_temp
        x_hash = x_hash.to(device)
        #Tricky code
        x_hash = x_hash.permute(1, 0, 2)  # shape: L, num_points, 4
        # lookup hash table:
        #print('x_hash shape:', x_hash.shape)
        #print('should be L, numPoints, 4')
        #print('hash_table shape:', self.hash_table.shape)
        def gather(a):
            return torch.gather(a[0], 0, a[1].unsqueeze(2).expand(-1, -1, 2))
        lookup = torch.stack([gather((self.hash_table, x_hash[i])) for i in range(16)], dim=0)
        lookup = lookup.permute(1,0,2,3)  # shape: num_points, L, 4, F
        lookup = lookup.to(device)
        #print(lookup.shape)
        
        #interpolation
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
        #print('f_stack: ', f_stack.shape)
        x_interp = torch.sum(f_stack[:, :, :, None] * lookup, dim=2)  # shape: num_points, L, F
        #print('x_interp shape:', x_interp.shape)
        x = x_interp.reshape(-1, self.L * self.F) # dim : num_points, L*F
        return self.layer_stack(x.to(device).type(torch.float32))

##Utils
import math
def PSNR(MSELoss, max):
    return (20*math.log10(max)) - (10*math.log10(MSELoss))
from pathlib import Path
def saveModel(modelPointer, psnr_note=35, frameNumber=0):
    # 1. Create models directory - won't create if it exists
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    # 2. Create model save path
    MODEL_NAME = "hashNerf_psnr_frame_"+frameNumber+"_psnr_"+str(psnr_note)+".pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    MODEL_SAVE_PATH
    #3. Save the model state dict
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=modelPointer.state_dict(),
            f=MODEL_SAVE_PATH)
    print(MODEL_SAVE_PATH)

#initialize model
model_0 = hashNerf(32, 64, 3)
lr1 = 0.0001
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr=lr1, eps=10e-15)

# Training
from tqdm import tqdm
train_loader = DataLoader(training_data, batch_size=2048, shuffle=False)
batchCount = 0
psnr_table = []
savedAt25 = False
savedAt30 = False
for epoch in tqdm(range(0,50)):
    #Training
    model_0.train()
    if savedAt30 == True:
        break
    for batch in iter(train_loader):
        batchCount += 1
        y_train = torch.as_tensor(batch[1]).to(device)
        y_train = torch.squeeze(y_train)
        #print(y_train.shape)
        X = torch.tensor(batch[0]).type(torch.float64).to(device)
        y_train = torch.as_tensor(y_train).type(torch.float64).to(device)
        # Forward Pass
        y_pred = model_0(X).to(device).type(torch.float64)
        loss = loss_fn(y_pred, y_train)
        psnr = PSNR(loss, 1.0)
        psnr_table.append(psnr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (savedAt25 == False) and (psnr >= 25):
        saveModel(model_0, 25, 0)
        savedAt25 = True
    elif (savedAt30 == False) and (psnr >= 30):
        saveModel(model_0, 30, 0)
        savedAt30 = True
        break
    if epoch % 1 == 0:
        print(f"Epoch: {epoch} | LR: {lr1} | Train loss: {loss} | PSNR: {psnr}")
    if epoch == 10:
        lr1 = lr1/10
    if epoch == 20:
        lr1 = lr1/2
print('Training Finished')


x_test_T = torch.as_tensor(croppedVideo[100])
reconstruction_input_matrix = torch.zeros(croppedVideo[100].shape[0], croppedVideo[100].shape[1], 2)
#encode coordinates into debug matrix
for i in range(0, x_test_T.shape[0]):
    for j in range(0, x_test_T.shape[1]):
        reconstruction_input_matrix[i][j] = torch.as_tensor([1.0*i,j*1.0]).type(torch.float64)
reconstruction_input_matrix = torch.flatten(reconstruction_input_matrix, 0, 1)
model_0.eval()
with torch.inference_mode():
    reconstruction = model_0(reconstruction_input_matrix).cpu()
    plt.imshow(reconstruction.reshape((croppedVideo[100].shape[0], croppedVideo[100].shape[1],3)))
    plt.axis(False)
    plt.title("Hash-Encoding")
    print(reconstruction)

