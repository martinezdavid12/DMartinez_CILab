import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import math
from tqdm import tqdm
from pathlib import Path
#from pthflops import count_ops

vRead = iio.imread('data/c_elegans.mp4')
video = np.array(vRead)
testFrame = video[0]

print("Cuda available: ", torch.cuda.is_available())
if(torch.cuda.is_available()):
    torch.cuda.set_device("cuda:1")

    print("Is cuDNN version:", torch.backends.cudnn.version())

    print("cuDNN enabled: ", torch.backends.cudnn.enabled)

    print("Device count: ", torch.cuda.device_count())

    print("Current device: ", torch.cuda.current_device())

    print("Device name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
device = "cuda" if torch.cuda.is_available() else "cpu"
gpuNum = 1
print(device)

#Video Metadata
import imageio.v3 as iio
props = iio.improps("data/c_elegans.mp4")
print("Shape (frames, w, h, RGB): \n" + str(props.shape))
numFrames = props.shape[0]
print(props.dtype)

gen = torch.Generator()
gen.manual_seed(42)
gauss_mat = torch.normal(0.0, 1, size=(256, 2), generator=gen)

def PSNR(MSELoss, max):
    return (20*math.log10(max)) - (10*math.log10(MSELoss))

def input_mapping(x, B):
    #print("shape of x:", x.shape)
    #print("shape of B:", B.shape)
    if B is None:
        return x
    else:
        B_T = np.array(B).T
        #print("B_T", B_T.shape)
        x_proj = torch.as_tensor(np.dot(2.0*(np.pi)* x, B_T))
        #print("x_proj:", x_proj, x_proj.shape)
        return np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1)   
#basic_mapping(np.array([1,1]), np.eye(2))
seed = 42
mapping_size = 256
mappings = {}
mappings['raw'] = None
mappings['identity'] = np.eye(2)
# ex: [1, 10, 100]
mappings['gaussian_fourier'] = lambda scale: scale * gauss_mat
print(mappings['gaussian_fourier'](25).shape)

#define Model
class MLP(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_shape, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), 
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), 
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), 
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.Sigmoid(),
            nn.Linear(hidden_units, output_shape),
        )
    def forward(self, X):
        return self.layer_stack(X.to(device).type(torch.float32))

#DataLoader
class SingleImageDataset(Dataset):
    def __init__(self, data, mapping, transform=None, target_transform=None):
        self.train_data = data[0]
        self.img = data[1]
        self.B = mapping
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return img.shape[0] * img.shape[1]
    def __getitem__(self, idx):
        row = idx // int(img.shape[1])
        col = idx % int(img.shape[1])
        coords = torch.as_tensor(self.train_data[row][col]).type(torch.float64)
        pixel = torch.as_tensor(self.img[row][col]).type(torch.float64)
        return torch.as_tensor(input_mapping(coords, self.B)).type(torch.float64).cpu(), pixel.cpu()
#Frame Loop
#set to existing model dict if continuing training with previous informed weights
inputModel = MLP(input_shape=512, 
              hidden_units=256,
              output_shape=3).to(device)

inputModel.load_state_dict(torch.load(f='models/pos_encoding_gauss_25_c_elegans_frame_99.pth'))

for t in range(100,151):
    testFrame = video[t]
    #Square the image
    l = min(2*(props.shape[1]//2), 2*(props.shape[2]//2))
    #Encode Tensor
    inputFrame = iio.imread('testFrame.png')
    frame = torch.as_tensor(testFrame)[:l, :l]

    preImg = torch.as_tensor(frame).to(device)
    img = torch.mul(preImg, 1/255.0).type(torch.float64)
    print(img)
    #plt.imshow(img.cpu())
    #plt.axis('off')
    print('img is now on', img.device)
    print(img.shape)
    #generate coordinates in unit square
    coords = np.linspace(0, 1, l, endpoint=False)
    x_test = np.stack(np.meshgrid(coords, coords), -1)
    test_data = [x_test, img]
    print('shape of test data: ', x_test.shape)
    training_data = SingleImageDataset(test_data, mappings['gaussian_fourier'](25))
    #training loop
    train_loader = DataLoader(training_data, batch_size=512, shuffle=True, pin_memory=True)
    batchCount = 0
    psnr_table = []
    #create model instance
    model_0 = inputModel
    lr1 = 0.0001
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=lr1)
    for epoch in tqdm(range(0,30)):
        #Training
        model_0.train()
        for batch in iter(train_loader):
            #print(batch[0].shape)
            batchCount += 1
            encodings = torch.as_tensor(batch[0]).to(device)
            y_train = torch.as_tensor(batch[1]).to(device)
            y_train = torch.squeeze(y_train)
            X = torch.as_tensor(encodings).type(torch.float64).to(device)
            y_train = torch.as_tensor(y_train).type(torch.float64).to(device)
            # Forward Pass
            y_pred = model_0(X).to(device).type(torch.float64)
            loss = loss_fn(y_pred, y_train)
            psnr = PSNR(loss, 1.0)
            psnr_table.append(psnr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 1 == 0:
            print(f"Epoch: {epoch} | LR: {lr1} | Train loss: {loss} | PSNR: {psnr}")
        if epoch == 10:
            lr1 = lr1/10
        if epoch == 20:
            lr1 = lr1/2
        if psnr > 35.50:
            print("PSNR of 35.50+ already satisfied")
            break
    print("Training Finished for Frame:", str(t))
    inputModel = model_0
    #Save Model
    # 1. Create models directory
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    # 2. Create model save path
    MODEL_NAME = "pos_encoding_gauss_25_c_elegans_frame_"+str(t)+".pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    MODEL_SAVE_PATH
    #3. Save the model state dict
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model_0.state_dict(),
            f=MODEL_SAVE_PATH)
    print(MODEL_SAVE_PATH)

    # Layer weight extraction
    model_0.eval()
    with torch.inference_mode():
        print(len(model_0.layer_stack))
        for i in range(0,len(model_0.layer_stack)):
            if isinstance(model_0.layer_stack[i], torch.nn.Linear):
                print('layer:', i)
                weightMat = torch.as_tensor(model_0.layer_stack[i].weight)
                weightMatPath = 'weights/pos_encoding_25_elegans_f'+str(t)
                torch.save(weightMat, weightMatPath)

#Plot PSNR
plt.plot(range(0,batchCount), psnr_table)
plt.title('Train error')
plt.ylabel('PSNR')
#NOT EPOCHS - THESE ARE BATCHES!!!
plt.xlabel('Batch Iter')
plt.legend()

plt.show()


