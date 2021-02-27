from model.CenterNet import CenterNet
from utils.util import draw, get_center_from_center_mask
from torch.utils.data import DataLoader
from data.base import CustomDataset
import matplotlib.pyplot as plt
import torch
import cv2
import time
import numpy as np
import urllib
# url = 'https://cdn.mos.cms.futurecdn.net/hbKifQWBTcdhTEw8zsJWnF-1200-80.jpg'

# resp = urllib.request.urlopen(url)
# image = np.asarray(bytearray(resp.read()), dtype="uint8")
# image = cv2.imdecode(image, cv2.IMREAD_COLOR)
# image = cv2.resize(image,(416,416))/255
# img = torch.Tensor(image).permute(2,0,1).unsqueeze(0)
model = CenterNet(num_class=2).to('cuda')
model.load_state_dict(torch.load('./model_final.pth'))
model.eval()
ds = CustomDataset('../valid/images','../valid/labels',num_class=2)
loader = DataLoader(ds,batch_size=1,shuffle=True)
img, center_mask, _, _ = next(iter(loader))
imga = img.to('cuda')
predictions = model(imga)
t = time.time()
predictions = model(imga)#.to('cuda'))
print(time.time()-t)
predictions = predictions.cpu()
center_predict, offset_predict, size_predict = torch.split(
            predictions, [2, 2, 2], 1)
i = 0
img = (255*img[i:i+1,:,:,:].permute(0, 2, 3, 1)).long().squeeze(0)#.cpu()
center_predict = center_predict[i:i+1,:,:,:].permute(0, 2, 3, 1).detach().numpy()
#center_mask = center_mask[i:i+1,:,:,:].permute(0, 2, 3, 1).detach().numpy()
#print(center_mask.shape)
size_predict = size_predict[i:i+1,:,:,:].permute(0, 2, 3, 1).detach().numpy()
#size_predict[0,:,:,0] *= 1024
#size_predict[0,:,:,1] *= 800
#centers = get_center_from_center_mask(center_predict)
offset_predict = offset_predict[i:i+1,:,:,:].permute(0, 2, 3, 1).detach().numpy()
centers = get_center_from_center_mask(center_predict)
print(centers)
try:
  imag = cv2.UMat.get(draw(img[:, :, :], centers,
                    offset_predict[0, :, :, :], size_predict[0, :, :, :]))
except:
  imag = draw(img[:, :, :], centers,
                    offset_predict[0, :, :, :], size_predict[0, :, :, :])
fig, ax = plt.subplots(2, 3)
ax[0, 0].imshow(center_predict[0,:, :, 0], cmap='gray')
ax[0, 1].imshow(center_predict[0,:, :, 1], cmap='gray')
#ax[0, 2].imshow(center_predict[0,:, :, 2], cmap='gray')
# ax[1, 0].imshow(center_predict[0,:, :, 3], cmap='gray')
# ax[1, 1].imshow(center_predict[0,:, :, 4], cmap='gray')
# ax[1, 2].imshow(center_predict[0,:, :, 5], cmap='gray')
ax[1, 0].imshow(imag[:, :, ::-1])  # [0, :, :, :])
#ax[2, 1].imshow(img[0, :, :, :])
ax[1, 1].imshow(size_predict[0,:, :, 0], cmap='gray')
ax[1, 2].imshow(size_predict[0,:, :, 1], cmap='gray')
plt.savefig('log.jpg')
#cv2.imwrite('res.jpg',imag)
