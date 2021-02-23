from torch import optim
from torch.utils.data import DataLoader
from data.base import CustomDataset
from model.CenterNet import CenterNet
import torch
from losses import Loss
num_class = 6
cnet = CenterNet(num_class=num_class).to('cuda')
# cnet.load_state_dict(torch.load('model_final.pth'))
optimizer = optim.Adam(cnet.parameters())
criterion = Loss().to('cuda')
ds = CustomDataset('../datasets/train/images',
                   '../datasets/train/labels', num_class=6)
sample_loader = DataLoader(ds, batch_size=4, shuffle=True)
EPOCH = 100
for e in range(EPOCH):
    running_loss = 0
    for idx, (img, center_mask, offset_mask, size_mask, centers) in enumerate(sample_loader):
        predictions = cnet(img.to('cuda'))
        center_predict, offset_predict, size_predict = torch.split(
            predictions, [num_class, 2, 2], 1)
        center_mask, offset_mask, size_mask = \
            center_mask.to('cuda'), offset_mask.to(
                'cuda'), size_mask.to('cuda')
        # print(centers[0])
        #assert False
        prediction = [center_predict, offset_predict, size_predict]
        target = [center_mask, offset_mask, size_mask]
        total_loss, size_loss, offset_loss, focal_loss = criterion(
            prediction, target)
        #total_loss /= 100000
        running_loss += total_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if idx % 50 == 49:
            print(f"Epoch: {e+1}, iter {idx+1}: running loss: {running_loss/(idx+1)}, size_loss: {size_loss}" +
                  f', offset_loss: {offset_loss}, center_loss: {focal_loss}')
torch.save(cnet.state_dict(), 'model_final.pth')
