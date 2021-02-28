from torch import optim
from torch.utils.data import DataLoader
from data.base import CustomDataset
from model.CenterNet import CenterNet
import torch
from losses import Loss
num_class = 2
cnet = CenterNet(num_class=num_class).to('cuda')
# cnet.load_state_dict(torch.load('model_final.pth'))
optimizer = optim.SGD(cnet.parameters(),lr=0.001)
criterion = Loss().to('cuda')
ds = CustomDataset('../train/images',
                   '../train/labels', num_class=num_class)
sample_loader = DataLoader(ds, batch_size=32, shuffle=True)
EPOCH = 100
for e in range(EPOCH):
    running_loss,wh_loss,hm_loss,off_loss = 0,0,0,0
    for idx, (img, center_mask, offset_mask, size_mask) in enumerate(sample_loader):
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
        #print(total_loss)
        running_loss += total_loss
        wh_loss += size_loss
        off_loss += offset_loss
        hm_loss += focal_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if idx % 25 == 24:
            print(f"Epoch: {e+1}, iter {idx+1}: running loss: {running_loss/(25):.2f}, size_loss: {wh_loss/25:.2f}" +
                  f', offset_loss: {off_loss/25:.2f}, center_loss: {hm_loss/25:.2f}')
            running_loss, hm_loss, wh_loss, off_loss = 0,0,0,0
torch.save(cnet.state_dict(), 'model_final.pth')
