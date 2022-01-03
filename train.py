import os
import time

import torch
import argparse
import dataset
from GeoTr import GeoTr
from seg import U2NETP
import torch.nn.init as init
import torch.optim as optim
import torch
from torch.utils.data import DataLoader

def reload_segmodel(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:0')
        print(len(pretrained_dict.keys()))
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict}
        print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model

def reload_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:0')
        print(len(pretrained_dict.keys()))
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model

def init_weight(model):
    for name, param in model.named_parameters():
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue
    return model

def visualize_data(images, mappings, mask):
    print(images.shape)
    assert False, "stop"

def train(args):

    # segmentation_model = None
    segmentation_model = U2NETP(3, 1)
    reload_segmodel(segmentation_model, args.segmentation_model)
    segmentation_model.to(args.device)

    # GeoTr = None
    GeoTr_model = GeoTr(num_attn_layers=6)
    GeoTr_model = init_weight(GeoTr_model)
    if args.GeoTr_model != "":
         reload_model(GeoTr_model, args.GeoTr_model)
    GeoTr_model = torch.nn.DataParallel(GeoTr_model).to(args.device)
    GeoTr_model.train()

    train_dataset = dataset.GeometricDataset(args.train_data, segmentation_model)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(GeoTr_model.parameters(), lr=0.0001)

    for epoch in range(50):

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            start = time.time()
            inputs, mappings = data
            inputs = inputs.to(args.device)
            mappings = mappings.to(args.device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            mask, _1,_2,_3,_4,_5,_6 = segmentation_model(inputs.float())
            mask = (mask > 0.5).float()
            # mask_copy = mask.to("cpu")

            # visualize_data()

            inputs = mask * inputs.float()
            outputs = GeoTr_model(inputs)
            outputs = (2 * (outputs / 286.8) - 1) * 0.99
            print()
            loss = criterion(outputs, mappings)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print("timeeeeeee", time.time()-start)
            if i % 1 == 0:  # print every 1 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss))
                running_loss = 0.0
            # torch.save(GeoTr_model.state_dict(), f"{args.saved_path}/{epoch}_{i}.pth")
        torch.save(GeoTr_model.state_dict(), f"{args.saved_path}/{epoch}.pth", _use_new_zipfile_serialization=False)
    print('Finished Training')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data', default='./data/training/', help='path to training dataset')
    # parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--segmentation_model', default='./model_pretrained/epoch95_iter3644.pth')
    parser.add_argument('--GeoTr_model', default="", help='path to GeoTr_model model')
    parser.add_argument('--saved_path', default='./images/result')
    parser.add_argument('--batch_size', default=1, help='batch_size')
    parser.add_argument('--device', default="cuda", help='cuda or cpu')

    args = parser.parse_args()

    train(args)

# if __name__ == '__main__':
#     print(os.listdir("/media/minh/HDD1/hblab/doctr/20211220_data/test"))