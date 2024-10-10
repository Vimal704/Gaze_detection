import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo
import torchvision
from torchvision import transforms
import torch.backends.cudnn as cudnn

from model import L2CS
from dataset import Mpiigaze

def getArch_weights(bins):
    model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)
    pre_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

    return model, pre_url

def get_ignored_params(model):
    b = [model.conv1, model.bn1]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model):
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    b = [model.fc_yaw_gaze, model.fc_pitch_gaze]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param

def load_filtered_state_dict(model, snapshot):
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

if __name__ == '__main__':
    cudnn.enable = True
    num_epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64 if torch.cuda.is_available() else 16
    alpha = 1
    output = '/kaggle/working/'
    data_set = 'mpiigaze'
    gazeMpiimage_dir = '/kaggle/input/mpiifacegaze/MPIIFaceGaze_preprocessed/Image'
    gazeMpiilabel_dir = '/kaggle/input/mpiifacegaze/MPIIFaceGaze_preprocessed/Label'
    lr = 1e-4
    gpu_id = 0

    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
            )
        ])

    folder = os.listdir(gazeMpiilabel_dir)
    folder.sort()
    testlabelpathcombined = [os.path.join(gazeMpiilabel_dir, j) for j in folder]
    for fold in range(15):
        model, pre_url = getArch_weights(90)
        load_filtered_state_dict(model, model_zoo.load_url(pre_url))
        model = nn.DataParallel(model)
        model.to(device)
        print('Loading data...')
        #self, pathorg, root, transform, train, angle,fold=0)
        dataset = Mpiigaze(testlabelpathcombined, gazeMpiimage_dir, transformations, True, 180,fold)
        train_loader_gaze = DataLoader(
                    dataset=dataset,
                    batch_size = int(batch_size),
                    shuffle = True,
                    # num_workers=1,
                    pin_memory=True
                )
        torch.backends.cudnn.benchmark = True

        summary_name = f'{"L2CS-mpiigaze"}_{int(time.time())}'

        if not os.path.exists(os.path.join(output+'/{}'.format(summary_name),'fold' + str(fold))):
            os.makedirs(os.path.join(output+'/{}'.format(summary_name),'fold' + str(fold)))
        # if not os.path.exists(os.path.join(output+'/'+f'{summary_name}'), 'fold' + str(fold)):
            # os.makedirs(os.path.join(output+'/'+ f'{summary_name}', 'fold'+ str(fold)))

        criterion = nn.CrossEntropyLoss().to(device)
        reg_criterion = nn.MSELoss().to(device)
        softmax = nn.Softmax(dim=1).to(device)
        idx_tensor = [idx for idx in range(90)]
        idx_tensor = torch.tensor(idx_tensor, dtype=torch.float)

        # optimizer gaze
        optimizer_gaze = torch.optim.Adam([
            {'params': get_ignored_params(model.module), 'lr':0},
            {'params': get_non_ignored_params(model.module), 'lr':lr},
            {'params': get_fc_params(model.module), 'lr':lr}
            ], lr)

        print('Training Started...')

        for epoch in range(num_epochs):
            sum_loss_pitch_gaze = sum_loss_yaw_gaze = iter_gaze = 0

            for i, (images_gaze, labels_gaze, cont_labels_gaze, name) in enumerate(train_loader_gaze):
                images_gaze = images_gaze.to(device)

                #Binned labels
                label_pitch_gaze = labels_gaze[:, 0].to(device)
                label_yaw_gaze = labels_gaze[:, 1].to(device)

                # Continuous labels
                label_pitch_cont_gaze = cont_labels_gaze[:, 0].to(device)
                label_yaw_cont_gaze = labels_gaze[:, 1].to(device)

                pitch, yaw = model(images_gaze)

                # Cross entropy loss
                loss_pitch_gaze = criterion(pitch, label_pitch_gaze)
                loss_yaw_gaze = criterion(yaw, label_yaw_gaze)

                # MSE Loss
                pitch_predicted = softmax(pitch)
                yaw_predicted = softmax(yaw)

                pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 -42
                yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 42
                
                # print(yaw_predicted.dtype)
                # print(label_yaw_cont_gaze.dtype)
                loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont_gaze)
                loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont_gaze.float())

                # Total loss
                loss_pitch_gaze += alpha * loss_reg_pitch
                loss_yaw_gaze += alpha * loss_reg_yaw

                sum_loss_pitch_gaze += loss_pitch_gaze
                sum_loss_yaw_gaze += loss_yaw_gaze

                loss_seq = [loss_pitch_gaze, loss_yaw_gaze]
                grad_seq = [torch.tensor(1.0).to(device) for _ in range(len(loss_seq))]

                optimizer_gaze.zero_grad(set_to_none=True)
                torch.autograd.backward(loss_seq, grad_seq)
                optimizer_gaze.step()

                iter_gaze += 1
                # print(f'--------{i}-----------')
                if (i+1) % 100 == 0:
                    print('Epoch [%d/%d], Iter [%d/%d] Losses: '
                        'Gaze Yaw %.4f, Gaze Pitch %.4f' % (
                            epoch+1,
                            num_epochs,
                            i+1,
                            len(dataset) // batch_size,
                            sum_loss_pitch_gaze/iter_gaze,
                            sum_loss_yaw_gaze/iter_gaze
                        )
                        )

            # Save models at numbered epochs
            if epoch % 1 == 0 and epoch < num_epochs:
                print('Taking snapshot...',
                        torch.save(model.state_dict(),
                        output+'/fold' + str(fold) + '/' +
                        '_epoch_' + str(epoch+1) + '.pkl')
                    )
                
 