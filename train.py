import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
from torch.autograd import Variable
from models.PoseNet import PoseNet, PoseLoss
from data.DataSource import *
import os
import time

from optparse import OptionParser


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(epochs, batch_size, learning_rate, save_freq, data_dir):
    # train dataset and train loader
    datasource = DataSource(data_dir, train=True)
    train_loader = Data.DataLoader(
        dataset=datasource, batch_size=batch_size, shuffle=True)

    # load model
    posenet = PoseNet().to(device)

    # Load 200th epoch checkpoint and lower Beta for further training
    save_filename = 'epoch_{}.pth'.format(str(402).zfill(5))
    save_path = os.path.join('checkpoints3_50beta', save_filename)
    posenet.load_state_dict(torch.load(save_path))
    print("Checkpoint {} loaded!".format(save_filename))

    # loss function
    criterion = PoseLoss(0.3, 0.3, 2., 35, 35, 35)

    # train the network
    optimizer = torch.optim.Adam(nn.ParameterList(posenet.parameters()),
                                 lr=learning_rate, eps=1,
                                 weight_decay=0.0625,
                                 betas=(0.9, 0.999))

    batches_per_epoch = len(train_loader.batch_sampler)
    for epoch in range(epochs):
        print("Starting epoch {}:".format(epoch))
        start = time.time()
        posenet.train()
        for step, (images, poses) in enumerate(train_loader):
            optimizer.zero_grad()
            b_images = Variable(images, requires_grad=True).to(device)
            poses[0] = np.array(poses[0])
            poses[1] = np.array(poses[1])
            poses[2] = np.array(poses[2])
            poses[3] = np.array(poses[3])
            poses[4] = np.array(poses[4])
            poses[5] = np.array(poses[5])
            poses[6] = np.array(poses[6])
            poses = np.transpose(poses)
            b_poses = Variable(torch.Tensor(
                poses), requires_grad=True).to(device)

            p1_x, p1_q, p2_x, p2_q, p3_x, p3_q = posenet(b_images)
            if epoch % 10 == 0:
                print(f'Ground Truth position Prediction: {b_poses[0][:3]}')
                print(f'Ground Truth orientation Prediction: {b_poses[0][3:]}')
                print(f'Header 1 position Prediction: {p1_x[0]}')
                print(f'Header 1 orientation Prediction: {p1_q[0]}')
                print(f'Header 2 position Prediction: {p2_x[0]}')
                print(f'Header 2 orientation Prediction: {p2_q[0]}')
                print(f'Header 3 position Prediction: {p3_x[0]}')
                print(f'Header 3 orientation Prediction: {p3_q[0]}')

            loss = criterion(p1_x, p1_q, p2_x, p2_q, p3_x, p3_q, b_poses)
            loss.backward()
            optimizer.step()

            print("{}/{}: loss = {}".format(step+1, batches_per_epoch, loss))

        end = time.time()
        print(
            f"Epoch completed in: {end-start} seconds")

        # Save state
        if epoch % save_freq == 0:
            save_filename = 'epoch_{}.pth'.format(str(201+epoch+1).zfill(5))
            save_path = os.path.join('checkpoints4_50beta', save_filename)
            torch.save(posenet.state_dict(), save_path)
            print("Network saved!")

    input_names = ['301558925 - Anurag Parcha']
    output_names = ['Predicted Position Output',
                    'Predicted Orientation Output', 'Predicted Position Output2',
                    'Predicted Orientation Output2', 'Predicted Position Output3',
                    'Predicted Orientation Output3']
    torch.onnx.export(posenet, b_images, 'posenet.onnx', training=torch.onnx.TrainingMode.TRAINING,
                      input_names=input_names, output_names=output_names)


def get_args():
    parser = OptionParser()
    parser.add_option('--epochs', default=200, type='int')
    parser.add_option('--learning_rate', default=0.0001)
    parser.add_option('--batch_size', default=75, type='int')
    parser.add_option('--save_freq', default=10, type='int')
    parser.add_option('--data_dir', default='data/datasets/KingsCollege/')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    main(epochs=args.epochs, batch_size=args.batch_size,
         learning_rate=args.learning_rate, save_freq=args.save_freq, data_dir=args.data_dir)
