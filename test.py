import os

import torch
import cv2
import numpy as np

from train import set_fold
from network.vos import LSTMVOSWithSCTestTime
from network.track import TrackerKP
from dataloaders import VOSDataloader
from render import PRRenderer
from utils import tensor_to_cv2, cv2_to_tensor, matrix_from_7_parameters


def test_vos(base_dir='../2d3d_dataset', weights='weights/vos_fold0_best.pth'):
    vos = LSTMVOSWithSCTestTime().cuda()
    vos.load_state_dict(torch.load(weights))
    vos.eval()
    test_cases = set_fold(0)[1]
    dataloader = VOSDataloader(
        image_dirs=[os.path.join(base_dir, case_id) for case_id in test_cases],
        label_dirs=[os.path.join(base_dir, case_id, 'label') for case_id in test_cases],
    )
    images, labels = dataloader.get_test_data()
    for case in range(len(images)):
        case_image = torch.stack(images[case], dim=0)
        case_label = torch.stack(labels[case], dim=0)
        vos.test_init(case_image[0: 1].cuda(), case_label[0: 1].cuda())
        for i in range(1, len(case_image)):
            pred = vos.test(case_image[i].unsqueeze(0).cuda()).detach().cpu()
            gt = case_label[i].unsqueeze(0)
            test_loss = torch.nn.MSELoss()(pred, gt)
            print('Case [{}/{}], Image [{}/{}], Loss: {}'.format(
                case + 1, len(images), i, len(case_image) - 1, float(test_loss)
            ))
            cv2.imwrite('vos_test/{}_{}.png'.format(case+1, i), tensor_to_cv2(pred.squeeze()))
            print('Write OK.')


def test_tracker(base_dir='../2d3d_dataset'):
    os.makedirs('tracker_test', exist_ok=True)
    train_cases, test_cases = set_fold(1)
    dataloader = VOSDataloader(
        image_dirs=[os.path.join(base_dir, case_id) for case_id in test_cases],
        label_dirs=[os.path.join(base_dir, case_id, 'label') for case_id in test_cases],
    )
    images, labels = dataloader.get_test_data()
    for case in range(len(images)):
        case_image = np.asarray([tensor_to_cv2(image) for image in images[case]])
        case_label = np.asarray([tensor_to_cv2(label) for label in labels[case]])
        tracker = TrackerKP(case_image[0], case_label[0])
        for i in range(1, len(case_image)):
            pred = tracker(case_image[i])
            gt = case_label[i]
            test_loss = torch.nn.MSELoss()(cv2_to_tensor(pred), cv2_to_tensor(gt))
            print('Case [{}/{}], Image [{}/{}], Loss: {}'.format(
                case + 1, len(images), i, len(case_image) - 1, float(test_loss)
            ))
            cv2.imwrite('tracker_test/{}_{}.png'.format(case+1, i), pred)
            print('Write OK.')


def test_colmap(path):
    # parse colmap generated images.txt
    txt_file = open(path)
    parameters = []
    for line in txt_file.readlines():
        if line.endswith('.png\n'):
            numbers = line.split(' ')[1:-2]
            numbers = [float(num) for num in numbers]
            parameters.append(numbers)
    txt_file.close()
    # render each frame
    renderer = PRRenderer('../2d3d_dataset/GongDaoming/kidney_and_tumor.obj')
    curr_mat = np.identity(4)
    for i, param in enumerate(parameters):
        this_mat = matrix_from_7_parameters(param)
        curr_mat = np.matmul(this_mat, curr_mat)
        image = renderer.render(mat=curr_mat)
        cv2.imwrite('colmap_gen/{}.png'.format(i), image)
    return


if __name__ == '__main__':
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    # test_vos()
    test_tracker()
