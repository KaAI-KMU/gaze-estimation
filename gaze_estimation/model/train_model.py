import os
from argparse import ArgumentParser
from functools import partial

import h5py
import numpy as np
import torch
import pytorch_lightning as pl
from PIL import ImageFilter
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataloader import EstimationFileDataset, EstimationHdf5DatasetMyDataset
from model import GazeEstiamationModel_resent18, GazeEstimationModel_vgg16, GazeEstimationPreactResnet
from GazeAngleAccuracy import GazeAngleAccuracy

class TrainEstimationModel(pl.LightningModule):
    def __init__(self, hparams, train_subjects, validate_subjects, test_subjects):
        super(TrainEstimationModel,self).__init__()
        _loss_fn = {
            "mse" : torch.nn.MSELoss
        }
        _param_num = {
            "mse" : 2
        }
        _models = {
            "vgg16" : partial(GazeEstimationModel_vgg16, num_out = _param_num.get(hparams['loss_fn'])),
            "resnet18" : partial(GazeEstiamationModel_resent18, num_out = _param_num.get(hparams['loss_fn'])),
            "preactresnet" : partial(GazeEstimationPreactResnet, num_out = _param_num.get(hparams['loss_fn']))
        }
        self._model = _models.get(hparams['model_base'])()
        self._criterion = _loss_fn.get(hparams['loss_fn'])()
        self._angle_acc = GazeAngleAccuracy()
        self._train_subjects = train_subjects
        self._validate_subjects = validate_subjects
        self._test_subjects = test_subjects
        self._hparams = hparams
        self.val_outputs = {'val_loss' : [], 'angle_acc' : []}
        self.test_outputs = {'angle_acc' : []}

    def forward(self, left_patch, right_patch, head_pose):
        return self._model(left_patch, right_patch, head_pose)
    
    def training_step(self, batch, batch_idx):
        _left_patch, _right_patch, _headpose_label, _gaze_labels, _landmark_labels = batch

        angular_out = self.forward(_left_patch, _right_patch, _headpose_label)
        loss = self._criterion(angular_out, _gaze_labels)
        self.log("train_loss", loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        _left_patch, _right_patch, _headpose_label, _gaze_labels, _landmark_labels = batch
        angular_out = self.forward(_left_patch, _right_patch, _headpose_label)
        loss = self._criterion(angular_out, _gaze_labels)
        angle_acc = self._angle_acc(angular_out[:, :2], _gaze_labels)
        self.val_outputs['val_loss'].append(loss)
        self.val_outputs['angle_acc'].append(angle_acc)
        return self.val_outputs

    def on_validation_epoch_end(self):
        _losses = torch.stack([x for x in self.val_outputs['val_loss']])
        _angles = np.array([x for x in self.val_outputs['angle_acc']])
        # _losses = torch.stack([x['val_loss'] for x in self.outputs])
        # _angles = np.array([x['angle_acc'] for x in self.outputs])
        self.log('val_loss', _losses.mean(), prog_bar=True)
        self.log('val_acc', _angles.mean(), prog_bar=True)
        self.val_outputs.clear()
        self.val_outputs = {'val_loss' : [], 'angle_acc' : []}

    def test_step(self, batch, batch_idx):
        _left_patch, _right_patch, _headpose_label, _gaze_labels, _landmark_labels = batch

        angular_out = self.forward(_left_patch, _right_patch, _headpose_label)
        angle_acc = self._angle_acc(angular_out[:, :2], _gaze_labels)

        self.test_outputs['angle_acc'].append(angle_acc)
        return self.test_outputs

    def on_test_epoch_end(self):
        _angles = np.array([x for x in self.test_outputs['angle_acc']])

        self.log("test_angle_mean", _angles.mean())
        self.log("test_angle_std", _angles.std())
        self.test_outputs.clear()
        self.test_outputs = {'angle_acc': []}

    def configure_optimizers(self):
        _params_to_update = []
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                _params_to_update.append(param)

        _learning_rate = self._hparams['learning_rate']
        _optimizer = torch.optim.Adam(_params_to_update, lr=_learning_rate)
        _scheduler = torch.optim.lr_scheduler.StepLR(_optimizer, step_size=30, gamma=0.1)

        return [_optimizer], [_scheduler]
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--augment', action="store_true", dest="augment")
        parser.add_argument('--loss_fn', choices=["mse"], default="mse")
        parser.add_argument('--batch_size', default=128, type=int)
        parser.add_argument('--batch_norm', default=True, type=bool)
        parser.add_argument('--learning_rate', type=float, default=0.0003)
        parser.add_argument('--model_base', choices=["vgg16", "resnet18", "preactresnet"], default="resnet18")
        return parser

    def train_dataloader(self):
        _train_transforms = None
        if self._hparams['augment']:
            _train_transforms = transforms.Compose([transforms.RandomResizedCrop(size=(36, 60), scale=(0.5, 1.3)),
                                                    transforms.RandomGrayscale(p=0.1),
                                                    transforms.ColorJitter(brightness=0.5, hue=0.2, contrast=0.5,
                                                                           saturation=0.5),
                                                    lambda x: x if np.random.random_sample() <= 0.1 else x.filter(
                                                        ImageFilter.GaussianBlur(radius=3)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])])
        _data_train = EstimationHdf5DatasetMyDataset(h5_file=h5py.File(self._hparams['hdf5_file'], mode="r"),
                                      subject_list=self._train_subjects,
                                      transform=_train_transforms)
        return DataLoader(_data_train, batch_size=self._hparams['batch_size'], shuffle=True,
                          num_workers=self._hparams['num_io_workers'], pin_memory=False)

    def val_dataloader(self):
        _data_validate = EstimationHdf5DatasetMyDataset(h5_file=h5py.File(self._hparams['hdf5_file'], mode="r"),
                                         subject_list=self._validate_subjects)
        return DataLoader(_data_validate, batch_size=self._hparams['batch_size'], shuffle=False,
                          num_workers=self._hparams['num_io_workers'], pin_memory=False)
    
    def test_dataloader(self):
        _data_test = EstimationHdf5DatasetMyDataset(h5_file=h5py.File(self._hparams['hdf5_file'], mode="r"),
                                     subject_list=self._test_subjects)
        return DataLoader(_data_test, batch_size=self._hparams['batch_size'], shuffle=False,
                          num_workers=self._hparams['num_io_workers'], pin_memory=False)

if __name__ == "__main__":
    from pytorch_lightning import Trainer

    root_dir = os.path.dirname(os.path.realpath(__file__))#h5파일이 저장된 디렉토리로 설정

    _root_parser = ArgumentParser(add_help=False)
    _root_parser.add_argument('--accelerator', choices=['cpu','gpu'], default='gpu',
                              help='gpu to use, can be repeated for mutiple gpus i.e. --gpu 1 --gpu 2')
    _root_parser.add_argument('--hdf5_file', type=str,
                              default=os.path.abspath(os.path.join(root_dir, "../KaAI_dataset_3case.hdf5")))#데이터셋이 담긴 h5 파일 지명
    _root_parser.add_argument('--dataset', type=str, choices=["KaAI", "other"], default="KaAI")
    _root_parser.add_argument('--save_dir', type=str, default='Gaze Estimate Model/custum/checkpoints')#체크포인트 저장할 디렉토리 설정
    _root_parser.add_argument('--benchmark', action='store_true', dest="benchmark")
    _root_parser.add_argument('--no_benchmark', action='store_false', dest="benchmark")
    _root_parser.add_argument('--num_io_workers', default=0, type=int)
    _root_parser.add_argument('--k_fold_validation', default=False, type=bool)
    _root_parser.add_argument('--accumulate_grad_batches', default=1, type=int)
    _root_parser.add_argument('--seed', type=int, default=0)
    _root_parser.add_argument('--min_epochs', type=int, default=5, help="Number of Epochs to perform at a minimum")
    _root_parser.add_argument('--max_epochs', type=int, default=20,
                              help="Maximum number of epochs to perform; the trainer will Exit after.")
    _root_parser.add_argument('--checkpoint',type=list, default=[])
    _root_parser.set_defaults(benchmark=False)
    _root_parser.set_defaults(augment=True)

    _model_parser = TrainEstimationModel.add_model_specific_args(_root_parser)
    _hyperparams = _model_parser.parse_args()
    _hyperparams = vars(_hyperparams)
    pl.seed_everything(_hyperparams['seed'])

    _train_subjects = []
    _valid_subjects = []
    _test_subjects = []
    if _hyperparams['dataset'] == "KaAI":
        if _hyperparams['k_fold_validation']:
            _train_subjects.append([1, 2, 8, 10, 3, 4, 7, 9])
            _train_subjects.append([1, 2, 8, 10, 5, 6, 11, 12, 13])
            _train_subjects.append([3, 4, 7, 9, 5, 6, 11, 12, 13])
            # validation set is always subjects 14, 15 and 16
            _valid_subjects.append([0, 14, 15, 16])
            _valid_subjects.append([0, 14, 15, 16])
            _valid_subjects.append([0, 14, 15, 16])
            # test subjects
            _test_subjects.append([5, 6, 11, 12, 13])
            _test_subjects.append([3, 4, 7, 9])
            _test_subjects.append([1, 2, 8, 10])
        else:
            _train_subjects.append([0,1,2])#, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
            _train_subjects.append([0,1])
            _valid_subjects.append([1])  # Note that this is a hack and should not be used to get results for papers
            _test_subjects.append([0])
    else:
        file = h5py.File(_hyperparams['hdf5_file'], mode="r")
        keys = [int(subject[1:]) for subject in list(file.keys())]
        file.close()
        if _hyperparams['k_fold_validation']:
            all_subjects = range(len(keys))
            for leave_one_out_idx in all_subjects:
                _train_subjects.append(all_subjects[:leave_one_out_idx] + all_subjects[leave_one_out_idx + 1:])
                _valid_subjects.append([leave_one_out_idx])  # Note that this is a hack and should not be used to get results for papers
                _test_subjects.append([leave_one_out_idx])
        else:
            _train_subjects.append(keys[1:])
            _valid_subjects.append([keys[0]])
            _test_subjects.append([keys[0]])
 
    for fold, (train_s, valid_s, test_s) in enumerate(zip(_train_subjects, _valid_subjects, _test_subjects)):
        complete_path = os.path.abspath(os.path.join(_hyperparams['save_dir'], f"fold_{fold}/"))

        _model = TrainEstimationModel(hparams=_hyperparams,
                             train_subjects=train_s,
                             validate_subjects=valid_s,
                             test_subjects=test_s)
        # from torchsummary import summary
        # summary(_model, input_size=(3,256,256))
        # save all models
        checkpoint_callback = ModelCheckpoint(dirpath=complete_path, filename= "{epoch}-{val_loss:.3f}",
                                              monitor='val_loss', mode='min', verbose=True, save_top_k=-1 if not _hyperparams['augment'] else 5)
        from lightning.pytorch.loggers import TensorBoardLogger

        logger = TensorBoardLogger(save_dir='Gaze Estimate Model/custum/logs/',name='gaze_direction_estimation_logs')

        # start training _hyperparams.accelerator
        trainer = Trainer(accelerator=_hyperparams['accelerator'],
                          precision=32,
                          callbacks=[checkpoint_callback],
                          min_epochs=_hyperparams['min_epochs'],
                          max_epochs=_hyperparams['max_epochs'],
                          accumulate_grad_batches=_hyperparams['accumulate_grad_batches'],
                          benchmark=_hyperparams['benchmark'],
                          logger=logger)
        trainer.fit(_model)
        trainer.test()