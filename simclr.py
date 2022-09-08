import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        # SummaryWriter: Tensorboard 시각화를 위한 로그 데이터로 기본 directory는 ./runs
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    """Loss Function 정의"""
    # features: FC layer의 Output vector 
    # CrossEntropyLoss를 계산하기 위해 필요한 logits과 labels 반환
    def info_nce_loss (self, features):

        # Similarity_matrix.shape == labels.shape == (n_views*batch_size, n_views*batch_size)
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)
        features = F.normalize(features, dim=1) ## Normalized embedding
        similarity_matrix = torch.matmul(features, features.T)

        # labels와 similarity matrix에서 대각 값 제거
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # Positive pair 간 Similarity 추출
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # Negative pair 간 Similarity 추출
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature ## Temperature parameter
        return logits, labels

    def train(self, train_loader):

        # GradScaler: Gradient Scaling. 32 -> 16 float
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # Config file 저장
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    # self.model을 통해 Projection head 이후 latent vector z 추출
                    features = self.model(images)
                    # z로부터 Info NCE Loss 계산
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)
                
                # Loss 최소화시키도록 Network f와 g 업데이트
                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                # top-1, top-5 정확도 계산 하고 loss와 learning rate 저장
                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # 첫 번째 10 epoch 동안 Warmup
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\t Top1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # Model checkpoints 저장
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")