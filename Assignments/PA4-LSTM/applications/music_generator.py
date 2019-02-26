import torch
import torch.nn as nn
import torch.optim as optim
from dataloader.text_dataloader import split_dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.rnn import RNN

import os
import time
import pathlib
import copy
from tqdm import tqdm
import tensorboard_logger

class MusicGenerator:

    def __init__(self, conf_path):
        self.dataloader_train, self.dataloader_val, self.dataloader_test, self.conf = split_dataset(conf_path)
        self.conf_train, self.conf_val, self.conf_test = self.conf['train'], self.conf['val'], self.conf['test']

    def cross_entropy(self, input_, target, reduction='elementwise_mean'):
        logsoftmax = nn.LogSoftmax(dim=2)
        res = - target * logsoftmax(input_)
        return torch.mean(torch.sum(res, dim=2))

    def train(self):

        model_type = self.conf['model_type']
        epoches = self.conf_train['epochs']
        input_size = output_size = self.conf_train['voc_size']
        hidden_size = self.conf_train['hidden_size']
        learning_rate = self.conf_train['learning_rate']
        val_every_n = self.conf_val['val_every_n']

        # Set up folder for model saving
        cur_time = time.strftime("%Y%m%d-%H%M%S")
        self.model_path = '{}/saved/{}-{}/models/'.format(os.getcwd(), cur_time, model_type)
        model_pathlib = pathlib.Path(self.model_path)
        if not model_pathlib.exists():
            pathlib.Path(model_pathlib).mkdir(parents=True, exist_ok=True)

        # Set up folder for model saving
        result_path = '{}/saved/{}-{}/results/'.format(os.getcwd(), cur_time, model_type)
        result_pathlib = pathlib.Path(result_path)
        if not result_pathlib.exists():
            pathlib.Path(result_pathlib).mkdir(parents=True, exist_ok=True)

        # Check if your system supports CUDA
        use_cuda = torch.cuda.is_available()

        # Setup GPU optimization if CUDA is supported
        if use_cuda:
            self.computing_device = torch.device("cuda")
            print("CUDA is supported")
        else:  # Otherwise, train on the CPU
            self.computing_device = torch.device("cpu")
            print("CUDA NOT supported")

        # Model
        self.model = RNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size, type=model_type)
        self.model.to(self.computing_device)
        self.hidden_train = self.model.init_hidden(type=model_type)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

        # Some variables for training
        self.minibatch_counter = 0
        self.val_loss_min = float('inf')
        self.early_stop_counter = 0
        self.is_converged = False
        tensorboard_logger.configure(result_path, flush_secs=2)

        # Start training
        for epoch in range(epoches):

            N = 50
            N_minibatch_train_loss = 0.0

            for inputs, targets in tqdm(self.dataloader_train):
                self.minibatch_counter += 1
                inputs, targets = inputs.to(self.computing_device), targets.to(self.computing_device)

                self.model.train()
                self.optimizer.zero_grad()

                outputs, self.hidden_train = self.model(inputs, self.hidden_train)

                loss = self.cross_entropy(outputs, targets)
                tensorboard_logger.log_value('total_loss', loss.item(), self.minibatch_counter)
                N_minibatch_train_loss += loss

                loss.backward()
                self.optimizer.step()

                # Validation
                if self.minibatch_counter % val_every_n == 0:
                    self.val(epoch)

                # Calculate avg train loss
                if self.minibatch_counter % N == 0:
                    # Print the loss averaged over the last N mini-batches
                    N_minibatch_train_loss /= N
                    print('Epoch %d, average minibatch %d loss: %.3f' %
                          (epoch + 1, self.minibatch_counter, N_minibatch_train_loss))

                    # Add the averaged loss over N minibatches and reset the counter
                    tensorboard_logger.log_value('avg_train_loss', N_minibatch_train_loss.item(), self.minibatch_counter)
                    N_minibatch_train_loss = 0.0

                if self.is_converged:
                    break

            print("Finished", epoch + 1, "epochs of training")

            if self.is_converged:
                break

        print("Training complete after", epoch, "epochs")

    def val(self, epoch):
        model_type = self.conf['model_type']
        early_stop_n = self.conf_val['early_stop_n']

        self.model.eval()
        with torch.no_grad():
            val_loss = 0
            hidden_val = self.model.init_hidden(model_type)

            for val_batch_count, (inputs_val, labels_val) in enumerate(self.dataloader_val):
                inputs_val, labels_val = inputs_val.to(self.computing_device), labels_val.to(self.computing_device)
                val_outputs, hidden_val = self.model(inputs_val, hidden_val)
                loss = self.cross_entropy(val_outputs, labels_val)
                val_loss += loss

            val_loss /= (val_batch_count + 1)
            print("\nmini batch {}, val loss{}".format(self.minibatch_counter, val_loss))
            tensorboard_logger.log_value('val_loss', val_loss.item(), self.minibatch_counter)
            self.scheduler.step(val_loss)

            if val_loss < self.val_loss_min:
                model_name = os.path.join(self.model_path, "epoch_{}-batch_{}-loss_{}-{}.pt".format(epoch, self.minibatch_counter, val_loss,
                                                                      time.strftime("%Y%m%d-%H%M%S")))

                if model_type == 'LSTM':
                    hidden_to_save = (self.hidden_train[0].clone(), self.hidden_train[1].clone())
                elif model_type == 'GRU':
                    hidden_to_save = self.hidden_train.clone()

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'hidden': hidden_to_save,
                    'loss': val_loss,
                    'val_loss_min': self.val_loss_min,
                    'early_stop_counter': self.early_stop_counter
                }, model_name)
                print("\nModel saved to {}".format(model_name))
                self.best_model = model_name
                self.val_loss_min = val_loss
                self.early_stop_counter = 0

            else:
                self.early_stop_counter += 1

            if self.early_stop_counter >= early_stop_n:
                self.is_converged = True

    def test(self):
        checkpoint = torch.load(self.best_model)
        input_size = output_size = self.conf_train['voc_size']
        hidden_size = self.conf_train['hidden_size']
        model = RNN(input_size=input_size, output_size=output_size, hidden_size=hidden_size)
        model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()
        with torch.no_grad():
            test_loss = 0
            hidden_test = checkpoint['hidden']
            for test_batch_count, (inputs_test, labels_test) in enumerate(self.dataloader_test):
                inputs_test, labels_test = inputs_test.to(self.computing_device), labels_test.to(self.computing_device)
                outputs_test, hidden_test = self.model(inputs_test, hidden_test)
                loss = self.cross_entropy(outputs_test, labels_test)
                test_loss += loss
            test_loss /= (test_batch_count + 1)
            print("test loss = {}".format(test_loss))
            return test_loss

    # Todo: Train from a saved model
    def cont_train(self, model_path, model_name):
        pass

    def generate(self, temperature):
        pass