import math
import warnings
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F
import os
import pynvml
from transformers import Trainer, TrainerCallback
import time

#print current time in hh:mm:ss
from datetime import datetime
start_time = time.time()
start_time_str = time.strftime("%H:%M:%S", time.localtime(start_time))
print("Current Time =", start_time_str, flush=True)

# Initialize NVML
pynvml.nvmlInit()
# Get the handle of the first GPU
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

#print cuda details
try:
    print("cuda details", (torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_name()), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
except:
    print("cuda not available", flush=True)



os.environ['TRANSFORMERS_CACHE'] = 'cache'

#3D reservoir
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepReservoirNet(nn.Module):
    def __init__(self, configs):
        super(DeepReservoirNet, self).__init__()

        #input_size=768, reservoir_size=1000, output_size=768, spectral_radius=0.9, leaky=0.3, sparsity=0.5

        self.input_size = configs.seq_len
        self.reservoir_size = configs.reservoir_size
        self.output_size = configs.seq_len
        self.spectral_radius = configs.spectral_radius
        self.leaky = configs.leaky

        self.W_in = nn.Linear(self.input_size, self.reservoir_size, bias=False)
        self.W_in.weight.requires_grad = False
        self.W_res = nn.Linear(self.reservoir_size, self.reservoir_size, bias=False)
        self.W_res.weight.requires_grad = False
        self.W_out = nn.Linear(self.reservoir_size, self.output_size)
        self.res_state = torch.zeros(1, configs.reservoir_size)

        self.W_res_norm = self.compute_spectral_radius(configs.sparsity)



    def compute_spectral_radius(self, sparsity=0.5):
        with torch.no_grad():
            self.W_res.weight.data = torch.randn(self.reservoir_size, self.reservoir_size)
            # set a fraction of the entries to zero
            num_zeros = int(sparsity * self.reservoir_size ** 2)
            idxs = torch.randperm(self.reservoir_size ** 2)[:num_zeros]
            self.W_res.weight.data.view(-1)[idxs] = 0

            eigenvals = torch.linalg.eigvals(self.W_res.weight)
            radius = torch.max(torch.abs(eigenvals))
            self.W_res.weight.data /= radius
        return radius

    def forward(self, input_data, res_state):
        #print()
        # Compute reservoir state
        re_init=False
        outputs = []
        batch_size = input_data.shape[0]
        input_data = input_data.permute(0, 2, 1)



        #print("i_data", input_data.shape)
        input_proj = self.W_in(input_data)

        res_proj = self.W_res(res_state.to(input_data.device))

        res_state=res_state.to(input_data.device)
        #print('res_proj', res_proj.shape, 'input_proj', input_proj.shape)



        res_state = (1 - self.leaky) * res_proj + self.leaky * F.tanh(input_proj + res_proj)
        #print('fres_state', res_state.shape)
        #print( (1 - self.leaky), (0.2*res_state).shape)
        # Normalize reservoir state
        res_state = res_state / self.W_res_norm
        #print('here-1',res_state.shape )

        # Compute output
        output = self.W_out(res_state)



        return {'Output':output.permute(0, 2, 1), "State": res_state}

import math

class OurLayers(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, configs):
        super(OurLayers, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len #+ configs.label_len

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.hidden_size

        num_head = self.seq_len//configs.num_seq_len_heads

        self.seq_layers = nn.ModuleList([nn.Linear(self.seq_len, num_head) for _ in range(configs.num_seq_len_heads)])

        self.position_embedding = nn.Embedding(self.seq_len, self.channels)
        self.position_indices = torch.arange(self.seq_len)

        self.norm = nn.LayerNorm(self.channels)

        self.reservoir_state = None

        self.reservoir_layer = DeepReservoirNet(configs)
        self.reservoir_size = configs.reservoir_size

        self.norm_value = configs.hidden_size


        #self.transform = TabularBertPredictionHeadTransform(config)
        #print('decoder_dropout', decoder_dropout)
        self.dropout = nn.Dropout(configs.hidden_dropout_prob)
        self.act = nn.SiLU()

        self.softmax = nn.Softmax()

    def forward(self, x, memory=None, memory_init=None):
        position_embeddings = self.position_embedding(self.position_indices.to(x.device))

        # x: [Batch, Input length, Channel]
        seq = x.permute(0,2,1)

        futures = [torch.jit.fork(layer, seq) for layer in self.seq_layers]

        # Wait for each computation to finish and collect outputs
        seq = [torch.jit.wait(future) for future in futures]
        #print(len(seq))
        #print(seq[0].shape)

        # Concatenate all outputs along the feature dimension
        seq = torch.cat(seq, dim=2)

        seq = seq.permute(0,2,1)
        #print(seq.shape)

        #x = x + seq_last
        if memory_init or self.reservoir_state is None:
            self.reservoir_state = torch.zeros(x.shape[0], self.channels, self.reservoir_size).to(x.device)
        if memory:
            if memory.shape[0] > seq.shape[0]:
                re_init=True
                #print('here')
                memory =memory[0:seq.shape[0], :, :]
            elif memory.shape[0] < seq.shape[0]:
                re_init=True
                #print('here-222')
                memory = torch.zeros(x.shape[0], self.channels, self.reservoir_size).to(x.device)
            re_output = self.reservoir_layer(seq, memory)
        else:
            if self.reservoir_state.shape[0] > seq.shape[0]:
                re_init=True
                #print('here')
                self.reservoir_state =self.reservoir_state[0:seq.shape[0], :, :]
            elif self.reservoir_state.shape[0] < seq.shape[0]:
                re_init=True
                #print('here-222')
                self.reservoir_state = torch.zeros(x.shape[0], self.channels, self.reservoir_size).to(x.device)

            re_output = self.reservoir_layer(seq, self.reservoir_state.detach() )

        self.reservoir_state = re_output['State']
        mem = re_output['Output']


        #seq = (self.softmax(mem))* seq + seq
        score = self.softmax(mem* seq) #/ math.sqrt(self.norm_value)
        seq = (1-score) + seq
        x = self.dropout(seq)
        #x = x + position_embeddings

        return x # [Batch, Output length, Channel]

class Normalize(nn.Module):
    def __init__(self, seq_len: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param seq_len: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.seq_len = seq_len
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.seq_len))
        self.affine_bias = nn.Parameter(torch.zeros(self.seq_len))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import PretrainedConfig

class ContinuousScalingEmbedding(nn.Module):
    def __init__(self, config):
        super(ContinuousScalingEmbedding, self).__init__()
        self.seq_len = config.seq_len
        self.embedding_dim = config.hidden_size
        self.embedding_weights = nn.Parameter(torch.randn( self.seq_len,  self.embedding_dim))
        self.transform = nn.Linear(config.feature_dim, config.hidden_size)



    def forward(self, input_data):
        #print(input_data.shape)
        _, sequence_length, _ = input_data.shape
        transform = self.transform(input_data.float())
        #print('transform', transform.shape)
        #print('self.embedding_weights', self.embedding_weights.shape)

        embedded_data = transform * self.embedding_weights[:sequence_length].unsqueeze(0)
        return embedded_data


class Encoder(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, configs):
        super(Encoder, self).__init__()
        self.ContinuousScalingEmbedding = ContinuousScalingEmbedding(configs)



        self.layers = nn.ModuleList(OurLayers(configs) for _ in range(configs.num_layers))
        self.dropout = nn.Dropout(configs.hidden_dropout_prob)

    def forward(self, x):


        resd = x.clone()

        #x = self.ContinuousScalingEmbedding(x)
        for layer in self.layers:
                x = layer(x+ resd)
                x = self.dropout(x)

        return x + resd

class TModel(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, configs):
        super(TModel, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len #+ configs.label_len

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs.feature_dim
        self.dim = nn.Linear(configs.hidden_size, self.channels)

        self.Linear = nn.Linear(self.seq_len, self.pred_len)

        decoder_dropout = (
            configs.decoder_dropout if configs.decoder_dropout is not None else configs.hidden_dropout_prob
        )
        #self.transform = TabularBertPredictionHeadTransform(config)
        #print('decoder_dropout', decoder_dropout)

        self.dropout = nn.Dropout(decoder_dropout)


    def forward(self, x):
        #x = self.dim(x)
        #x = self.dropout(x)

        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = self.dropout(x)
        x = x + seq_last


        return x # [Batch, Output length, Channel]



class RTTimeSeries(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.decoder = TModel(config)
        self.encoder = Encoder(config)
        self.normalize_layers = Normalize(config.feature_dim, affine=False)

    def forward(
        self,
        past_inputs=None,
        feature_inputs=None,

    ):


        batch_size = past_inputs.shape[0]
        past_inputs = self.normalize_layers(past_inputs, 'norm')

        output = self.encoder(past_inputs)

        logits = self.decoder(output)
        logits = self.normalize_layers(logits, 'denorm')

        loss = None
        if feature_inputs is not None:


            feature_inputs = feature_inputs.to(logits.dtype)
            loss_fct = nn.MSELoss()#nn.HuberLoss(delta=1.0)
            loss = loss_fct(logits[:, -self.config.pred_len:, :].squeeze(), feature_inputs[:, -self.config.pred_len:, :].squeeze())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )

import pickle

import argparse

def get_model_config_from_args():
    parser = argparse.ArgumentParser(description="Model Configuration")

    parser.add_argument("--seq_len", type=int, default=12*30, help="Sequence length")
    parser.add_argument("--feature_dim", type=int, default=7, help="Feature dimension")
    parser.add_argument("--pred_len", type=int, default=720, help="Prediction length")
    parser.add_argument("--decoder_dropout", type=float, default=0.00, help="Decoder dropout rate")
    parser.add_argument("--inner", type=int, default=2, help="Inner dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--hidden_size", type=int, default=7, help="Hidden size")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.0, help="Hidden dropout probability")
    parser.add_argument("--reservoir_size", type=int, default=70, help="Reservoir size")
    parser.add_argument("--spectral_radius", type=float, default=0.5, help="Spectral radius")
    parser.add_argument("--leaky", type=float, default=0.3, help="Leaky rate")
    parser.add_argument("--sparsity", type=float, default=0.1, help="Sparsity")
    parser.add_argument("--num_seq_len_heads", type=int, default=12, help="Number of sequence length heads")

    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--data_path", type=str, default="dataset/ETT-small/ETTm2.csv", help="Data path")
    parser.add_argument("--label_len", type=int, default=1, help="Label length")
    parser.add_argument("--data", type=str, default="ETTh1", help="Dataset")

    args = parser.parse_args()

    model_config = PretrainedConfig()
    model_config.seq_len = args.seq_len
    model_config.feature_dim = args.feature_dim
    model_config.pred_len = args.pred_len
    model_config.decoder_dropout = args.decoder_dropout
    model_config.inner = args.inner
    model_config.num_layers = args.num_layers
    model_config.hidden_size = args.hidden_size
    model_config.hidden_dropout_prob = args.hidden_dropout_prob
    model_config.reservoir_size = args.reservoir_size
    model_config.spectral_radius = args.spectral_radius
    model_config.leaky = args.leaky
    model_config.sparsity = args.sparsity
    model_config.num_seq_len_heads = args.num_seq_len_heads
    
    model_config.epochs = args.epochs
    model_config.batch_size = args.batch_size
    model_config.data_path = args.data_path
    model_config.label_len = args.label_len
    model_config.data = args.data

    for key, value in vars(args).items():
        print(f"{key}: {value}", flush=True)

    return model_config

model_config = get_model_config_from_args()
model = RTTimeSeries(config=model_config)


model = RTTimeSeries(config=model_config)

# Try to pickle the model
try:
    serialized = pickle.dumps(model)
    print("Model is serializable!", flush=True)
except TypeError as e:
    print(f"Failed to serialize model: {e}", flush=True)

import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from data_provider_reservoir.data_factory import data_provider

import os

size=(model_config.seq_len, model_config.label_len, model_config.pred_len)
model_config.size = size
data_path=model_config.data_path
# train_dataset = Dataset_ETT_minute(root_path='', flag='train', size=size,
#                 features='M', data_path=data_path,
#                 target='OT', scale=True, timeenc=0, freq='t',)
# test_dataset = Dataset_ETT_minute(root_path='', flag='test', size=size,
#                 features='M', data_path=data_path,
#                 target='OT', scale=True, timeenc=0, freq='t')
# val_dataset = Dataset_ETT_minute(root_path='', flag='val', size=size,
#                 features='M', data_path=data_path,
#                 target='OT', scale=True, timeenc=0, freq='t')
train_dataset = data_provider(model_config, 'train')
val_dataset = data_provider(model_config, 'val')
test_dataset = data_provider(model_config, 'test')

for i in train_dataset:

    X_train = i[0]
    y_train = i[1]
    print(len(X_train),len(y_train), flush=True)
    break

for i in test_dataset:
    X_test = i[0]
    y_test = i[1]
    print(len(X_test),len(y_test), flush=True)
    break

for i in val_dataset:
    X_val = i[0]
    y_val = i[1]
    print(len(X_val),len(y_val), flush=True)
    break

import numpy as np

def reorganize_datasets(xtrain, ytrain, batch_size):
    # Get the original number of samples
    num_samples = xtrain.shape[0]

    # Generate the original indices
    original_indices = list(range(num_samples))

    # Create the new index list based on the given pattern
    new_indices = []
    for i in range(batch_size):
        new_indices.extend(original_indices[i::batch_size])

    # Reorder the datasets using the new indices
    xtrain_reordered = xtrain[new_indices]
    ytrain_reordered = ytrain[new_indices]

    return xtrain_reordered, ytrain_reordered

import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, tokenized_inputs,  labels=None, pos=None):
        self.tokenized_inputs = tokenized_inputs
        self.labels = labels
        self.pos = pos
        self.id_list = None
        self.re = None

    def __len__(self):
        return len(self.tokenized_inputs)

    def __getitem__(self, idx):
        if self.labels is not None:
            return {
                "past_inputs": torch.tensor(self.tokenized_inputs[idx]).float(),
                "feature_inputs": torch.tensor(self.labels[idx]).float(),
                #"id": torch.tensor(self.id_list[idx]),  # Include the id directly
                #"reservoir_ids": torch.tensor(self.re[idx]),
            }
        else:
            return {
                "past_inputs": torch.tensor(self.tokenized_inputs[idx]),
            }

# Assuming you have X_train, y_train, X_test, y_test, trainpos, and testpos defined

X_train, y_train = reorganize_datasets(np.array(X_train), np.array(y_train), model_config.batch_size)

X_test, y_test = reorganize_datasets(np.array(X_test), np.array(y_test), model_config.batch_size)
X_val, y_val = reorganize_datasets(np.array(X_val), np.array(y_val), model_config.batch_size)

# Assuming you have X_train, y_train, X_test, y_test, trainpos, and testpos defined

train_dataset = CustomDataset(X_train, y_train)

test_dataset = CustomDataset(X_test, y_test)

val_dataset = CustomDataset(X_val, y_val)

for i in train_dataset:

    print(i['past_inputs'].shape,i['feature_inputs'].shape, flush=True)
    break

from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

import time
from transformers import TrainerCallback
import pynvml

class MetricsCallback(TrainerCallback):
  def __init__(self, trainer):
      self.trainer = trainer
      self.epoch_start_time = None

  def on_epoch_begin(self, args, state, control, **kwargs):
      self.epoch_start_time = time.time()

  def on_epoch_end(self, args, state, control, **kwargs):
      # Calculate epoch duration
      epoch_duration = time.time() - self.epoch_start_time

      # Evaluate on validation set
      vali_output = self.trainer.evaluate(eval_dataset=self.trainer.eval_dataset)
      vali_loss = vali_output['eval_loss']
      vali_mae_loss = vali_output.get('eval_mae', None)

      # Get GPU utilization
      gpu_utilization = 0
      def get_gpu_utilization():
          handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming we're using the first GPU
          memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
          return memory.used
      try:
          pynvml.nvmlInit()
          gpu_utilization = get_gpu_utilization()
          pynvml.nvmlShutdown()
      except:
          gpu_utilization = 0

      print(
          "Epoch: {0} | Val MSE Loss: {1:.7f} | "
          "Val MAE Loss: {2:.7f} | Max GPU Utilization: {3:.2f} MB | "
          "Epoch Duration: {4:.2f} s".format(
              state.epoch, vali_loss, vali_mae_loss, 
              gpu_utilization/1024**2, epoch_duration
          ), flush=True)

class TimeTrainer(Trainer):
    def __init__(self, *args, gradient_accumulation_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.scaler = GradScaler()
        self.add_callback(MetricsCallback(self))

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        loss = loss / self.gradient_accumulation_steps
        self.scaler.scale(loss).backward()

        return loss.detach()


    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset


        loader =  DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            drop_last=self.args.dataloader_drop_last,
            shuffle = False,
        )
        return loader

torch.cuda.is_available()

from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import mean_squared_error, mean_absolute_error
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback, IntervalStrategy
from sklearn.metrics import r2_score, accuracy_score
import numpy as np

def compute_metrics_regression(p):


    preds = p.predictions[:, -model_config.pred_len:, :].flatten()
    labels = p.label_ids[:, -model_config.pred_len:, :].flatten()

    r2 = r2_score(labels, preds)
    mse = mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)

    return {"r2_score": r2, "mse": mse, "mae": mae}

if torch.cuda.is_available():
    model.to(device)
    print("Using device:", device, flush=True)
else:
    print("Using CPU", flush=True)

#  create checkpoint folder
if not os.path.exists('checkpoint'):
    os.makedirs('checkpoint')

training_args = TrainingArguments(
    #reservoir_datasetname_pred_len
    output_dir='checkpoint/reservoir_'+model_config.data+'_'+str(model_config.pred_len),
    num_train_epochs=model_config.epochs,
    label_names=["feature_inputs"],
    #disable_tqdm = False,
    weight_decay=0.0,
    learning_rate=1e-4,

    do_eval=True,

    per_device_train_batch_size=model_config.batch_size,
    per_device_eval_batch_size=model_config.batch_size,
    logging_dir='./logs',
    logging_strategy="steps",
    logging_steps=3000,
    evaluation_strategy="steps",
    eval_steps = 3000,
    save_strategy="steps",
    save_steps=3000,
    remove_unused_columns=False,


    save_total_limit=2,
    load_best_model_at_end=True,
)

trainer = TimeTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics_regression, #compute_metrics1,#compute_metrics_classification,

)

#
trainer.train()

trainer.evaluate(test_dataset)

# Initialize counters for trainable and non-trainable parameters
trainable_params = 0
non_trainable_params = 0

# Iterate over model parameters
for param in model.parameters():
    if param.requires_grad:
        trainable_params += param.numel()
    else:
        non_trainable_params += param.numel()

# Print the results
print(f"Trainable parameters: {trainable_params}", flush=True)
print(f"Non-trainable parameters: {non_trainable_params}", flush=True)

end_time = time.time()
end_time_str = time.strftime("%H:%M:%S", time.localtime(end_time))
print("Current Time =", end_time_str, flush=True)
#start_time - end_time in hh:mm:ss
print("Duration: ", end_time - start_time, flush=True)