import sys
import numpy as np
import pandas as pd 
import pip

from sklearn.preprocessing import LabelEncoder

import torch
from tqdm import tqdm
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader

is_debug=False

print("Data import...")
# Reading competition data
df = pd.read_csv("transactions_train.csv", dtype={"article_id": str})

print("Data cleaning...")
# Converting 't_dat' column to datetime
df['t_dat'] = pd.to_datetime(df['t_dat'])

# Grouping active customer purchases
active_articles = df.groupby("article_id")["t_dat"].max().reset_index()
active_articles = active_articles[active_articles["t_dat"] >= "2019-09-01"].reset_index()

df = df[df["article_id"].isin(active_articles["article_id"])].reset_index(drop=True)

# Creating 'week' column
df["week"] = (df["t_dat"].max() - df["t_dat"]).dt.days // 7

article_ids = np.concatenate([["placeholder"], np.unique(df["article_id"].values)])

le_article = LabelEncoder()
le_article.fit(article_ids)
df["article_id"] = le_article.transform(df["article_id"])

WEEK_HIST_MAX = 5

def create_dataset(df, week):
    hist_df = df[(df["week"] > week) & (df["week"] <= week + WEEK_HIST_MAX)]
    hist_df = hist_df.groupby("customer_id").agg({"article_id": list, "week": list}).reset_index()
    hist_df.rename(columns={"week": "week_history"}, inplace=True)

    target_df = df[df["week"] == week]
    target_df = target_df.groupby("customer_id").agg({"article_id": list}).reset_index()
    target_df.rename(columns={"article_id": "target"}, inplace=True)
    target_df["week"] = week

    return target_df.merge(hist_df, on="customer_id", how="left")

#train_weeks = [0, 1, 2, 3]
train_weeks = [i for i in range(WEEK_HIST_MAX)]

train_df = pd.concat([create_dataset(df, w) for w in train_weeks]).reset_index(drop=True)

if is_debug:
  train_df = train_df.sample(n = 10000)

train_articles = train_df['customer_id'].unique().tolist()

# Preparing data loaders for images

class HMDataset(Dataset):
  def __init__(self, df, seq_len, is_test=False):
    self.df = df.reset_index(drop=True)
    self.seq_len = seq_len
    self.is_test = is_test

  def __len__(self):
    return self.df.shape[0]

  def __getitem__(self, index):
    row = self.df.iloc[index]

    if self.is_test:
      target = torch.zeros(2).float()
    else:
      target = torch.zeros(len(article_ids)).float()
      for t in row.target:
        target[t] = 1.0

    article_hist = torch.zeros(self.seq_len).long()
    week_hist = torch.ones(self.seq_len).float()

    if isinstance(row.article_id, list):
      if len(row.article_id) >= self.seq_len:
        article_hist = torch.LongTensor(row.article_id[-self.seq_len:])
        week_hist = (torch.LongTensor(row.week_history[-self.seq_len:]) - row.week) / WEEK_HIST_MAX / 2
      else:
        article_hist[-len(row.article_id):] = torch.LongTensor(row.article_id)
        week_hist[-len(row.article_id):] = (torch.LongTensor(row.week_history) - row.week) / WEEK_HIST_MAX / 2

    return article_hist, week_hist, target

# function for adjusting learning rate
def adjust_lr(optimizer, epoch):
  if epoch < 1:
    lr = 5e-5
  elif epoch < 6:
    lr = 1e-3
  elif epoch < 9:
    lr = 1e-4
  else:
    lr = 1e-5

  for p in optimizer.param_groups:
    p['lr'] = lr
  return lr 

def get_optimizer(net):
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=3e-4, betas=(0.9, 0.999), eps=1e-08)
  return optimizer

class HMModel(nn.Module):
  def __init__(self, article_shape):
    super(HMModel, self).__init__()

    self.article_emb = nn.Embedding(article_shape[0], embedding_dim=article_shape[1])

    self.article_likelihood = nn.Parameter(torch.zeros(article_shape[0]), requires_grad=True)
    self.top = nn.Sequential(nn.Conv1d(3, 8, kernel_size=1), nn.LeakyReLU(), nn.BatchNorm1d(8),
                             nn.Conv1d(8, 32, kernel_size=1), nn.LeakyReLU(), nn.BatchNorm1d(32),
                             nn.Conv1d(32, 8, kernel_size=1), nn.LeakyReLU(), nn.BatchNorm1d(8),
                             nn.Conv1d(8, 1, kernel_size=1))
    
  def forward(self, inputs):
    article_hist, week_hist = inputs[0], inputs[1]
    x = self.article_emb(article_hist)
    x = F.normalize(x, dim=2)

    x = x@F.normalize(self.article_emb.weight).T

    x, indices = x.max(axis=1)
    x = x.clamp(1e-3, 0.999)
    x = -torch.log(1/x - 1)

    max_week = week_hist.unsqueeze(2).repeat(1, 1, x.shape[-1]).gather(1, indices.unsqueeze(1).repeat(1, week_hist.shape[1], 1))

    x = torch.cat([x.unsqueeze(1), max_week, self.article_likelihood[None, None, :].repeat(x.shape[0], 1, 1)], axis=1)

    x = self.top(x).squeeze(1)
    return x


model = HMModel((len(le_article.classes_), 512))
model = model.cuda()

def calc_map(topk_preds, target_array, k=12):
  metric = []
  tp, fp = 0, 0

  for pred in topk_preds:
    if target_array[pred]:
      tp += 1
      metric.append(tp/(tp + fp))
    else:
      fp += 1

  return np.sum(metric) / min(k, target_array.sum())

def read_data(data):
  return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()

def validate(model, val_loader, k=12):
  model.eval()

  tbar = tqdm(val_loader, file=sys.stdout)

  maps = []

  with torch.no_grad():
    for idx, data in enumerate(tbar):
      inputs, target = read_data(data)

      logits = model(inputs)

      _, indices = torch.topk(logits, k, dim=1)

      indices = indices.detach().cpu().numpy()
      target = target.detach().cpu().numpy()

      for i in range(indices.shape[0]):
        maps.append(calc_map(indices[i], target[i]))

  return np.mean(maps)

SEQ_LEN = 16
BS = 256
NW = 2

def dice_loss(y_pred, y_true):
  y_pred = y_pred.sigmoid()
  intersect = (y_true * y_pred).sum(axis=1)

  return 1 - (intersect/(intersect + y_true.sum(axis=1) + y_pred.sum(axis=1))).mean()

def train(model, train_loader, epochs, val_loader = None):
  SEED = 0
  np.random.seed(SEED)

  optimizer = get_optimizer(model)
  scaler = torch.cuda.amp.GradScaler()

  criterion = torch.nn.BCEWithLogitsLoss()

  for e in range(epochs):
    model.train()
    tbar = tqdm(train_loader, file=sys.stdout)

    lr = adjust_lr(optimizer, e)

    loss_list = []

    for idx, data in enumerate(tbar):
      inputs, target = read_data(data)
      optimizer.zero_grad()

      with torch.cuda.amp.autocast():
        logits = model(inputs)
        loss = criterion(logits, target) + dice_loss(logits, target)

      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

      loss_list.append(loss.detach().cpu().item())
      avg_loss = np.round(100 * np.mean(loss_list), 4)
      tbar.set_description(f"Epoch {e+1} Loss: {avg_loss} lr: {lr}")

    log_text = f"Epoch {e+1} \n Train Loss: {avg_loss}"
  return model

test_df = pd.read_csv("sample_submission.csv").drop("prediction", axis=1)

def create_test_dataset(test_df):
  week = -1
  test_df["week"] = week

  hist_df = df[(df["week"] > week) & (df["week"] <= week + WEEK_HIST_MAX)]
  hist_df = hist_df.groupby("customer_id").agg({"article_id": list, "week": list}).reset_index()
  hist_df.rename(columns={"week": "week_history"}, inplace=True)

  return test_df.merge(hist_df, on="customer_id", how="left")

test_df = create_test_dataset(test_df)

def inference(model, loader, k=12):
  model.eval()

  tbar = tqdm(loader, file=sys.stdout)

  preds = []

  with torch.no_grad():
    for idx, data in enumerate(tbar):
      inputs, target = read_data(data)
      logits = model(inputs)

      _, indices = torch.topk(logits, k, dim=1)

      indices = indices.detach().cpu().numpy()
      target = target.detach().cpu().numpy()

      for i in range(indices.shape[0]):
        preds.append(" ".join(list(le_article.inverse_transform(indices[i]))))

  return preds

if __name__ == "__main__":
  MODEL_NAME = "exp001"

  train_dataset = HMDataset(train_df, SEQ_LEN)
  train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=NW, pin_memory=False, drop_last=True)
  "Training model ..."
  model = train(model, train_loader, epochs=50)

  test_ds = HMDataset(test_df, SEQ_LEN, is_test=True)
  test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=NW, pin_memory=False, drop_last=False)

  print("Predicting...")
  test_df["prediction"] = inference(model, test_loader)

  print("Creating submission file...")
  test_df.to_csv("submission1.csv", index=False, columns=["customer_id", "prediction"])