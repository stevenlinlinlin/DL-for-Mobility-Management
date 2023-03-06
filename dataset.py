from torch.utils.data import Dataset, DataLoader, random_split


class myselfDataset(Dataset):
  def __init__(self, x, y):
    self.x  = x
    self.y  = y
    self.len = len(x)
  def __getitem__(self,index):
    return self.x[index], self.y[index]
  def __len__(self):
    return self.len


def get_dataloader(batch_size, x, y):
  dataset = myselfDataset(x,y)
  train_len = int(0.9 * len(dataset))
  length = [train_len, len(dataset)-train_len]
  trainset, validset = random_split(dataset,length)
  train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
  valid_loader = DataLoader(dataset=validset, batch_size=batch_size, shuffle=True)
  return train_loader, valid_loader