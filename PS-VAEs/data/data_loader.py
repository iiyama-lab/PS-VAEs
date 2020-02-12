import torch.utils.data
from data.unaligned_dataset import UnalignedDataset


class DataLoader():
    def name(self):
        return 'DataLoader'

    def __init__(self, opt, val=False):
        self.opt = opt
        self.size = self.opt.max_dataset_size
        if val:
            self.size = 10
        self.dataset = UnalignedDataset(opt, val)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            num_workers=int(opt.nThreads))

    def __len__(self):
        return min(len(self.dataset), self.size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batchSize >= self.opt.max_dataset_size:
                break
            yield data

