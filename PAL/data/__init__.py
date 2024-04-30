import os

import torch
import torchvision.transforms as T

from torch.utils.data import DataLoader
from data.dataset import SYSUDataset
from data.dataset import RegDBDataset
from data.dataset import LLCMData
from data.dataset import MarketDataset

from data.sampler import CrossModalityIdentitySampler
from data.sampler import CrossModalityRandomSampler
from data.sampler import RandomIdentitySampler
from data.sampler import NormTripletSampler
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing


class Denormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be denormalized.
        Returns:
            Tensor: Denormalized Tensor image.
        """
        return self.denormalize(tensor, self.mean, self.std, self.inplace)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, inplace={2})'.format(self.mean, self.std, self.inplace)

    def denormalize(self, tensor, mean, std, inplace=False):
        """Denormalize a tensor image with mean and standard deviation.
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be denormalized.
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.
            inplace(bool,optional): Bool to make this operation in-place.
        Returns:
            Tensor: Denormalized Tensor image.
        """
        if not torch.is_tensor(tensor):
            raise TypeError('tensor is not a torch image.')

        if not inplace:
            tensor = tensor.clone()

        dtype = tensor.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
        tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
        return tensor

# 使用方法
# denormalizer = Denormalize(mean=[0.4275, 0.4189, 0.4229], std=[0.2175, 0.2183, 0.2166])
# original_tensor = denormalizer(normalized_tensor)



def compute_dataset_mean_std(dataset):
    # 使用 DataLoader 来遍历数据集
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    mean = 0.0
    square_sum = 0.0
    n_samples = 0

    # for data in loader:
    #     images = data[0]
    #     batch_samples = images.size(0)
    #     n_samples += batch_samples
    #
    #     # 将图像重塑为 [batch_size, channels, -1]
    #     images = images.view(batch_samples, images.size(1), -1)
    #
    #     # 计算当前批次的均值和方差
    #     mean += images.mean(2).sum(0)
    #     square_sum += (images ** 2).mean(2).sum(0)
    #
    # # 计算整个数据集的均值和标准差
    # mean /= n_samples #n_samples
    # var = (square_sum / n_samples) - (mean ** 2)
    # std = torch.sqrt(var)

    for data in loader:
        images = data[0]
        batch_samples = images.size(0)
        n_samples += batch_samples

        # 将图像重塑为 [batch_size, channels, -1]
        images = images.view(batch_samples, images.size(1), -1)

        # 计算当前批次的均值和方差
        mean += images.mean(2).sum(0)
        square_sum += (images ** 2).mean(2).sum(0)

    # 计算整个数据集的均值和方差
    mean /= n_samples
    var = (square_sum / n_samples) - (mean ** 2)
    std = torch.sqrt(var)

    return mean, std

def collate_fn(batch):  # img, label, cam_id, img_path, img_id
    samples = list(zip(*batch))

    data = [torch.stack(x, 0) for i, x in enumerate(samples) if i != 3]
    data.insert(3, samples[3])
    return data


def get_train_loader(dataset, root, sample_method, batch_size, p_size, k_size, image_size, random_flip=False, random_crop=False,
                     random_erase=False, color_jitter=False, padding=0, num_workers=4):
    # data pre-processing
    t = [T.Resize(image_size)]
    if random_flip:
        t.append(T.RandomHorizontalFlip())

    if color_jitter:
        t.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0))

    if random_crop:
        t.extend([T.Pad(padding, fill=127), T.RandomCrop(image_size)])

    #t.extend([T.ToTensor()])
    t.extend([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #,Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if random_erase:
        # t.append(ChannelRandomErasing(probability=0.5))
        # t.append(ChannelAdapGray(probability=0.5))
        t.append(T.RandomErasing())

        # t.append(T.RandomErasing()) ###58
        # t.append(ChannelAdapGray(probability=0.5)) ###58
        # t.append(Jigsaw())

    # [0.4275, 0.4189, 0.4229]
    # [0.2175, 0.2183, 0.2166]
    #[0.4162, 0.4041, 0.4065]
    #[0.2191, 0.2208, 0.2195]

    # t = [T.Resize(image_size)] ##计算均值方差
    # t.extend([T.ToTensor()]) ##计算均值方差
    transform = T.Compose(t)

    # dataset
    if dataset == 'sysu':
        train_dataset = SYSUDataset(root, mode='train', transform=transform)
    elif dataset == 'regdb':
        train_dataset = RegDBDataset(root, mode='train', transform=transform)
    elif dataset == 'llcm':
        train_dataset = LLCMData(root, mode='train', transform=transform)
    elif dataset == 'market':
        train_dataset = MarketDataset(root, mode='train', transform=transform)
    # dataset_mean, dataset_std = compute_dataset_mean_std(train_dataset)
    # print(dataset_mean)
    # print(dataset_std)
    # import pdb
    # pdb.set_trace()

    # loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=4)
    # mean = torch.zeros(3)
    # std = torch.zeros(3)
    # for images, _, _, _, _ in loader:
    #     batch_samples = images.size(0)  # batch size (the last batch can be smaller!)
    #     images = images.view(batch_samples, images.size(1), -1)
    #     mean += images.mean(2).sum(0)
    #     std += images.std(2).sum(0)
    #
    # mean /= len(train_dataset)
    # std /= len(train_dataset)
    #
    # print(len(train_dataset))
    # import pdb
    # pdb.set_trace()

    # sampler
    assert sample_method in ['random', 'identity_uniform', 'identity_random', 'norm_triplet']
    if sample_method == 'identity_uniform':
        batch_size = p_size * k_size
        sampler = CrossModalityIdentitySampler(train_dataset, p_size, k_size)
    elif sample_method == 'identity_random':
        batch_size = p_size * k_size
        sampler = RandomIdentitySampler(train_dataset, p_size * k_size, k_size)
    elif sample_method == 'norm_triplet':
        batch_size = p_size * k_size
        sampler = NormTripletSampler(train_dataset, p_size * k_size, k_size)
    else:
        sampler = CrossModalityRandomSampler(train_dataset, batch_size)

    # loader

    # Compute the mean and std

    train_loader = DataLoader(train_dataset, batch_size, sampler=sampler, drop_last=True, pin_memory=True,
                              collate_fn=collate_fn, num_workers=num_workers)


    # mean = torch.zeros(3)
    # std = torch.zeros(3)
    # for images, _, _, _, _ in train_loader:
    #     batch_samples = images.size(0)  # batch size (the last batch can be smaller!)
    #     images = images.view(batch_samples, images.size(1), -1)
    #     mean += images.mean(2).sum(0)
    #     std += images.std(2).sum(0)
    #
    # mean /= len(train_dataset)
    # std /= len(train_dataset)
    #
    # print(len(train_dataset))
    # import pdb
    # pdb.set_trace()

    return train_loader


def get_test_loader(dataset, root, batch_size, image_size, num_workers=4):
    # transform
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])  #T.Normalize(mean=[0.4162, 0.4041, 0.4065], std=[0.2191, 0.2208, 0.2195]),Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    # transform = T.Compose([
    #     T.Resize(image_size),
    #     T.ToTensor(),
    #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    # dataset
    if dataset == 'sysu':
        gallery_dataset = SYSUDataset(root, mode='gallery', transform=transform)
        query_dataset = SYSUDataset(root, mode='query', transform=transform)
    elif dataset == 'regdb':
        gallery_dataset = RegDBDataset(root, mode='gallery', transform=transform)
        query_dataset = RegDBDataset(root, mode='query', transform=transform)
    elif dataset == 'llcm':
        gallery_dataset = LLCMData(root, mode='gallery', transform=transform)
        query_dataset = LLCMData(root, mode='query', transform=transform)
    elif dataset == 'market':
        gallery_dataset = MarketDataset(root, mode='gallery', transform=transform)
        query_dataset = MarketDataset(root, mode='query', transform=transform)

    # dataloader
    query_loader = DataLoader(dataset=query_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=False,
                              collate_fn=collate_fn,
                              num_workers=num_workers)

    gallery_loader = DataLoader(dataset=gallery_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False,
                                collate_fn=collate_fn,
                                num_workers=num_workers)

    return gallery_loader, query_loader
