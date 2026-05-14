from torch.utils.data import DataLoader
import torchvision.transforms as TF
import torchvision.datasets as datasets
import kagglehub
import os

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
def scale_image(t):
    return (t * 2) - 1

def get_dataset(dataset_name='MNIST'):
    img_size = 32 if dataset_name == "MNIST" else 64
    transforms = TF.Compose(
        [
            TF.ToTensor(),
            TF.Resize((img_size, img_size), 
                      interpolation=TF.InterpolationMode.BICUBIC, 
                      antialias=True),
#             TF.RandomHorizontalFlip(),
            TF.Lambda(scale_image) # Scale between [-1, 1] 
        ]
    )
    
    if dataset_name.upper() == "MNIST":
        dataset = datasets.MNIST(root="data", train=True, download=True, transform=transforms)
    elif dataset_name == "Cifar-10":    
        dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transforms)
    elif dataset_name == "Cifar-100":
        dataset = datasets.CIFAR100(root="data", train=True, download=True, transform=transforms)
    elif dataset_name == "Flowers":
        base_path = kagglehub.dataset_download("alxmamaev/flowers-recognition", output_dir="data/Flowers")
        dataset = datasets.ImageFolder(root=os.path.join(base_path, "flowers"), transform=transforms)
    
    return dataset

def get_dataloader_external(dataset_name='MNIST', 
                   batch_size=32, 
                   pin_memory=False, 
                   shuffle=True, 
                   num_workers=0, 
                   device="cpu"
                  ):
    dataset    = get_dataset(dataset_name=dataset_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            pin_memory=pin_memory, 
                            num_workers=num_workers, 
                            shuffle=shuffle
                           )
    device_dataloader = DeviceDataLoader(dataloader, device)
    return device_dataloader

def inverse_transform(tensors):
    """Convert tensors from [-1., 1.] to [0., 255.]"""
    return ((tensors.clamp(-1, 1) + 1.0) / 2.0) * 255.0 