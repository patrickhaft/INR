from torch.utils.data import Dataset
from utils.utils import get_tensor, get_mgrid

class ImageFitting(Dataset):
    def __init__(self, sidelength, input_img):
        super().__init__()
        img = get_tensor(sidelength, input_img)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels