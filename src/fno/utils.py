import torch


class GaussianNormalizer:
    """Normalize data to zero mean and unit variance."""

    def __init__(self, x, eps=1e-5):
        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'epoch': epoch,
    }, path)


def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        if optimizer and ckpt.get('optimizer_state_dict'):
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        return ckpt.get('epoch', 0)
    else:
        # Raw state_dict (from official training script)
        model.load_state_dict(ckpt)
        return 0
