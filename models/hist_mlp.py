import torch.nn as nn

class HistMLP(nn.Module):
    def __init__(self, pretrained=False, auxiliary=False, num_classes=8, multi_cls=False, input_dim=1860):
        super(HistMLP, self).__init__()
        assert pretrained == False
        assert auxiliary == False
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 1024), 
            nn.BatchNorm1d(1024), 
            nn.ReLU(), 
            nn.Linear(1024, 1024), 
            nn.BatchNorm1d(1024), 
            nn.ReLU(), 
            nn.Linear(1024, 512)
        )

        if multi_cls:
            self.fc = nn.ModuleList([nn.Linear(512, 2), nn.Linear(512, 8)])
        else:
            self.fc = nn.ModuleList([nn.Linear(512, 2) for _ in range(num_classes)])

    def forward(self, x):
        feature = self.backbone(x)

        outs = []
        for i in range(len(self.fc)):
            out = self.fc[i](feature)
            outs.append(out)
        
        return outs, feature

def hist_mlp(**kwargs):
    return HistMLP(**kwargs)

def hist_mlp_16(**kwargs):
    return HistMLP(input_dim=1860, **kwargs)
    
def hist_mlp_64(**kwargs):
    return HistMLP(input_dim=53155, **kwargs)

    