import torch
import torch.nn as nn
from torchvision.models import resnet18

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet18 = resnet18(pretrained=True)
    
    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        x = self.resnet18.avgpool(x)
        x = x.squeeze()
        if len(x.size()) < 2:
            x = x.unsqueeze(0)
        return x

class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.classifier = nn.Linear(512, 7)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        #print(torch.min(x), torch.max(x))
        x = self.category_encoder(x)
        x = self.classifier(x)
        return x

class DomainDisentangleModel(nn.Module):
    def __init__(self):
        super(DomainDisentangleModel, self).__init__()
        self.feature_extractor = FeatureExtractor()

        #evaluate if use encoder with 256 output dimension
        #remember to change the 2 encoder and also the 2 classifier

        self.domain_encoder = nn.Sequential(        #reduce by an half 
            nn.Linear(512, 512),                    
            nn.BatchNorm1d(512),
            nn.ReLU(),
 
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            
        )

        self.category_encoder = nn.Sequential(         #reduce by an half
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            
        )

        self.domain_classifier = nn.Linear(256, 2)
        self.category_classifier = nn.Linear(256, 7)

        self.reconstructor = nn.Sequential(

            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),

            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),

            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512)
            #maybe it need a nn.ReLU or Sigmoind at the end
        )

    def forward(self, x, y): #x = batch y = labels of the batch
        f = self.feature_extractor(x)
        fd = self.domain_encoder(f)
        fc = self.category_encoder(f)
        
        fc_source = torch.empty(256)

        for (fi, yi) in zip(fc,y):
            if yi < 7:
                fc_source.add(fi)

        
        print("fc ",len(fc_source))
        rec = self.reconstructor(torch.cat((fd, fc), 1))
        yc = self.category_classifier(fc_source)
        yd = self.domain_classifier(fd)
        return rec, yc, yd, f
