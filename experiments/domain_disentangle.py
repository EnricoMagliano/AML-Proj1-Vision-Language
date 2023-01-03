import torch
from models.base_model import DomainDisentangleModel

class DomainDisentangleExperiment: # See point 2. of the project
    
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')

        # Setup model
        self.model = DomainDisentangleModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # Setup optimization procedure
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.loss_rec = torch.nn.MSELoss()
        self.loss_class_ce = torch.nn.CrossEntropyLoss()
        self.loss_domain_ce = torch.nn.CrossEntropyLoss()
    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss

        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return iteration, best_accuracy, total_train_loss

    def train_iteration(self, data):
        x, y = data     #a batch of x and y, not a single sample
        x = x.to(self.device)
        y = y.to(self.device)
        y_d = torch.tensor([])
        y_source = torch.tensor([])
        

        for yi in y:
            if yi < 7:
                y_d.add(0)
                y_source.add(yi)
                
            else:
                y_d.add(1)
        print(len(y))
        print(len(y_d))
        print("y source: ", y_source)
        print("len ", len(y_source))

        logits = self.model(x, y)
        print(logits[1])
        loss_class_ce = self.loss_class_ce(logits[1], y_source)
        loss_domain_ce = self.loss_domain_ce(logits[2], y_d)
        loss_rec = self.loss_rec(logits[1], logits[3])
        self.optimizer.zero_grad()
        loss_class_ce.backward()
        loss_domain_ce.backward()
        loss_rec.backward()
        self.optimizer.step()

        return loss_class_ce.item()

    def validate(self, loader):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad(): #not compute grad
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss += self.criterion(logits, y)
                pred = torch.argmax(logits, dim=-1)

                accuracy += (pred == y).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss    

    