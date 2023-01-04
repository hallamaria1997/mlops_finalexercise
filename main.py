import argparse
import sys

import torch

from data import CorruptMnist
from model import MyAwesomeModel

import matplotlib.pyplot as plt

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=1e-3)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        model = model.to(self.device)
        train_set = CorruptMnist(train=True)
        dataloader = torch.utils.data.DataLoader(train_set, batch_size=128)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        n_epoch = 5
        for epoch in range(n_epoch):
            loss_tracker = []
            for batch in dataloader:
                optimizer.zero_grad()
                x, y = batch
                preds = model(x.to(self.device))
                loss = criterion(preds, y.to(self.device))
                loss.backward()
                optimizer.step()
                loss_tracker.append(loss.item())
            print(f"Epoch {epoch+1}/{n_epoch}. Loss: {loss}")        
        torch.save(model.state_dict(), 'trained_model.pt')
            
        plt.plot(loss_tracker, '-')
        plt.xlabel('Training step')
        plt.ylabel('Training loss')
        plt.savefig("training_curve.png")
        
        return model
            
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model = MyAwesomeModel()
        model.load_state_dict(torch.load(args.load_model_from))
        model = model.to(self.device)

        test_set = CorruptMnist(train=False)
        dataloader = torch.utils.data.DataLoader(test_set, batch_size=128)
        
        correct, total = 0, 0
        for batch in dataloader:
            x, y = batch
            
            preds = model(x.to(self.device))
            preds = preds.argmax(dim=-1)
            
            correct += (preds == y.to(self.device)).sum().item()
            total += y.numel()
            
        print(f"Test set accuracy {correct/total}")


if __name__ == '__main__':
    TrainOREvaluate()


''' import argparse
import sys
import torch
import click
from data import mnist
from model import MyAwesomeModel
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr= 0.03):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    print('model created')
    train_set, _ = mnist()
    print('accessed data')
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    print('created trainloader')
    images, labels = iter(trainloader).next()
    print(labels)

    #define the criterion
    criterion = nn.CrossEntropyLoss()
    print('created criterion')
    #define the optimizer
    optimizer = optim.SGD(model.parameters(), lr = lr)
    epochs = 30
    print('created optimizer')
    
    train_accuracies = []
    for e in range(epochs):
        print(e)
        training_error = 0
        for images, labels in trainloader:
            output = model(images)
            loss = criterion(output, labels.long())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            training_error = loss.item()
            train_accuracies.append(training_error)
        
        print("training error", training_error)
    
    
    
    #checkpoint = {'input_size': 784,
    #          'output_size': 10,
    #          'hidden_layers': [9216,128],
    #          'state_dict': model.state_dict()}
    #torch.save(model, 'C:/Users/Lenovo/Documents/dtu_mlops/dtu_mlops/s1_development_environment/exercise_files/final_exercise/checkpoint.pth')
    #torch.save(model.state_dict(), 'checkpoint.pth')
    
    
    return train_accuracies



@click.command()
@click.argument("model_checkpoint")
def evaluate(model_in):
    print("Evaluating until hitting the ceiling")

    # TODO: Implement evaluation logic here
    model = model_in
    #model = load_checkpoint('checkpoint.pth')
    #model = torch.load(model_checkpoint)
    _, test_set = mnist()
    
    testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()

    epochs = 2
    print('testing')
    train_losses, test_losses = [], []
    for e in range(epochs):
        test_loss = 0
        accuracy = 0
        ## TODO: Implement the validation pass and print out the validation accuracy
        with torch.no_grad():
            for images, labels in testloader:
                log_ps = model(images)
                loss = criterion(log_ps.LongTensor, labels.LongTensor)
                test_loss += loss.item()
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equal = top_class == labels.view(*top_class.shape)
                
                #print(torch.mean(equal.type(torch.FloatTensor)).item())
                accuracy += torch.mean(equal.type(torch.FloatTensor)).item()
        print("testing data", test_loss/len(testloader))    
        print("Accuracy", accuracy/len(testloader))


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()     '''
      
    