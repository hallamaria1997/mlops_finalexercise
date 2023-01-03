import argparse
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
    cli()    
    
    
    