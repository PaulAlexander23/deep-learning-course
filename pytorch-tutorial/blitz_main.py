import torch
import torchvision
import torchvision.transforms as transforms
import blitz
import numpy as np
import matplotlib.pyplot as plt
import subprocess


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('images.png')
    subprocess.call(["eog", "images.png"])

def load():
    transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5),
                                                        (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4,
                                              shuffle = True, num_workers = 2)

    testset = torchvision.datasets.CIFAR10(root='./data', train = False,
                                           download = True, transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 4,
                                             shuffle = False, num_workers = 2)
    return trainset, trainloader, testset, testloader

def example_output(loader):
    dataiter = iter(loader)
    images, labels = dataiter.next()
    
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

if __name__ == "__main__":
    # Initialise
    trainset, trainloader, testset, testloader = load()
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
               'ship', 'truck')
    net = blitz.Net()
    
    # Train
    print('Started Training')
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            net.train(inputs, labels)
            running_loss += net.loss.item()
            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')

    # Test
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    outputs = nets(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
            
