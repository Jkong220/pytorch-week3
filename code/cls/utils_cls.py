import torch

def train(net, trainloader, optimizer, criterion, device, epoch):
    net.train()
    total, correct, running_loss = 0, 0, 0.0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print(f'Epoch {epoch}: Train Loss {running_loss/len(trainloader):.3f} | Acc {100.*correct/total:.3f}%')

def test(net, testloader, criterion, device, epoch):
    net.eval()
    total, correct, test_loss = 0, 0, 0.0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print(f'Epoch {epoch}: Test Loss {test_loss/len(testloader):.3f} | Acc {100.*correct/total:.3f}%')
