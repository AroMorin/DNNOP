import torch

class SGD(Algorithm):
    def __init__(self, model, x, y, x_t, y_t, optimizer):
        super().__init__(model, x, y, x_t, y_t, model, optimizer) # Initialize base class
        self.x = x
        self.y = y
        self.x_t = x_t
        self.y_t = y_t
        self
        self.model = model
        self.optimizer = optimizer

    def train(self):
        images = self.x # List of batches of images
        labels = self.y # List of batches of labels
        model = self.model
        optimizer = self.optimizer

        model.train() #Sets behavior to "training" mode
        for i in range(len(images)):
            optimizer.zero_grad()
            output = model(images[i])
            loss = F.nll_loss(output, labels[i])
            loss.backward()
            optimizer.step()
        print('Train Loss: %f' %loss.item())

    def test(self):
        data = dataset.test_data
        labels = dataset.test_labels
        nb_images = len(dataset.test_data)
        model.eval() #Sets behavior to "training" mode
        test_loss = 0
        correct = 0
        with torch.no_grad():
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, labels, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()

        test_loss /= nb_images
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, nb_images, 100.*correct/nb_images))
        return 100.*correct/nb_images
