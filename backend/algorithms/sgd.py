import torch

class SGD(Algorithm):
    def __init__(self, env, optimizer):
        super().__init__(model) # Initialize base class
        self.optimizer = optimizer
        self.training_loss = ''
        self.test_loss = ''

        self.training_acc = ''
        self.test_acc = ''

        self.correct_test_preds = ''

    def optimize(self, env):
        self.model.train() #Sets behavior to "training" mode
        self.optimizer.zero_grad()
        predictions = self.model(env.x)
        self.train_loss = F.nll_loss(predictions, env.y)
        self.train_loss.backward()
        optimizer.step()

    def test(self):
        # variable definitions
        images = env.x_t
        labels = env.y_t

        self.model.eval() #Sets behavior to "training" mode
        self.test_loss = 0
        self.correct_test_preds = 0
        with torch.no_grad():
            predictions = self.model(images)
            self.test_loss = F.nll_loss(predictions, labels, reduction='sum').item()
            # get the index of the max log-probability
            pred = predictions.max(1, keepdim=True)[1]
            self.correct_test_preds += pred.eq(labels.view_as(pred)).sum().item()

    def get_test_accuracy(self):
        self.test_loss /= env.nb_test_items
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            self.test_loss, correct, nb_images, 100.*correct/nb_images))
        return 100.*correct/nb_images
