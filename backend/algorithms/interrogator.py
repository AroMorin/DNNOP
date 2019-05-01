"""Base class for an Algorithm. The placeholder methods here are meant to guide
the developer, to make the class extendable intuitively.

This is a somewhat useless class, just like the model class. There is not much
that is shared among all algorithms to justify having a class.

Candidate for removal.
"""
class Interrogator(object):
    def __init__(self):
        self.inference = None

    def set_inference(self, model, test=False):
        """This method runs inference on the given environment using the models.
        I'm not sure, but I think there could be many ways to run inference. For
        that reason, I designate this function, to be a single point of contact
        for running inference, in whatever way the user/problem requires.
        """
        with torch.no_grad():
            if not test:
                # Training
                self.inference = model(self.env.observation)
            else:
                # Testing
                model.eval()  # Turn on evaluation mode
                self.inference = model(self.env.test_data)

    def print_inference(self):
        """Prints the inference of the neural networks. Attempts to extract
        the output items from the tensors.
        """
        if len(self.inference) == 1:
            x = self.inferences.item()
        elif len(self.inference) == 2:
            x = (self.inference[0].item(), self.inference[1].item())
        else:
            x = [a.item() for a in self.inference]
        print("Inference: ", x)

    def test(self):
        """This is a method for testing."""
        assert self.env.test_data is not None  # Sanity check
        self.get_inference(test=True)
        self.optim.calculate_correct_predictions(self.inference, test=True, acc=True)
        if self.env.loss:
            self.optim.calculate_loss(self.inference, test=True)

    def print_test_accuracy(self):
        """Prints the accuracy figure for the test/validation case/set."""
        test_acc = self.optim.test_acc
        if self.env.loss:
            test_loss = self.optim.test_loss  # Assuming minizming loss
            test_loss /= len(self.env.test_data)
            print('Test set: Loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(test_loss,
                                                                test_acc))
        else:
            print('Test set: Accuracy: ({:.0f}%)\n'.format(test_acc))
