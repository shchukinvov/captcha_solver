import torch
import torch.nn as nn


class CaptchaCTCLoss(nn.Module):
    def __init__(self):
        super(CaptchaCTCLoss, self).__init__()
        self.ctc = nn.CTCLoss(blank=0)

    def forward(self, predictions, targets, predictions_length, targets_length):
        return self.ctc(predictions, targets, predictions_length, targets_length)


""" TESTING """
if __name__ == "__main__":
    criterion = CaptchaCTCLoss()
    inputs = torch.tensor([[[0.25, 0.91, 0.42],
                            [0.51, 0.16, 0.1],
                            [0.1, 0.66, 0.98]],
                           [[0.25, 0.91, 0.42],
                            [0.51, 0.16, 0.1],
                            [0.1, 0.66, 0.98]],
                           [[0.25, 0.91, 0.42],
                            [0.51, 0.16, 0.1],
                            [0.1, 0.66, 0.98]]])
    input_lengths = torch.full(size=(3,), fill_value=3, dtype=torch.long)
    target = torch.tensor([[1, 1, 1],
                           [0, 0, 0],
                           [2, 2, 2]])
    target_lengths = torch.tensor([3, 3, 3])
    inputs = inputs.log_softmax(2)
    loss = criterion(inputs, target, input_lengths, target_lengths)
    print(loss.item())
