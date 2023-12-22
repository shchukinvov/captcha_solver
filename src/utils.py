from itertools import groupby
import random
from math import ceil
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from jiwer import cer
from captcha_solver.data.scripts.generator_config import CHAR_LIST


ENCODER = LabelEncoder()
ENCODER.fit(CHAR_LIST)


def encode_seq(seq: str) -> list:
    return ENCODER.transform([ch for ch in seq])


def decode_seq(prediction: torch.Tensor) -> str:
    blank_idx = 0
    prediction = prediction.argmax(dim=1)
    a = [k.item() for k, _ in groupby(prediction) if k != blank_idx]
    return "".join(ENCODER.inverse_transform(a))


def show_losses(train_loss: list, val_loss: list) -> None:
    """
    :param train_loss: list of train losses during training
    :param val_loss: list of validation losses during training
    :return: loss/epoch plot
    """
    k = ceil(len(train_loss) / len(val_loss))
    plt.plot(train_loss, label='train_loss')
    plt.plot(np.arange(0, len(val_loss) * k, k), val_loss, label='val_loss')
    plt.legend()
    plt.ylim(0, 1)
    plt.show()


def cer_accuracy_score(predictions: list[str], targets: list[str] | tuple[str]) -> tuple[list, list]:
    """
    :param predictions: List of predicted labels
    :param targets: List or tuple of target labels
    :return: Tuple of (CER, accuracy) where accuracy is % fully corrected labels.
    """
    if len(predictions) != len(targets):
        raise RuntimeError("List of predictions must be same shape as list of labels")
    cer_scores = []
    accuracy = []
    for ref, hip in zip(predictions, targets):
        accuracy.append(1) if ref == hip else accuracy.append(0)
        if ref:
            cer_scores.append(cer(ref, hip))
        else:
            cer_scores.append(1)
    return cer_scores, accuracy


def seed_all(seed: int) -> None:
    """
    Seed everything for deterministic experiment
    :param seed:
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_fn(model, dataloaders, criterion, optimizer, num_epochs, device, scheduler=None):
    train_dataloader, val_dataloader = dataloaders
    train_loss = []
    val_loss = []
    cer_scores = []
    acc_scores = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch+1, num_epochs), flush=True)

        with tqdm(total=len(train_dataloader)) as progress:
            losses = []
            model.train()

            for inputs, labels in train_dataloader:
                inputs = inputs.to(device)
                targets = [encode_seq(label) for label in labels]
                targets_length = torch.tensor([len(target) for target in targets])
                targets = torch.cat([torch.tensor(target) for target in targets], dim=0)
                optimizer.zero_grad()
                predictions = model(inputs).log_softmax(2)
                t, n, c = predictions.shape
                predictions_length = torch.full(size=(n,), fill_value=t, dtype=torch.long)

                loss = criterion(predictions, targets, predictions_length, targets_length)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                progress.update()
            mean_loss = sum(losses) / len(losses)
            train_loss.append(mean_loss)
            progress.set_postfix({f'Train loss': mean_loss})

        if scheduler:
            scheduler.step()

        if epoch % 3 == 0:
            with tqdm(total=len(val_dataloader)) as progress:
                losses = []
                cer_score = []
                accuracy = []
                model.eval()
                for inputs, labels in val_dataloader:
                    inputs = inputs.to(device)
                    targets = [encode_seq(label) for label in labels]
                    targets_length = torch.tensor([len(target) for target in targets])
                    targets = torch.cat([torch.tensor(target) for target in targets])

                    with torch.set_grad_enabled(False):
                        predictions = model(inputs).log_softmax(2)
                        t, n, c = predictions.shape
                        predictions_length = torch.full(size=(n,), fill_value=t, dtype=torch.long)
                        loss = criterion(predictions, targets, predictions_length, targets_length)
                        predictions = predictions.cpu()
                        predictions_labels = [decode_seq(prediction) for prediction in predictions.permute(1, 0, 2)]
                        score, acc = cer_accuracy_score(predictions_labels, labels)
                        cer_score.extend(score)
                        accuracy.extend(acc)
                    losses.append(loss.item())
                    progress.update()

                mean_loss = sum(losses) / len(losses)
                mean_cer = sum(cer_score) / len(cer_score)
                mean_acc = sum(accuracy) / len(accuracy)
                val_loss.append(mean_loss)
                cer_scores.append(mean_cer)
                acc_scores.append(mean_acc)
                progress.set_postfix({'Val loss': mean_loss,
                                      'CER': mean_cer,
                                      'accuracy': mean_acc,
                                      })

    return train_loss, val_loss, cer_scores, acc_scores


def validate_func(model, dataloader):
    device = model.device()
    model.eval()
    cer_score = []
    accuracy = []
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        with torch.no_grad():
            predictions = model(inputs).log_softmax(2)
            predictions = predictions.cpu()
            predictions_labels = [decode_seq(prediction) for prediction in predictions.permute(1, 0, 2)]
            score, acc = cer_accuracy_score(predictions_labels, labels)
            cer_score.extend(score)
            accuracy.extend(acc)
    mean_cer = sum(cer_score) / len(cer_score)
    mean_accuracy = sum(accuracy) / len(accuracy)

    print('Mean CER {:.3f} | Mean accuracy {:.3f}'.format(mean_cer, mean_accuracy))

    return mean_cer, mean_accuracy


if __name__ == "__main__":
    pass
