import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        #print('accuracy: ', accuracies.float().mean().item())
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                #print(accuracy_in_bin)
                avg_confidence_in_bin = confidences[in_bin].mean()
                #print(avg_confidence_in_bin)
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

class ECELoss_SM(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss_SM, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        #softmaxes = F.softmax(logits, dim=1)
        softmaxes = logits
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        #print('accuracy: ', accuracies.float().mean().item())
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                #print(accuracy_in_bin)
                avg_confidence_in_bin = confidences[in_bin].mean()
                #print(avg_confidence_in_bin)
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class ECELoss_Top1(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss_Top1, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, softmaxes, confidences, labels):
        _, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        #print('accuracy: ', accuracies.float().mean().item())
        ece = torch.zeros(1, device=softmaxes.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                #print(accuracy_in_bin)
                avg_confidence_in_bin = confidences[in_bin].mean()
                #print(avg_confidence_in_bin)
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece







def AdaECE(preds, targets, n_bins=15, threshold=0, **args):

    preds = F.softmax(preds, dim=1)
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
   
    confidences, predictions = np.max(preds, 1), np.argmax(preds, 1)
    n_objects, n_classes = preds.shape
    
    ece = 0.0
    #################################################################################################################
    targets_sorted = targets[confidences.argsort()]
    predictions_sorted = predictions[confidences.argsort()]
    confidence_sorted = np.sort(confidences)
    
    targets_sorted = targets_sorted[confidence_sorted > threshold]
    predictions_sorted = predictions_sorted[confidence_sorted > threshold]
    confidence_sorted = confidence_sorted[confidence_sorted > threshold]
    
    bin_size = len(confidence_sorted) // n_bins

    for bin_i in range(n_bins):
        bin_start_ind = bin_i * bin_size
        if bin_i < n_bins-1:
            bin_end_ind = bin_start_ind + bin_size
        else:
            bin_end_ind = len(targets_sorted)
            bin_size = bin_end_ind - bin_start_ind  # extend last bin until the end of prediction array
        bin_acc = (targets_sorted[bin_start_ind : bin_end_ind] == predictions_sorted[bin_start_ind : bin_end_ind])
        bin_conf = confidence_sorted[bin_start_ind : bin_end_ind]
        avg_confidence_in_bin = np.mean(bin_conf)
        avg_accuracy_in_bin = np.mean(bin_acc)
        delta = np.abs(avg_confidence_in_bin - avg_accuracy_in_bin)
#             print(f'bin size {bin_size}, bin conf {avg_confidence_in_bin}, bin acc {avg_accuracy_in_bin}')
        ece += delta * bin_size / (n_objects)
    #################################################################################################################

    return ece


def AdaECE_SM(preds, targets, n_bins=15, threshold=0, **args):
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
   
    confidences, predictions = np.max(preds, 1), np.argmax(preds, 1)
    n_objects, n_classes = preds.shape
    
    ece = 0.0
    #################################################################################################################
    targets_sorted = targets[confidences.argsort()]
    predictions_sorted = predictions[confidences.argsort()]
    confidence_sorted = np.sort(confidences)
    
    targets_sorted = targets_sorted[confidence_sorted > threshold]
    predictions_sorted = predictions_sorted[confidence_sorted > threshold]
    confidence_sorted = confidence_sorted[confidence_sorted > threshold]
    
    bin_size = len(confidence_sorted) // n_bins

    for bin_i in range(n_bins):
        bin_start_ind = bin_i * bin_size
        if bin_i < n_bins-1:
            bin_end_ind = bin_start_ind + bin_size
        else:
            bin_end_ind = len(targets_sorted)
            bin_size = bin_end_ind - bin_start_ind  # extend last bin until the end of prediction array
        bin_acc = (targets_sorted[bin_start_ind : bin_end_ind] == predictions_sorted[bin_start_ind : bin_end_ind])
        bin_conf = confidence_sorted[bin_start_ind : bin_end_ind]
        avg_confidence_in_bin = np.mean(bin_conf)
        avg_accuracy_in_bin = np.mean(bin_acc)
        delta = np.abs(avg_confidence_in_bin - avg_accuracy_in_bin)
#             print(f'bin size {bin_size}, bin conf {avg_confidence_in_bin}, bin acc {avg_accuracy_in_bin}')
        ece += delta * bin_size / (n_objects)
    #################################################################################################################

    return ece
