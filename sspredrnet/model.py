"""Original three-block SSPredRNet, with two component views per puzzle."""
import torch
from torch import nn
from torch.nn import functional as F

from .layers import ConvNormAct, ResBlock


class PredictiveBlock(nn.Module):
    def __init__(self, channels, downsample, *, last, dropout):
        super().__init__()
        self.last = last
        self.pconv = ConvNormAct(channels, channels, (5, 1))
        if not last:
            self.conv1 = ConvNormAct(channels, channels * 4, 3, 1)
            self.conv2 = ConvNormAct(channels * 4, channels, 3, 1)
            self.drop = nn.Dropout(dropout)
            self.downsample = downsample

    def forward(self, x):
        contexts, choices = x[:, :, :-1], x[:, :, -1:]
        predictions = self.pconv(contexts)
        residual = F.relu(choices) - predictions
        choices = F.relu(choices).mean(dim=[2, 3])
        predictions = predictions.mean(dim=[2, 3])
        error = F.normalize(choices, dim=1) - F.normalize(predictions, dim=1)
        distance = error.pow(2).sum(dim=1).sqrt()
        if self.last:
            return None, distance
        out = torch.cat((contexts, residual), dim=2)
        out = self.drop(self.conv2(self.conv1(out)))
        return out + self.downsample(x), distance


class Reasoner(nn.Module):
    """RAVEN-only original encoder and three predictive reasoning blocks."""
    def __init__(self):
        super().__init__()
        previous = 1
        for index, channels in enumerate((32, 64, 96, 128)):
            downsample = nn.Sequential(
                nn.AvgPool2d(2, stride=2),
                ConvNormAct(previous, channels, 1, activate=False),
            )
            setattr(self, f'res{index}', ResBlock(
                previous, channels, downsample, stride=2, dropout=.1))
            previous = channels
        self.channel_reducer = ConvNormAct(128, 32, 1, activate=False)
        for index in range(3):
            # Preserve official initialization order, including the final
            # block's unregistered identity draw. No unused weights are kept.
            downsample = ConvNormAct(32, 32, 1, activate=False)
            setattr(self, f'prb{index}', PredictiveBlock(
                32, downsample, last=index == 2, dropout=.1))

    def _features(self, images):
        batch, panels, height, width = images.shape
        x = images.reshape(batch * panels, 1, height, width)
        for index in range(4):
            x = getattr(self, f'res{index}')(x)
        x = self.channel_reducer(x)
        return x.reshape(batch, panels, x.shape[1], x.shape[2] * x.shape[3])

    def _errors(self, matrices):
        batch, choices, _, channels, locations = matrices.shape
        x = matrices.reshape(-1, 6, channels, locations).permute(0, 2, 1, 3)
        errors = []
        for index in range(3):
            x, distance = getattr(self, f'prb{index}')(x)
            if index > 0:  # The original loss supervises blocks two and three.
                errors.append(distance.view(batch, choices))
        return errors

    def forward(self, images):
        features = self._features(images)
        contexts, choices = features[:, :8], features[:, 8:]
        if self.training:
            matrices = contexts[:, :6].clone().unsqueeze(1).repeat(1, 9, 1, 1, 1)
            matrices[:, 1:9, -1] = choices
            return self._errors(matrices)
        errors = []
        for indices in ([0, 1, 2, 6, 7], [3, 4, 5, 6, 7]):
            support = contexts[:, indices]
            matrices = torch.stack([
                torch.cat((support, choices[:, i].unsqueeze(1)), dim=1)
                for i in range(8)
            ], dim=1)
            errors.append(self._errors(matrices))
        return [a + b for a, b in zip(*errors)]


class SSPredRNet(nn.Module):
    def __init__(self, margin=.7):
        super().__init__()
        self.reasoner = Reasoner()
        self.margin = margin

    def forward(self, views):
        if views.ndim != 5 or tuple(views.shape[1:]) != (2, 16, 80, 80):
            raise ValueError('expected B x 2 x 16 x 80 x 80 normalized views')
        batch = views.shape[0]
        flat = views.flatten(0, 1)
        errors = self.reasoner(flat)
        if not self.training:
            return errors[-1].reshape(batch, 2, 8).mean(1)
        identical = (flat[:, 8:] == flat[:, 5:6]).flatten(2).all(2)
        return [torch.cat((error[:, :1], error[:, 1:].masked_fill(
            identical, self.margin)), dim=1) for error in errors]


def prediction_loss(errors, margin=.7):
    """Original summed margin loss, averaged over two views and two PRBs."""
    losses = []
    for distances in errors:
        labels = torch.zeros_like(distances)
        labels[:, 0] = 1  # Known sixth panel; never the puzzle answer index.
        positive = labels * distances.pow(2)
        negative = (1 - labels) * (margin - distances).clamp(min=0).pow(2)
        losses.append(torch.sum(positive + negative) * .5 / 2)
    return sum(losses) / len(losses)
