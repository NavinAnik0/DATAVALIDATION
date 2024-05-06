from torch import nn, mean, cat
import re
from nfnets import NFBlockCustom, WSConv2D, activations_dict
from torch.profiler import profile, record_function, ProfilerActivity


nfnet_params = {
    'F0': {
        'width': [1536, 1536, 1536], 'depth': [1, 2, 3],
        'train_imsize': 192, 'test_imsize': 256,
        'RA_level': '405', 'drop_rate': 0.4},

    'F1': {
        'width': [256, 512, 1536, 1536], 'depth': [2, 4, 12, 6],
        'train_imsize': 224, 'test_imsize': 320,
        'RA_level': '410', 'drop_rate': 0.3},
    'F2': {
        'width': [256, 512, 1536, 1536], 'depth': [3, 6, 18, 9],
        'train_imsize': 256, 'test_imsize': 352,
        'RA_level': '410', 'drop_rate': 0.4},
    'F3': {
        'width': [256, 512, 1536, 1536], 'depth': [4, 8, 24, 12],
        'train_imsize': 320, 'test_imsize': 416,
        'RA_level': '415', 'drop_rate': 0.4},
    'F4': {
        'width': [256, 512, 1536, 1536], 'depth': [5, 10, 30, 15],
        'train_imsize': 384, 'test_imsize': 512,
        'RA_level': '415', 'drop_rate': 0.5},
    'F5': {
        'width': [256, 512, 1536, 1536], 'depth': [6, 12, 36, 18],
        'train_imsize': 416, 'test_imsize': 544,
        'RA_level': '415', 'drop_rate': 0.5},
    'F6': {
        'width': [256, 512, 1536, 1536], 'depth': [7, 14, 42, 21],
        'train_imsize': 448, 'test_imsize': 576,
        'RA_level': '415', 'drop_rate': 0.5},
    'F7': {
        'width': [256, 512, 1536, 1536], 'depth': [8, 16, 48, 24],
        'train_imsize': 480, 'test_imsize': 608,
        'RA_level': '415', 'drop_rate': 0.5},
}


class MulticlassClassifier(nn.Module):
    def __init__(self, pretrained_model):
        super(MulticlassClassifier, self).__init__()
        self.pretrained = pretrained_model
        self.new_fc = nn.Sequential(
            nn.Linear(in_features=1000, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=3)
        )

    def exclude_from_weight_decay(self, name: str) -> bool:
        # Regex to find layer names like
        # "stem.6.bias", "stem.6.gain", "body.0.skip_gain",
        # "body.0.conv0.bias", "body.0.conv0.gain"
        regex = re.compile('stem.*(bias|gain)|conv.*(bias|gain)|skip_gain')
        return len(regex.findall(name)) > 0

    def exclude_from_clipping(self, name: str) -> bool:
        # Last layer should not be clipped
        return name.startswith('linear')

    def forward(self, x):
        return self.new_fc(self.pretrained(x))


class CustomNfNet(nn.Module):
    def __init__(self, stem, body, stochdepth_rate, num_classes=1000, alpha: float = 0.2, se_ratio: float = 0.5,
                 activation: str = 'gelu'):
        super(CustomNfNet, self).__init__()
        self.stem = stem
        self.body = body
        blocks = []
        expected_std = 1.0
        block_params = nfnet_params['F0']
        in_channels = self.body[5].out_channels
        self.drop_rate = block_params['drop_rate']
        self.num_classes = num_classes
        self.activation = activations_dict[activation]

        block_args = zip(
            block_params['width'],
            block_params['depth'],
            [0.5] * 3,  # bottleneck pattern
            [64] * 3,  # group pattern. Original groups [128] * 4
            [1, 2, 2]  # stride pattern
        )
        index = 6
        for (block_width, stage_depth, expand_ratio, group_size, stride) in block_args:
            for block_index in range(stage_depth):
                beta = 1. / expected_std
                block_sd_rate = stochdepth_rate * index / 12
                out_channels = block_width
                blocks.append(NFBlockCustom(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride if block_index == 0 else 1,
                    # stride=stride,
                    alpha=alpha,
                    beta=beta,
                    se_ratio=se_ratio,
                    group_size=group_size,
                    stochdepth_rate=block_sd_rate,
                    activation=activation))

                in_channels = out_channels
                index += 1

                if block_index == 0:
                    expected_std = 1.0

                expected_std = (expected_std ** 2 + alpha ** 2) ** 0.5

        self.bottleneck = nn.Sequential(*blocks)
        final_conv_channels = 2 * in_channels
        self.final_conv = WSConv2D(in_channels=out_channels, out_channels=final_conv_channels, kernel_size=1)
        self.pool = nn.AvgPool2d(1)

        if self.drop_rate > 0.:
            self.dropout = nn.Dropout(self.drop_rate)

        self.linear1 = nn.Linear(final_conv_channels, self.num_classes)
        self.linear2 = nn.Linear(1000, 100)
        self.hole = nn.Linear(100, 2)
        self.growth = nn.Linear(100, 3)
        self.relu = activations_dict['relu']
        nn.init.normal_(self.linear1.weight, 0, 0.01)
        nn.init.normal_(self.linear2.weight, 0, 0.01)
        nn.init.normal_(self.hole.weight, 0, 0.01)
        nn.init.normal_(self.growth.weight, 0, 0.01)

    def exclude_from_weight_decay(self, name: str) -> bool:
        # Regex to find layer names like
        # "stem.6.bias", "stem.6.gain", "body.0.skip_gain",
        # "body.0.conv0.bias", "body.0.conv0.gain"
        regex = re.compile('stem.*(bias|gain)|conv.*(bias|gain)|skip_gain')
        return len(regex.findall(name)) > 0

    def exclude_from_clipping(self, name: str) -> bool:
        # Last layer should not be clipped
        return name.startswith('linear')

    def forward(self, x):
        out = self.stem(x)
        out = cat((out, out[:, :, 0:out.shape[3]-out.shape[2], :]), 2)
        # out = self.up(out)
        out = self.body(out)

        out = self.bottleneck(out)

        out = self.activation(self.final_conv(out))
        pool = mean(out, dim=(2, 3))

        if self.training and self.drop_rate > 0.:
            pool = self.dropout(pool)

        return {'hole': self.hole(self.relu(self.linear2(self.linear1(pool)))),
                'growth': self.growth(self.relu(self.linear2(self.linear1(pool))))}
