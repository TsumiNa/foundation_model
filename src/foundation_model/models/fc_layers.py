import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        normalization=True,
        activation: None | nn.Module = nn.LeakyReLU(0.1),
    ):
        """
        Parameters
        ----------
        n_in: int
            Size of each input sample.
        n_out: int
            Size of each output sample
        """
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.normal = nn.BatchNorm1d(n_out) if normalization else None
        self.activation = nn.LeakyReLU(0.1)
        self.activation = activation

    def forward(self, x):
        _out = self.layer(x)
        if self.normal:
            _out = self.normal(_out)
        if self.activation:
            _out = self.activation(_out)

        return _out


class ResidualBlock(nn.Module):
    def __init__(
        self,
        n_features,
        normalization=True,
        n_layers=2,
        layer_activation: None | nn.Module = nn.LeakyReLU(0.1),
        output_active: None | nn.Module = nn.LeakyReLU(0.1),
    ):
        super().__init__()
        self.layers = nn.Sequential(
            *[
                LinearLayer(
                    n_features,
                    n_features,
                    normalization=normalization,
                    activation=layer_activation,
                )
                if i != n_layers - 1
                else LinearLayer(
                    n_features,
                    n_features,
                    normalization=normalization,
                    activation=None,
                )
                for i in range(n_layers)
            ]
        )
        self.output_active = output_active

    def forward(self, x):
        y = self.layers(x)
        y += x
        if self.output_active:
            return self.output_active(y)
        return y


class LinearBlock(nn.Module):
    """
    Block of multiple LinearLayers with optional normalization and residual connections.

    dim_output_layer : int | None, optional
        If given, an extra ``LinearLayer(shared_layer_dims[-1], dim_output_layer)`` is
        appended automatically, mirroring the pattern used at call‑sites.
    """

    def __init__(
        self,
        shared_layer_dims: list[int],
        normalization=True,
        residual=False,
        layer_activation: None | nn.Module = nn.LeakyReLU(0.1),
        output_active: None | nn.Module = nn.LeakyReLU(0.1),
        *,
        dim_output_layer: int | None = None,
    ):
        super().__init__()
        counter = len(shared_layer_dims) - 1
        if counter < 1:
            raise ValueError("shared_layer_dims must have at least 2 elements")

        if residual:
            self.layers = nn.Sequential(
                *[
                    # Add residual block after each layer
                    nn.Sequential(
                        LinearLayer(
                            shared_layer_dims[i],
                            shared_layer_dims[i + 1],
                            normalization=normalization,
                            activation=layer_activation,
                        ),
                        ResidualBlock(
                            shared_layer_dims[i + 1],
                            normalization=normalization,
                            layer_activation=layer_activation,
                            output_active=None,
                        ),
                    )
                    if i == counter - 1 and output_active is None
                    else nn.Sequential(
                        LinearLayer(
                            shared_layer_dims[i],
                            shared_layer_dims[i + 1],
                            normalization=normalization,
                            activation=layer_activation,
                        ),
                        ResidualBlock(
                            shared_layer_dims[i + 1],
                            normalization=normalization,
                            layer_activation=layer_activation,
                            output_active=output_active,
                        ),
                    )
                    for i in range(counter)
                ]
            )
        else:
            self.layers = nn.Sequential(
                *[
                    LinearLayer(
                        shared_layer_dims[i],
                        shared_layer_dims[i + 1],
                        normalization=normalization,
                        activation=None,
                    )
                    if i == counter - 1 and output_active is None
                    else LinearLayer(
                        shared_layer_dims[i],
                        shared_layer_dims[i + 1],
                        normalization=normalization,
                        activation=layer_activation,
                    )
                    for i in range(counter)
                ]
            )

        # Optionally add a final output LinearLayer for convenience
        if dim_output_layer is not None:
            self.layers = nn.Sequential(
                self.layers,
                LinearLayer(
                    shared_layer_dims[-1],
                    dim_output_layer,
                    normalization=False,
                    activation=None,
                ),
            )

    def forward(self, x):
        return self.layers(x)
