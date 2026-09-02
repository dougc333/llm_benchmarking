from enginebench.providers.cli_providers import (
    DigitalOceanProvider,
    HyperbolicProvider,
    VastProvider,
)

PROVIDERS = {
    "vast": VastProvider(),
    "digitalocean": DigitalOceanProvider(),
    "hyperbolic": HyperbolicProvider(),
}

__all__ = ["PROVIDERS", "DigitalOceanProvider", "HyperbolicProvider", "VastProvider"]
