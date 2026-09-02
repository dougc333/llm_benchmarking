import pytest

from enginebench.providers import PROVIDERS


@pytest.mark.parametrize("provider", ["vast", "digitalocean", "hyperbolic"])
def test_provider_has_lifecycle_commands(provider: str) -> None:
    adapter = PROVIDERS[provider]

    assert adapter.credential_env
    assert adapter.status("instance-id")
    assert adapter.destroy("instance-id")


def test_digitalocean_create_is_tagged_and_monitored() -> None:
    command = PROVIDERS["digitalocean"].create(
        {
            "name": "bench",
            "region": "atl1",
            "size": "gpu-h100",
            "image": "gpu-image",
            "ssh_key_ids": [123],
        }
    )
    rendered = " ".join(command)

    assert "--monitoring" in command
    assert "--wait" in command
    assert "--tag-names enginebench,ephemeral" in rendered


def test_vast_requires_concrete_offer() -> None:
    with pytest.raises(ValueError, match="offer_id"):
        PROVIDERS["vast"].create({"image": "nvidia/cuda"})
