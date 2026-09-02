from __future__ import annotations

import json
from typing import Any

from enginebench.providers.base import Provider, required


class VastProvider(Provider):
    """Official Vast CLI adapter. Creation accepts a concrete marketplace offer."""

    name = "vast"
    credential_env = "VAST_API_KEY"

    def discover(self, options: dict[str, Any]) -> tuple[str, ...]:
        gpu = options.get("gpu_name", "H100 SXM")
        count = int(options.get("gpu_count", 1))
        reliability = float(options.get("reliability", 0.99))
        query = (
            f"gpu_name={gpu} num_gpus={count} reliability>={reliability} "
            "verified=true rentable=true"
        )
        return ("vastai", "search", "offers", query, "--order", "dph_total", "--raw")

    def create(self, options: dict[str, Any]) -> tuple[str, ...]:
        offer_id = str(required(options, "offer_id"))
        image = str(required(options, "image"))
        return (
            "vastai",
            "create",
            "instance",
            offer_id,
            "--image",
            image,
            "--disk",
            str(options.get("disk_gb", 200)),
            "--ssh",
            "--direct",
            "--label",
            str(options.get("label", "enginebench-h100")),
            "--raw",
        )

    def status(self, instance_id: str) -> tuple[str, ...]:
        return ("vastai", "show", "instance", instance_id, "--raw")

    def destroy(self, instance_id: str) -> tuple[str, ...]:
        return ("vastai", "destroy", "instance", instance_id, "--raw")


class DigitalOceanProvider(Provider):
    """Official doctl adapter for GPU Droplets."""

    name = "digitalocean"
    credential_env = "DIGITALOCEAN_TOKEN"

    def discover(self, options: dict[str, Any]) -> tuple[str, ...]:
        return ("doctl", "compute", "size", "list", "--output", "json")

    def create(self, options: dict[str, Any]) -> tuple[str, ...]:
        command = [
            "doctl",
            "compute",
            "droplet",
            "create",
            str(required(options, "name")),
            "--region",
            str(required(options, "region")),
            "--size",
            str(required(options, "size")),
            "--image",
            str(required(options, "image")),
            "--monitoring",
            "--wait",
            "--output",
            "json",
        ]
        ssh_keys = options.get("ssh_key_ids", [])
        if ssh_keys:
            command.extend(["--ssh-keys", ",".join(str(value) for value in ssh_keys)])
        tags = options.get("tags", ["enginebench", "ephemeral"])
        if tags:
            command.extend(["--tag-names", ",".join(str(value) for value in tags)])
        return tuple(command)

    def status(self, instance_id: str) -> tuple[str, ...]:
        return ("doctl", "compute", "droplet", "get", instance_id, "--output", "json")

    def destroy(self, instance_id: str) -> tuple[str, ...]:
        return ("doctl", "compute", "droplet", "delete", instance_id, "--force")


class HyperbolicProvider(Provider):
    """Official Hyperbolic CLI adapter; the provider's CLI handles API evolution."""

    name = "hyperbolic"
    credential_env = "HYPERBOLIC_API_KEY"

    def discover(self, options: dict[str, Any]) -> tuple[str, ...]:
        return ("hyperbolic", "ondemand", "--json")

    def create(self, options: dict[str, Any]) -> tuple[str, ...]:
        command = [
            "hyperbolic",
            "rent",
            "ondemand",
            "--instance-type",
            str(options.get("instance_type", "virtual-machine")),
            "--gpu-count",
            str(options.get("gpu_count", 1)),
        ]
        if options.get("network_type"):
            command.extend(["--network-type", str(options["network_type"])])
        command.append("--json")
        return tuple(command)

    def status(self, instance_id: str) -> tuple[str, ...]:
        # The CLI lists all instances; consumers select instance_id from JSON.
        return ("hyperbolic", "instances", "--json")

    def destroy(self, instance_id: str) -> tuple[str, ...]:
        return ("hyperbolic", "terminate", instance_id, "--json")


def options_from_json(raw: str) -> dict[str, Any]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("provider options must be a JSON object")
    return payload
