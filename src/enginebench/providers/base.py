from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Provider(ABC):
    name: str
    credential_env: str

    @abstractmethod
    def discover(self, options: dict[str, Any]) -> tuple[str, ...]: ...

    @abstractmethod
    def create(self, options: dict[str, Any]) -> tuple[str, ...]: ...

    @abstractmethod
    def status(self, instance_id: str) -> tuple[str, ...]: ...

    @abstractmethod
    def destroy(self, instance_id: str) -> tuple[str, ...]: ...

    def ssh(self, instance_id: str, options: dict[str, Any]) -> tuple[str, ...]:
        raise NotImplementedError(f"{self.name} resolves SSH details from its status response")


def required(options: dict[str, Any], key: str) -> Any:
    value = options.get(key)
    if value in (None, "", [], {}):
        raise ValueError(f"provider option '{key}' is required")
    return value
