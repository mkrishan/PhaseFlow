"""Discovery and validation for separately distributed PhaseFlow plugins."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Any, List, Mapping, Tuple

from .model import PLUGIN_API_VERSION, Capability


@dataclass(frozen=True)
class PluginManifest:
    name: str
    version: str
    api_version: str = PLUGIN_API_VERSION
    capabilities: Tuple[Capability, ...] = ()
    description: str = ""
    citation: str = ""
    license: str = ""


@dataclass(frozen=True)
class LoadedPlugin:
    manifest: PluginManifest
    component: Any


def validate_manifest(manifest: PluginManifest) -> None:
    if manifest.api_version != PLUGIN_API_VERSION:
        raise ValueError(
            f"plugin {manifest.name!r} targets API {manifest.api_version}; "
            f"PhaseFlow expects {PLUGIN_API_VERSION}"
        )
    if not manifest.name.strip() or not manifest.version.strip():
        raise ValueError("plugin name and version must be non-empty")


def discover_plugins(group: str = "phaseflow.plugins") -> List[LoadedPlugin]:
    """Load plugins registered through Python package entry points."""

    discovered = []
    entry_points = metadata.entry_points()
    selected = entry_points.select(group=group) if hasattr(entry_points, "select") else entry_points.get(group, [])
    for entry_point in selected:
        component = entry_point.load()
        manifest = getattr(component, "manifest", None)
        if not isinstance(manifest, PluginManifest):
            raise TypeError(f"plugin {entry_point.name!r} does not expose a PluginManifest")
        validate_manifest(manifest)
        discovered.append(LoadedPlugin(manifest=manifest, component=component))
    return discovered


def plugin_inventory() -> Mapping[str, str]:
    return {plugin.manifest.name: plugin.manifest.version for plugin in discover_plugins()}

