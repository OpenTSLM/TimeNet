"""Test helpers for building HEARTS with all five corpus parents."""

from dataclasses import dataclass
from pathlib import Path

from timenet.composition import BuildContext
from timenet.dataset import TimeFDataset
from timenet.registry import LocalRegistry
from timenet_connectors.datasets.yang_ai_lab.hearts.connector import HeartsConnector
from timenet_connectors.datasets.yang_ai_lab.hearts_cgmacros.connector import HeartsCgmacrosConnector
from timenet_connectors.datasets.yang_ai_lab.hearts_coswara.connector import HeartsCoswaraConnector
from timenet_connectors.datasets.yang_ai_lab.hearts_coughvid.connector import HeartsCoughvidConnector
from timenet_connectors.datasets.yang_ai_lab.hearts_harespod.connector import HeartsHarespodConnector
from timenet_connectors.datasets.yang_ai_lab.hearts_vctk.connector import HeartsVctkConnector


_PARENT_CONNECTORS = (
    HeartsCgmacrosConnector,
    HeartsCoswaraConnector,
    HeartsCoughvidConnector,
    HeartsHarespodConnector,
    HeartsVctkConnector,
)


@dataclass(frozen=True)
class BuiltHearts:
    """A composed synthetic HEARTS dataset and its complete local registry."""

    dataset: TimeFDataset
    registry: LocalRegistry


def build_composed_hearts(root: Path, registry_root: Path) -> BuiltHearts:
    """Build all five parent layers and HEARTS from one synthetic case tree."""
    registry = LocalRegistry(registry_root)
    for connector_type in _PARENT_CONNECTORS:
        connector = connector_type()
        registry.store(connector.convert([root]), values_backend=connector.values_backend)

    connector = HeartsConnector()
    with BuildContext.open(connector.metadata(), registry) as context:
        dataset = connector.convert([root], context)
        dataset.set_dependencies(context.dependency_lock())
    return BuiltHearts(dataset, registry)
