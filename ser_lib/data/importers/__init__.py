"""Importer 子包：注册全部 importer 到默认注册表。"""

from __future__ import annotations

from ser_lib.data.importers.base import DatasetImporter, ImportIssue, ImportPreview
from ser_lib.data.importers.casia import CasiaImportConfig, CasiaImporter
from ser_lib.data.importers.csv_importer import CsvImportConfig, CsvImporter
from ser_lib.data.importers.folder import FolderImportConfig, FolderImporter
from ser_lib.data.importers.jsonl_importer import (
    JsonlImportConfig,
    JsonlImporter,
    normalize_raw_record,
    normalize_raw_records,
)
from ser_lib.data.importers.ravdess import RavdessImportConfig, RavdessImporter
from ser_lib.data.registry import default_registry

__all__ = [
    "DatasetImporter",
    "ImportIssue",
    "ImportPreview",
    "CasiaImporter",
    "CasiaImportConfig",
    "CsvImporter",
    "CsvImportConfig",
    "FolderImporter",
    "FolderImportConfig",
    "JsonlImporter",
    "JsonlImportConfig",
    "RavdessImporter",
    "RavdessImportConfig",
    "normalize_raw_record",
    "normalize_raw_records",
    "register_importers",
]


def register_importers(registry=default_registry) -> None:
    """把全部 importer 注册到注册表。"""
    for cls in (FolderImporter, CsvImporter, JsonlImporter, CasiaImporter, RavdessImporter):
        registry.register(
            namespace="importer",
            name=cls.descriptor.id,
            factory=cls,
            config_model=None,
            descriptor=cls.descriptor,
        )
