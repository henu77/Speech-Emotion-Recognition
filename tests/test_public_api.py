from __future__ import annotations

import ser_lib


def test_top_level_public_api_is_resolvable_and_unique():
    assert len(ser_lib.__all__) == len(set(ser_lib.__all__))
    for name in ser_lib.__all__:
        assert hasattr(ser_lib, name), name
    assert ser_lib.__version__ == "0.2.0"


def test_removed_service_is_not_part_of_public_api():
    assert all("service" not in name.lower() for name in ser_lib.__all__)
