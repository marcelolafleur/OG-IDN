"""
Tests of macro_params.py module.

Only ``g_y_annual`` is refreshed from a live API (World Bank). Every other
macro parameter is held in the packaged JSON, so ``get_macro_params`` never
returns it and a live update cannot clobber it.
"""

import datetime
import json
from importlib.resources import files

import pytest
import requests
from ogcore.parameters import Specifications

from ogidn import macro_params

# Parameters that used to be pulled live and must now never be returned by
# get_macro_params, so a connected run cannot overwrite the documented values.
FROZEN = [
    "alpha_T",
    "alpha_G",
    "initial_debt_ratio",
    "initial_foreign_debt_ratio",
    "zeta_D",
    "gamma",
    "zeta_K",
    "world_int_rate_annual",
    "debt_ratio_ss",
    "r_gov_scale",
    "r_gov_shift",
    "r_gov_DY",
    "r_gov_DY2",
]


class MockResponse:
    """Minimal mock response for requests.get()."""

    def __init__(self, *, json_data=None, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _wb_payload(observations):
    return [
        {
            "page": 1,
            "pages": 1,
            "per_page": "10000",
            "total": len(observations),
        },
        [
            {"date": date, "value": value, "indicator": {"id": "mock"}}
            for date, value in observations
        ],
    ]


def _mock_wb(monkeypatch, payloads):
    def fake_get(url, params=None, headers=None, timeout=None):
        if "worldbank.org" in url:
            code = url.rstrip("/").split("/")[-1]
            return MockResponse(json_data=payloads[code])
        raise AssertionError(f"Unexpected URL requested in test: {url}")

    monkeypatch.setattr(macro_params.requests, "get", fake_get)


def test_fetch_wb_data_annual_success(monkeypatch):
    _mock_wb(
        monkeypatch,
        {
            "NY.GDP.PCAP.KD": _wb_payload(
                [("2024", 100.0), ("2023", 80.0), ("2022", 64.0)]
            )
        },
    )
    data = macro_params._fetch_wb_data(
        {"GDP per capita (constant 2015 US$)": "NY.GDP.PCAP.KD"},
        "IDN",
        2022,
        2024,
        source=2,
    )
    assert list(data.index) == ["2024", "2023", "2022"]
    assert data.loc["2024", "GDP per capita (constant 2015 US$)"] == 100.0


def test_fetch_wb_data_empty_payload_raises_value_error(monkeypatch):
    _mock_wb(
        monkeypatch,
        {"NY.GDP.PCAP.KD": [{"page": 1, "pages": 1, "total": 0}, []]},
    )
    with pytest.raises(ValueError, match="Empty or malformed World Bank"):
        macro_params._fetch_wb_data(
            {"GDP per capita (constant 2015 US$)": "NY.GDP.PCAP.KD"},
            "IDN",
            2022,
            2024,
            source=2,
        )


def test_fetch_wb_data_rejects_unsupported_source():
    with pytest.raises(ValueError, match="Unsupported World Bank source"):
        macro_params._fetch_wb_data(
            {"GDP per capita (constant 2015 US$)": "NY.GDP.PCAP.KD"},
            "IDN",
            2022,
            2024,
            source=99,
        )


def test_get_macro_params_update_from_api_false_returns_empty_dict():
    assert macro_params.get_macro_params(update_from_api=False) == {}


def test_get_macro_params_update_from_api_true_returns_only_g_y(monkeypatch):
    # 25%/yr GDP-per-capita growth inside the 2000-2019 window.
    _mock_wb(
        monkeypatch,
        {
            "NY.GDP.PCAP.KD": _wb_payload(
                [("2000", 100.0), ("2001", 125.0), ("2002", 156.25)]
            )
        },
    )
    result = macro_params.get_macro_params(
        data_end_date=datetime.datetime(2024, 12, 31),
        update_from_api=True,
    )
    assert list(result.keys()) == ["g_y_annual"]
    assert result["g_y_annual"] == pytest.approx(0.25)
    # frozen params must never be overwritten from an API
    for k in FROZEN:
        assert k not in result


def test_g_y_uses_pre_pandemic_window(monkeypatch):
    # Growth is 10%/yr through 2019 then a COVID-era swing; the window must
    # exclude 2020+ so the swing does not enter the long-run target.
    _mock_wb(
        monkeypatch,
        {
            "NY.GDP.PCAP.KD": _wb_payload(
                [
                    ("2018", 100.0),
                    ("2019", 110.0),
                    ("2020", 60.0),  # COVID contraction, must be excluded
                    ("2021", 130.0),  # rebound, must be excluded
                ]
            )
        },
    )
    result = macro_params.get_macro_params(update_from_api=True)
    # Only 2018->2019 pct_change (0.10) falls inside [2000, 2019].
    assert result["g_y_annual"] == pytest.approx(0.10)


def test_get_macro_params_wb_failure_keeps_packaged_value(monkeypatch):
    def offline(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(macro_params.requests, "get", offline)
    # the World Bank pull failing must not raise; g_y stays packaged
    assert macro_params.get_macro_params(update_from_api=True) == {}


@pytest.mark.parametrize(
    "defaults_file",
    [
        "ogidn_default_parameters.json",
        "ogidn_multisector_default_parameters.json",
    ],
)
def test_packaged_defaults_load_into_specifications(defaults_file):
    p = Specifications()
    with files("ogidn").joinpath(defaults_file).open("r") as file:
        defaults = json.load(file)

    p.update_specifications(defaults)
