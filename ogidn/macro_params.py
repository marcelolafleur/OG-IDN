"""
Macro parameters for the OG-IDN model.

Most OG-IDN macro parameters are documented, point-in-time Indonesian values
held in the packaged default-parameter JSON. Only a parameter whose documented
source is genuinely a live API is refreshed here when ``update_from_api=True``;
the rest stay in the JSON so a live update cannot clobber a value we source
elsewhere.

Previously this module also pulled ``initial_debt_ratio``,
``initial_foreign_debt_ratio``, ``zeta_D`` (World Bank QPSD), ``gamma``
(ILOSTAT), and ``alpha_T``/``alpha_G`` (IMF GFS) on every connected run. Those
pulls overwrote the documented baseline with whatever the API happened to
return -- from series whose definitions do not match the model parameter and
which drift run to run -- so the offline default and the connected run
diverged. They are now frozen at documented values (see the calibration
docs and the
per-parameter notes on ``get_macro_params``); only ``g_y_annual`` stays live.
Demographics and the earnings profile stay live (UN) in ``ogidn.calibrate``.
"""

import datetime

import pandas as pd
import requests

# GDP-per-capita growth window for g_y_annual. The 2000 start is a
# structural-break boundary: it excludes the 1997-98 Asian Financial Crisis and
# the post-Suharto Reformasi transition. The 2019 end is pre-pandemic: it keeps
# the 2020 COVID contraction and its rebound out of the long-run productivity
# target. See docs/book/content/calibration/macro.md.
GDP_GROWTH_START_YEAR = 2000
GDP_GROWTH_END_YEAR = 2019


def _fetch_wb_data(indicators, country_iso, start_year, end_year, source):
    """
    Fetch a set of World Bank indicators and return a single DataFrame.

    Args:
        indicators (dict): mapping of human-readable labels to indicator codes
        country_iso (str): ISO country code
        start_year (int): first year to request
        end_year (int): last year to request
        source (int): World Bank source ID

    Returns:
        pandas.DataFrame: DataFrame indexed by year/quarter label
    """
    if source == 2:
        date_range = f"{start_year}:{end_year}"
    elif source == 20:
        date_range = f"{start_year}Q1:{end_year}Q4"
    else:
        raise ValueError(f"Unsupported World Bank source: {source}")

    data_frames = []
    for label, indicator_code in indicators.items():
        response = requests.get(
            (
                "https://api.worldbank.org/v2/country/"
                f"{country_iso}/indicator/{indicator_code}"
            ),
            params={
                "date": date_range,
                "source": source,
                "format": "json",
                "per_page": 10000,
            },
            timeout=30,
        )
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as exc:
            raise ValueError(
                f"Malformed World Bank response for {indicator_code}"
            ) from exc

        if (
            not isinstance(payload, list)
            or len(payload) < 2
            or not isinstance(payload[1], list)
            or not payload[1]
        ):
            raise ValueError(
                f"Empty or malformed World Bank response for {indicator_code}"
            )

        series_data = {}
        for row in payload[1]:
            date = row.get("date")
            if date is None:
                continue
            series_data[date] = row.get("value")

        if not series_data:
            raise ValueError(
                f"No dated observations in World Bank response for "
                f"{indicator_code}"
            )

        series = pd.Series(series_data, name=label)
        series = pd.to_numeric(series, errors="coerce")
        data_frames.append(series.to_frame())

    data = pd.concat(data_frames, axis=1)
    data.index.name = "year"
    return data.sort_index(ascending=False)


def _annual_index(data):
    """
    Convert a World Bank annual response index to integer years.
    """
    annual_data = data.copy()
    annual_data.index = pd.to_numeric(annual_data.index, errors="coerce")
    annual_data = annual_data.loc[annual_data.index.notna()]
    annual_data.index = annual_data.index.astype(int)
    return annual_data.sort_index()


def get_macro_params(
    data_start_date=datetime.datetime(1947, 1, 1),
    data_end_date=datetime.datetime(2024, 12, 31),
    country_iso="IDN",
    update_from_api=False,
):
    """
    Return macro-parameter overrides for the OG-IDN calibration.

    Only ``g_y_annual`` is refreshed from a live API -- World Bank
    GDP-per-capita growth over the pre-pandemic 2000-2019 window, its
    documented source. Every other macro parameter is held in the packaged
    JSON and is NOT pulled here, so a live update cannot clobber a value
    sourced elsewhere:

      * alpha_T, alpha_G           -- IMF Government Finance Statistics,
                                      frozen at the documented values
      * initial_debt_ratio         -- World Bank QPSD / Kemenkeu general
                                      government debt-to-GDP
      * initial_foreign_debt_ratio, zeta_D -- foreign-held share of Indonesian
                                      government debt (Kemenkeu / World Bank
                                      QPSD external creditors)
      * gamma                      -- ILOSTAT-based capital share of income
      * zeta_K                     -- normalized Chinn-Ito capital-account
                                      openness index for Indonesia
      * world_int_rate_annual      -- global risk-free rate plus the Indonesian
                                      sovereign risk premium
      * debt_ratio_ss              -- Indonesian medium-term debt anchor
      * r_gov_scale, r_gov_shift,
        r_gov_DY, r_gov_DY2        -- Li et al. (2021) sovereign-vs-corporate
                                      yield mapping plus the centered
                                      debt-elastic premium

    Returns:
        dict: macro-parameter overlay (only ``g_y_annual`` when
        ``update_from_api`` and the World Bank call succeeds)
    """
    macro_parameters = {}
    if update_from_api:
        try:
            wb_data = _annual_index(
                _fetch_wb_data(
                    {"GDP per capita (constant 2015 US$)": "NY.GDP.PCAP.KD"},
                    country_iso,
                    data_start_date.year,
                    data_end_date.year,
                    source=2,
                )
            )
            # Pre-pandemic window avoids COVID-era volatility distorting the
            # steady-state productivity target (docs/calibration/macro.md).
            macro_parameters["g_y_annual"] = (
                wb_data["GDP per capita (constant 2015 US$)"]
                .loc[GDP_GROWTH_START_YEAR:GDP_GROWTH_END_YEAR]
                .pct_change()
                .mean()
            )
            print(
                "g_y_annual updated from World Bank API: "
                f"{macro_parameters['g_y_annual']}"
            )
        except Exception:
            print(
                "Failed to retrieve g_y_annual from World Bank; "
                "keeping packaged value"
            )
    else:
        print("Not updating macro params from World Bank API")
    return macro_parameters
