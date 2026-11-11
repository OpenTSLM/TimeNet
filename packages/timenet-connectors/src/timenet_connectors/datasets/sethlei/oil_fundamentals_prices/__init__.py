"""The ``sethlei/oil-fundamentals-prices`` connector: the open panel plus EIA's licensed price series."""

from timenet_connectors.datasets.sethlei.oil_fundamentals_prices.connector import (
    CONNECTOR,
    OilFundamentalsPricesConnector,
)


__all__ = ["CONNECTOR", "OilFundamentalsPricesConnector"]
