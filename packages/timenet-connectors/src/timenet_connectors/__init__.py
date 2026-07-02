"""Dataset connectors for TimeNet.

Concrete connectors that fetch raw sources and convert them into the TimeF format. The connector
contract itself (:class:`~timenet.connectors.BaseConnector`) lives in the ``timenet`` package.
"""

from timenet_connectors.hello_world import HelloWorldConnector


__all__ = ["HelloWorldConnector"]
