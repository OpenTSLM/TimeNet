"""Check image tensors shared by visual dataset connectors."""

import numpy as np
from PIL import Image

from timenet_connectors.sources.images import image_signal


def test_transparent_png_keeps_its_size_and_white_background(tmp_path):
    path = tmp_path / "chart.png"
    image = Image.new("RGBA", (3, 2), (0, 0, 0, 0))
    image.putpixel((1, 0), (255, 0, 0, 255))
    image.save(path)

    signal = image_signal(path, signal_id="chart", name="chart")
    restored = signal.to_numpy()

    assert signal.spec.value_shape == (2, 3, 3)
    assert restored.shape == (1, 2, 3, 3)
    np.testing.assert_array_equal(restored[0, 0, 0], [255, 255, 255])
    np.testing.assert_array_equal(restored[0, 0, 1], [255, 0, 0])
