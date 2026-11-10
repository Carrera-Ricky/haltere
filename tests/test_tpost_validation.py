import numpy as np

from haltere.vision.tposts import (
    add_distance_estimates,
    focal_length_from_vertical_fov,
    validate_against_metadata,
)


def test_focal_length_from_vertical_fov():
    assert round(focal_length_from_vertical_fov(360, 80.0), 3) == 214.516


def test_validation_reports_forward_depth_error():
    metadata = {
        "camera_position": [0.0, 0.0, 0.0],
        "camera_forward": [0.0, 0.0, -1.0],
        "post_positions": [[-1.0, 0.0, -10.0], [1.0, 0.0, -10.0]],
    }
    detections = add_distance_estimates([
        {"center_x": -0.2, "pixel_height": 20, "bbox": [0, 0, 4, 20]},
        {"center_x": 0.2, "pixel_height": 20, "bbox": [0, 0, 4, 20]},
    ], object_height_m=1.0, focal_length_px=200.0)

    result = validate_against_metadata(detections, metadata)

    assert len(result) == 2
    assert result[0]["expected_forward_m"] == 10.0
    assert result[0]["estimated_forward_m"] == 10.0
    assert result[0]["expected_range_m"] > 10.0
