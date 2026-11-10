import numpy as np

from haltere.vision.tposts import (add_body_observations, add_camera_directions, detect_tposts, estimate_distance_m,
                                   observation_from_detections)


def test_tpost_detection_merges_pole_and_crossbar():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[25:65, 28:32] = [220, 50, 10]
    image[25:29, 20:40] = [220, 50, 10]
    image[20:60, 88:92] = [220, 50, 10]
    image[20:24, 80:100] = [220, 50, 10]

    detections = detect_tposts(image)

    assert len(detections) == 2
    assert detections[0]["pixel_height"] >= 40
    assert detections[1]["pixel_height"] >= 40


def test_observation_reports_nearest_post_side():
    observation = observation_from_detections([
        {"bbox": [10, 20, 8, 30], "center_x": -0.4, "center_y": 0.1, "pixel_height": 30},
        {"bbox": [80, 20, 8, 50], "center_x": 0.5, "center_y": 0.2, "pixel_height": 50},
    ])

    assert observation["visible"] is True
    assert observation["side"] == "right"
    assert observation["pixel_height"] == 50.0


def test_observation_handles_no_detection():
    observation = observation_from_detections([])

    assert observation == {
        "visible": False,
        "center_x": 0.0,
        "center_y": 0.0,
        "pixel_height": 0.0,
        "pixel_width": 0.0,
        "distance_m": 0.0,
        "direction_camera": [0.0, 0.0, 1.0],
        "offset_right_m": 0.0,
        "offset_down_m": 0.0,
        "side": "none",
    }


def test_distance_uses_pinhole_camera_relation():
    assert estimate_distance_m(50.0, object_height_m=3.0, focal_length_px=200.0) == 12.0


def test_camera_direction_and_offset_are_consistent():
    detections = [{
        "center_px": [420.0, 180.0],
        "distance_m": 10.0,
    }]

    result = add_camera_directions(detections, 640, 360, 200.0)[0]

    assert result["direction_camera"][0] > 0.0
    assert result["direction_camera"][2] > 0.0
    assert result["offset_right_m"] > 0.0
    assert abs(result["offset_down_m"]) < 1e-9


def test_body_observation_has_forward_left_up_axes():
    detections = add_camera_directions([{
        "center_px": [320.0, 180.0],
        "distance_m": 10.0,
    }], 640, 360, 200.0)

    result = add_body_observations(detections, tilt_deg=30.0)[0]

    assert result["direction_body"][0] > 0.0
    assert abs(result["offset_left_m"]) < 1e-9
    assert result["offset_forward_m"] > 0.0
