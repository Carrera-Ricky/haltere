import numpy as np

from haltere.vision.tposts import detect_tposts, observation_from_detections


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
        "side": "none",
    }
