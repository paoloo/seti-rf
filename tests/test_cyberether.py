import pytest

from seti_rf.cyberether import CLASSIFICATION_COLORS, DEFAULT_COLOR, ScrollingWaterfall

WATERFALL = ScrollingWaterfall(rows=100, columns=200, row_s=0.01, low_mhz=1000.0, high_mhz=1200.0)


def test_newest_row_ends_at_the_bottom_and_older_rows_scroll_up():
    assert WATERFALL.row(5.0, 5.0) == 100
    assert WATERFALL.row(5.0, 4.5) == pytest.approx(50)
    assert WATERFALL.column(1000.0) == 0
    assert WATERFALL.column(1150.0) == pytest.approx(150)


def test_box_covers_the_region_in_frame_pixels():
    box = WATERFALL.box(5.0, 4.5, 4.6, 1100.0, 1110.0, label="pulse")
    assert box == {"x": pytest.approx(100), "y": pytest.approx(50), "width": pytest.approx(10),
                   "height": pytest.approx(10), "color": DEFAULT_COLOR, "label": "pulse"}


def test_box_is_clipped_at_the_top_and_dropped_once_off_screen():
    box = WATERFALL.box(5.0, 3.95, 4.05, 1100.0, 1110.0)
    assert box["y"] == 0
    assert box["height"] == pytest.approx(5)
    assert WATERFALL.box(5.0, 3.0, 4.0, 1100.0, 1110.0) is None
    assert WATERFALL.box(5.0, 4.5, 4.6, 1300.0, 1400.0) is None


def test_flipped_columns_put_the_highest_frequency_first():
    flipped = ScrollingWaterfall(100, 200, 0.01, 1000.0, 1200.0, flipped=True)
    box = flipped.box(5.0, 4.5, 4.6, 1000.0, 1010.0)
    assert box["x"] == pytest.approx(190)
    assert box["width"] == pytest.approx(10)


def test_events_label_their_first_visible_region_and_use_classification_colors():
    events = [
        {"kind": "sweep", "classification": "unknown",
         "regions": [[3.0, 3.5, 1100.0, 1200.0], [4.5, 4.6, 1000.0, 1100.0], [4.6, 4.7, 1000.0, 1050.0]]},
        {"kind": "carrier", "classification": "human_made", "regions": [[4.9, 5.0, 1010.0, 1012.0]]},
        {"kind": "no_location", "classification": "unknown"},
    ]
    boxes = WATERFALL.boxes(5.0, events)
    assert [box.get("label") for box in boxes] == ["sweep", None, "carrier"]
    assert boxes[0]["color"] == CLASSIFICATION_COLORS["unknown"]
    assert boxes[2]["color"] == CLASSIFICATION_COLORS["human_made"]


def test_invalid_geometry_is_rejected():
    with pytest.raises(ValueError):
        ScrollingWaterfall(0, 10, 0.01, 1.0, 2.0)
    with pytest.raises(ValueError):
        ScrollingWaterfall(10, 10, 0.01, 2.0, 1.0)
