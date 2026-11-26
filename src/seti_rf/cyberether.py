"""Helpers for CyberEther flowgraphs.

The CyberEther Frame block draws the boxes listed in its input tensor's
``boxes`` attribute. :class:`ScrollingWaterfall` turns detection regions in
seconds and MHz into those boxes for a waterfall whose newest row is at the
bottom, so a Python block can highlight each detection where it appears::

    waterfall = ScrollingWaterfall(rows=200, columns=672, row_s=0.01024,
                                   low_mhz=1063.75, high_mhz=1735.75)
    ctx.output_attrs[0]["boxes"] = waterfall.boxes(now_s, events)

An event marks its location with ``regions``, a list of
``[start_s, stop_s, low_mhz, high_mhz]`` rectangles. Events without regions
are skipped.
"""
from dataclasses import dataclass

__all__ = ["CLASSIFICATION_COLORS", "DEFAULT_COLOR", "ScrollingWaterfall"]

CLASSIFICATION_COLORS = {
    "human_made": [1.0, 0.62, 0.1, 1.0],
    "unknown": [0.2, 1.0, 0.75, 1.0],
}
DEFAULT_COLOR = [1.0, 0.85, 0.2, 1.0]


@dataclass(frozen=True)
class ScrollingWaterfall:
    """Pixel geometry of a waterfall that scrolls up, with the newest row at the bottom.

    Row ``rows - 1`` ends at the current time and each row spans ``row_s``
    seconds. Column 0 starts at ``low_mhz`` and the last column ends at
    ``high_mhz``. Set ``flipped`` when column 0 holds the highest frequency.
    """

    rows: int
    columns: int
    row_s: float
    low_mhz: float
    high_mhz: float
    flipped: bool = False

    def __post_init__(self):
        if self.rows <= 0 or self.columns <= 0:
            raise ValueError("rows and columns must be positive")
        if self.row_s <= 0:
            raise ValueError("row_s must be positive")
        if self.high_mhz <= self.low_mhz:
            raise ValueError("high_mhz must exceed low_mhz")

    def column(self, mhz):
        """Return the fractional column of a frequency."""
        x = (mhz - self.low_mhz) / (self.high_mhz - self.low_mhz) * self.columns
        return self.columns - x if self.flipped else x

    def row(self, now_s, time_s):
        """Return the fractional row of a time, given the time of the newest row's end."""
        return self.rows - (now_s - time_s) / self.row_s

    def box(self, now_s, start_s, stop_s, low_mhz, high_mhz, label="", color=None):
        """Return a Frame box for a region, or ``None`` when it is off screen."""
        top = self.row(now_s, min(start_s, stop_s))
        bottom = self.row(now_s, max(start_s, stop_s))
        left, right = sorted((self.column(low_mhz), self.column(high_mhz)))
        top, bottom = max(top, 0.0), min(bottom, float(self.rows))
        left, right = max(left, 0.0), min(right, float(self.columns))
        if bottom <= top or right <= left:
            return None
        box = {"x": left, "y": top, "width": right - left, "height": bottom - top,
               "color": list(color or DEFAULT_COLOR)}
        if label:
            box["label"] = label
        return box

    def boxes(self, now_s, events, colors=CLASSIFICATION_COLORS):
        """Return the visible Frame boxes for every region of every event.

        The first visible box of each event carries the event's ``kind`` as its
        label, and boxes take their color from the event's ``classification``.
        """
        result = []
        for event in events:
            color = colors.get(event.get("classification"), DEFAULT_COLOR)
            label = event.get("kind", "")
            for region in event.get("regions", ()):
                box = self.box(now_s, *region, label=label, color=color)
                if box is not None:
                    result.append(box)
                    label = ""
        return result
