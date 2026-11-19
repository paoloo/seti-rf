from sigint.rfi import bands


def test_lookup_finds_gps_l1_ca():
    hits = bands.lookup(1575.42e6)
    assert any(h.label == "GPS L1 C/A" for h in hits)
    assert all(h.origin == "space" for h in hits)


def test_lookup_finds_adsb_and_flags_it_terrestrial():
    hits = bands.lookup(1090e6)
    labels = [h.label for h in hits]
    assert "ADS-B / Mode S 1090ES" in labels
    hit = next(h for h in hits if h.label == "ADS-B / Mode S 1090ES")
    assert hit.origin == "terrestrial"


def test_lookup_empty_for_a_quiet_frequency():
    assert bands.lookup(1_000.0) == ()  # 1 kHz: below everything in the table


def test_every_band_is_well_formed():
    for b in bands.KNOWN_BANDS:
        assert b.lo_hz < b.hi_hz, b.label
        assert b.origin in ("space", "terrestrial"), b.label
        assert b.label


def test_lookup_range_overlaps_a_wide_span():
    hits = bands.lookup_range(1_570_000_000, 1_630_000_000)
    labels = {h.label for h in hits}
    assert "GPS L1 C/A" in labels
    assert "Iridium" in labels
