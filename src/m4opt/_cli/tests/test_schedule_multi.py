from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from click import UsageError
from typer.testing import CliRunner

from ... import missions
from .. import app
from .. import schedule_multi as schedule_multi_module


def test_invert_footprints_handles_overlap_and_uncovered_pixels():
    inverted = schedule_multi_module.invert_footprints(
        [[1, 2], [0, 2], [2]], n_pixels=4
    )

    assert [pixels.tolist() for pixels in inverted] == [
        [1],
        [0],
        [0, 1, 2],
        [],
    ]


@pytest.mark.parametrize(
    "footprints,n_pixels",
    [
        ([], 0),
        ([], 3),
        ([[], []], 2),
    ],
)
def test_invert_footprints_empty_coverage(footprints, n_pixels):
    inverted = schedule_multi_module.invert_footprints(footprints, n_pixels)
    assert len(inverted) == n_pixels
    assert all(indices.size == 0 for indices in inverted)


def test_invert_footprints_to_regions_groups_identical_coverage():
    pixel_to_region, region_to_fields = (
        schedule_multi_module.invert_footprints_to_regions(
            [[0, 1], [1, 2]], n_pixels=4
        )
    )

    assert len(region_to_fields) == 4
    for pixel, footprint in enumerate([(0,), (0, 1), (1,), ()]):
        fields = region_to_fields[pixel_to_region[pixel]]
        expected = [
            field_index
            for field_index, field_pixels in enumerate([[0, 1], [1, 2]])
            if pixel in field_pixels
        ]
        assert fields.tolist() == expected
        assert tuple(fields.tolist()) == footprint


def test_prepare_piecewise_breakpoints_truncates_at_first_large_exposure():
    breakpoints = np.array(
        [[0, 10], [10, 20], [20, schedule_multi_module.LARGE_EXPTIME], [30, 40]]
    )

    assert schedule_multi_module.prepare_piecewise_breakpoints(breakpoints) == [
        (0, 10),
        (10, 20),
    ]


def test_prepare_piecewise_breakpoints_keeps_all_finite_breakpoints():
    breakpoints = np.array([[0.0, 1.5], [1.5, 3.0]])
    assert schedule_multi_module.prepare_piecewise_breakpoints(breakpoints) == [
        (0.0, 1.5),
        (1.5, 3.0),
    ]


def test_prepare_piecewise_breakpoints_accepts_empty_input():
    assert schedule_multi_module.prepare_piecewise_breakpoints(
        np.empty((0, 2))
    ) == []


def test_select_missions_accepts_single_and_multiple_missions():
    selected = schedule_multi_module.select_missions(["ztf", "rubin"])

    assert list(selected) == ["ztf", "rubin"]
    assert selected["ztf"] is missions.ztf
    assert selected["rubin"] is missions.rubin


def test_select_missions_rejects_unknown_missions_with_sorted_names():
    with pytest.raises(UsageError, match="alpha, zulu"):
        schedule_multi_module.select_missions(["zulu", "alpha"])


def test_select_missions_rejects_duplicates():
    with pytest.raises(UsageError, match="Each mission may only be specified once"):
        schedule_multi_module.select_missions(["ztf", "ztf"])


def test_select_missions_requires_at_least_one_mission():
    with pytest.raises(UsageError, match="Specify at least one mission"):
        schedule_multi_module.select_missions([])


def test_prepare_visit_configuration_cycles_bands_and_exposure_times():
    bands, exposure_times = schedule_multi_module.prepare_visit_configuration(
        visits=5,
        bandpass=["g", "r"],
        exptime_min=[120 * u.s, 5 * u.min],
    )

    assert bands == ["g", "r", "g", "r", "g"]
    np.testing.assert_allclose(exposure_times, [120, 300, 120, 300, 120])


def test_prepare_visit_configuration_repeats_single_exposure_for_all_visits():
    bands, exposure_times = schedule_multi_module.prepare_visit_configuration(
        visits=3,
        bandpass=["i"],
        exptime_min=[2 * u.min],
    )

    assert bands == ["i", "i", "i"]
    np.testing.assert_allclose(exposure_times, [120, 120, 120])


def test_prepare_visit_configuration_supports_no_bandpass():
    bands, exposure_times = schedule_multi_module.prepare_visit_configuration(
        visits=2,
        bandpass=None,
        exptime_min=[30 * u.s],
    )

    assert bands == [None, None]
    np.testing.assert_allclose(exposure_times, [30, 30])


@pytest.mark.parametrize(
    "visits,bandpass,exptime_min,message",
    [
        (0, ["g"], [30 * u.s], "at least one"),
        (2, ["g"], [], "minimum exposure time"),
    ],
)
def test_prepare_visit_configuration_rejects_empty_required_inputs(
    visits, bandpass, exptime_min, message
):
    with pytest.raises(UsageError, match=message):
        schedule_multi_module.prepare_visit_configuration(
            visits, bandpass, exptime_min
        )


def test_cli_help_exposes_multi_mission_options():
    result = CliRunner().invoke(app, ["schedule-multi", "--help"])

    assert result.exit_code == 0
    for option in (
        "--mission",
        "--bandpass",
        "--visits",
        "--cadence",
        "--exptime-min",
        "--exptime-max",
        "--jobs",
        "--cutoff",
        "--write-progress",
        "--write-model",
    ):
        assert option in result.output


@pytest.mark.parametrize(
    "mission_args,error_message",
    [
        (["--mission=unknown"], "Unknown mission"),
        (["--mission=ztf", "--mission=ztf"], "only be specified once"),
    ],
)
def test_cli_rejects_invalid_mission_selection(
    tmp_path, mission_args, error_message
):
    skymap_path = tmp_path / "placeholder.fits"
    skymap_path.write_bytes(b"")
    result = CliRunner().invoke(
        app,
        [
            "schedule-multi",
            str(skymap_path),
            str(tmp_path / "schedule"),
            *mission_args,
        ],
    )

    assert result.exit_code != 0
    assert error_message in str(result.exception)
    assert "loading sky map" not in result.output


@pytest.mark.parametrize(
    "invalid_options",
    [
        ["--visits=0"],
        ["--jobs=-1"],
        ["--cutoff=1.01"],
    ],
)
def test_cli_rejects_out_of_range_options_before_loading_skymap(
    tmp_path, invalid_options
):
    skymap_path = tmp_path / "placeholder.fits"
    skymap_path.write_bytes(b"")
    result = CliRunner().invoke(
        app,
        [
            "schedule-multi",
            str(skymap_path),
            str(tmp_path / "schedule"),
            "--mission=ztf",
            *invalid_options,
        ],
    )

    assert result.exit_code != 0
    assert "loading sky map" not in result.output


class _Mission:
    name = "test"
    filter_exchange_time = 30 * u.s

    def __init__(self):
        self.slew = SimpleNamespace(
            time=lambda *args: np.full(len(args[0]), 5.0) * u.s
        )

    def observer_location(self, times):
        zeros = np.zeros(len(times)) * u.m
        return EarthLocation.from_geocentric(zeros, zeros, zeros)


class _Solution:
    def __init__(self, fields, visit_times):
        self.values = {id(fields): np.array([0.0, 1.0]), id(visit_times): visit_times}

    def get_values(self, variables):
        return self.values[id(variables)]


def _build_table_inputs(solution_found, has_model=None):
    field_variables = np.array([10, 11])
    visit_times = np.array([[100.0, 400.0], [200.0, 700.0]])
    solution = (
        _Solution(field_variables, visit_times) if solution_found else None
    )
    if has_model is None:
        has_model = solution_found
    return schedule_multi_module.build_mission_table(
        "test",
        _Mission(),
        field_vars={"test": field_variables},
        time_field_visit_vars={"test": visit_times},
        n_fields={"test": 2},
        target_coords={
            "test": SkyCoord([10, 20] * u.deg, [0, 1] * u.deg)
        },
        rolls={"test": np.array([0, 0]) * u.deg},
        field_ids={"test": np.array([4, 7])},
        visit_bandpasses=["g", "r"],
        solution=solution,
        visits=2,
        visit_exptime_min_s=np.array([100.0, 200.0]),
        obstimes=Time(["2026-01-01T00:00:00"], scale="utc"),
        deadline=2 * u.h,
        delay=0 * u.s,
        meta_common={"visits": 2},
        objective_value=0.75 if solution_found else 0.0,
        model=SimpleNamespace(
            best_bound=0.8,
            solve_details=SimpleNamespace(status="optimal", time=1.25),
        ),
        has_model=has_model,
    )


def test_build_mission_table_includes_observations_slews_and_filter_exchange():
    table = _build_table_inputs(True)

    assert list(table["action"]) == ["observe", "slew", "observe"]
    assert list(table["bandpass"]) == ["g", "", "r"]
    assert list(table["field_id"][table["action"] == "observe"]) == [7, 7]
    np.testing.assert_allclose(
        table["duration"][table["action"] == "observe"].to_value(u.s),
        [100, 200],
    )
    assert table["duration"][table["action"] == "slew"] == 30 * u.s
    assert table.meta["objective_value"] == pytest.approx(0.75)
    assert table.meta["solution_status"] == "optimal"
    assert table.meta["total_time"]["slew"] == 30 * u.s


def test_build_mission_table_without_solution_is_valid_and_empty():
    table = _build_table_inputs(False)

    assert len(table) == 0
    assert table.meta["total_time"] == {}
    assert (
        table.meta["solution_status"]
        == "infeasible, no observable fields or pixels"
    )


def test_build_mission_table_without_feasible_solution_keeps_solver_status():
    table = _build_table_inputs(False, has_model=True)

    assert len(table) == 0
    assert table.meta["solution_status"] == "optimal"
    assert table.meta["best_bound"] == pytest.approx(0.8)
