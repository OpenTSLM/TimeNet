from pathlib import Path

from timenet.types import ClassificationTask, read_value
from timenet_connectors.datasets.cubico.kelmarsh import tasks
from timenet_connectors.datasets.cubico.kelmarsh.farm import SOURCE_COLUMNS, TurbineYearFiles
from timenet_connectors.datasets.cubico.kelmarsh.greenbyte import STEPS_PER_DAY
from timenet_connectors.datasets.cubico.penmanshiel.connector import PenmanshielConnector


# A synthetic export of one invented turbine for one day. It has the shape of the release and
# holds no value of it. The turbine table of the release ends each line with a comma and has
# rows without a turbine.
_COMMENTS = "# Synthetic export.\n#\n# Turbine: Testfarm 02\n# Time zone: UTC\n#\n#\n#\n#\n#\n"
_STATIC = (
    "\ufeffWind Farm,Title,Alternative Title,Identity,Manufacturer,Model,Rated power (kW),Hub Height (m),"
    "Rotor Diameter (m),Latitude,Longitude,Elevation (m),Country,Commercial Operations Date,\n"
    "Testfarm,Testfarm 01,T01,X 1,Maker,M1,2000,59,82,1.5,-2.5,100,UK,01/09/2016,\n"
    ",,,,,,,,,,,,,,\n"
    "Testfarm,Testfarm 02,T02,X 2,Maker,M1,2000,60,82,1.6,-2.6,100,UK,01/09/2016,\n"
    ",,,,,,,,,,,,,,\n"
)
_STATUS = (
    "Timestamp start,Timestamp end,Duration,Status,Code,Message,Comment,Service contract category,IEC category,"
    "Global contract category,Custom contract category\n"
    "2022-03-01 10:00:00,2022-03-01 10:30:00,00:30:00,Curtailment,108,Grid constraint curtailment,,,"
    "Partial Performance,,\n"
)


def _write_release(directory: Path) -> TurbineYearFiles:
    ref = TurbineYearFiles(
        data_path=directory / "Turbine_Data_Testfarm_02.csv",
        status_path=directory / "Status_Testfarm_02.csv",
        static_path=directory / "static.csv",
    )
    rows = [
        f"2022-03-01 {step // 6:02d}:{step % 6}0:00," + ",".join(["1.5"] * len(SOURCE_COLUMNS))
        for step in range(STEPS_PER_DAY)
    ]
    header = ",".join(["# Date and time", *SOURCE_COLUMNS])
    ref.data_path.write_text(_COMMENTS + "\n".join([header, *rows]) + "\n", encoding="utf-8")
    ref.status_path.write_text(_COMMENTS + _STATUS, encoding="utf-8")
    ref.static_path.write_text(_STATIC, encoding="utf-8")
    return ref


def test_the_card_names_the_dataset():
    assert PenmanshielConnector().metadata().dataset_id == "cubico/penmanshiel"


def test_each_year_has_an_archive_for_each_group_of_turbines():
    for year, archives in PenmanshielConnector.YEAR_ARCHIVES.items():
        assert [name.split("_")[2:4] for name, _ in archives] == [[str(year), "WT01-10"], [str(year), "WT11-15"]]


def test_a_day_of_the_release_format_becomes_a_record_with_tasks(tmp_path):
    dataset = PenmanshielConnector().convert([_write_release(tmp_path)])
    (record,) = dataset.records
    assert record.metadata["turbine"] == "Testfarm 02"
    assert record.metadata["hub_height_m"] == 60
    assert [(a.key, read_value(a.value)) for a in record.annotations] == [
        (tasks.CURTAILMENT_KEY, "Grid constraint curtailment")
    ]
    labels = {read_value(task.prompt): task.targets for task in dataset.tasks_of(ClassificationTask)}
    assert labels[tasks.STOPPED_PROMPT] == ("running",)
    assert labels[tasks.CURTAILED_PROMPT] == ("curtailed",)


def test_the_card_has_the_new_version():
    assert str(PenmanshielConnector().metadata().dataset_version) == "1.1.0"
