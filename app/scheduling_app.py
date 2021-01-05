#!/usr/bin/env python

""" This streamlit app is meant to help schedule shifts """
__VERSION__ = "0.1"
__AUTHOR__ = "Alberto Nava <alberto_nava@berkeley.edu>"

# =======================================================================#
# Importations
# =======================================================================#

# Native Python libraries
import base64
from typing import List, Dict, Optional
import statistics
from dataclasses import dataclass

# Non-native python libraries
import pandas as pd
import streamlit as st
from ortools.sat.python.cp_model import CpModel, CpSolver
import plotly.graph_objects as go
import plotly.express as px


# =======================================================================#
# Utility Classes
# =======================================================================#


@dataclass
class Lab:
    index: int
    name: str
    alias: Optional[str] = None


@dataclass
class Location:
    x: int
    y: int


@dataclass
class Bench:
    index: int
    full_name: str  # 4550-32
    name: str  # 32
    lab: Lab  # SB1, SB2, SB3
    location: Location
    active_shift: str  # AM, PM


@dataclass
class Shift:
    index: int
    label: str  # 'AM', 'PM', or 'FULL'
    max_capacity: int
    day: str  # Monday, Tuesday, etc.


@dataclass
class Day:
    index: int
    label: str  # Monday, Tuesday, etc.
    shifts: Dict[str, Shift]


@dataclass
class Employee:
    index: int
    name: str
    id: str
    max_slots: int
    preferences: Dict[int, int]
    home_bench: Optional[Bench] = None


@dataclass
class ConstraintParameters:
    max_unfairness: int = -1
    fairness_weight: int = 0
    fill_schedule_weight: int = 1
    include_location_optimization: bool = False


@dataclass
class MapParameters:
    labs: str


@dataclass
class Constraints:
    raw_slot_constraints: pd.DataFrame
    slot_constraints: pd.DataFrame
    raw_bench_constraints: Optional[pd.DataFrame] = None
    raw_lab_separation: Optional[pd.DataFrame] = None
    raw_lab_maps: Optional[Dict[str, pd.DataFrame]] = None


@dataclass
class Schedule:
    objective_score: int
    time_to_solve: float
    number_of_shifts_per_employee: Dict[int, int]
    score_per_employee: Dict[int, int]
    score_variance: float
    min_employee_score: int
    max_employee_score: int
    total_possible_shifts: int
    total_shifts_filled: int
    employee_schedule: Dict[int, List[Shift]]  # {Employee.id: [Shifts]}
    shift_schedule: Dict[int, List[Employee]]  # {Shift.id: [Employees]}
    shift_bench_schedule: Optional[
        Dict[int, Dict[int, Bench]]
    ] = None  # {Shift.id: {Employee.id: Bench}}
    employee_bench_schedule: Optional[
        Dict[int, Dict[int, Bench]]
    ] = None  # {Employee.id: {Shift.id: Bench}}
    bench_objective_score: Optional[int] = None
    bench_optimization_average_distance: Optional[
        Dict[int, float]
    ] = None  # {Employee.id: average distance}
    pretty_employee_schedule: Optional[pd.DataFrame] = pd.DataFrame()


PREFERENCE_TO_SCORE: Dict[int, int] = {
    1: 10,
    2: 9,
    3: 8,
    4: 7,
    5: 6,
    6: 5,
    7: 4,
    8: 3,
    9: 2,
    10: 1,
    11: 0,
    12: -1,
    13: -2,
    14: -3,
    15: -4,
    16: -5,
    17: -6,
    18: -7,
    19: -8,
    20: -9,
    21: -10,
    0: -1000,
}

# =======================================================================#
# Utility Functions
# =======================================================================#


def read_constraints(
    constraints_xlsx: str, parameters: ConstraintParameters
) -> Constraints:
    constraints_file: pd.ExcelFile = pd.ExcelFile(constraints_xlsx)
    raw_slot_constraints = constraints_file.parse(
        sheet_name="slot_constraints",
        header=0,
        index_col=None,
    )
    slot_constraints = prepare_schedule(
        raw_slot_constraints=raw_slot_constraints
    )

    constraints: Constraints
    if parameters.include_location_optimization:
        raw_bench_constraints = constraints_file.parse(
            sheet_name="bench_constraints",
            header=0,
            index_col=None,
        )
        raw_bench_constraints["benches"] = raw_bench_constraints[
            "benches"
        ].apply(lambda value: value.split(","))
        raw_lab_separation = constraints_file.parse(
            sheet_name="lab_separation",
            header=0,
            index_col=0,
        )

        raw_lab_maps: Dict[str, pd.DataFrame] = {}
        for lab in raw_lab_separation.index:
            raw_lab_maps[lab] = constraints_file.parse(
                sheet_name=str(lab), header=None, index_col=None, dtype=str
            ).fillna("")

        constraints = Constraints(
            raw_slot_constraints=raw_slot_constraints,
            slot_constraints=slot_constraints,
            raw_bench_constraints=raw_bench_constraints,
            raw_lab_separation=raw_lab_separation,
            raw_lab_maps=raw_lab_maps,
        )
    else:
        constraints = Constraints(
            raw_slot_constraints=raw_slot_constraints,
            slot_constraints=slot_constraints,
        )
    return constraints


def prepare_schedule(raw_slot_constraints: pd.DataFrame) -> pd.DataFrame:
    df: pd.DataFrame = raw_slot_constraints.melt(id_vars=["Shift"]).rename(
        columns={"variable": "day", "value": "slots", "Shift": "shift"}
    )
    # df = df.loc[df["slots"] != 0, :].reset_index(drop=True).reset_index()
    df = df.reset_index(drop=True).reset_index()
    return df


@st.cache
def read_roster(roster_csv: str) -> pd.DataFrame:
    original_df: pd.DataFrame = pd.read_csv(roster_csv)
    return original_df


def create_day(obj_in: pd.Series, db: Dict[str, Day]) -> Dict[str, Day]:
    if obj_in["day"] not in db.keys():
        db[obj_in["day"]] = Day(
            index=len(db.keys()), label=obj_in["day"], shifts={}
        )
    if obj_in["shift"] not in db[obj_in["day"]].shifts:
        db[obj_in["day"]].shifts[obj_in["shift"]] = Shift(
            index=obj_in["index"],
            label=obj_in["shift"],
            max_capacity=obj_in["slots"],
            day=obj_in["day"],
        )
    return db


def create_employee(
    obj_in: pd.Series, db: Dict[int, Employee], days: Dict[str, Day]
) -> Dict[int, Employee]:
    existing_objs = db.keys()
    employee_preferences: Dict[int, int] = prepare_preferences(
        employee_data=obj_in, days=days
    )
    if obj_in.id not in existing_objs:
        obj_index: int = len(existing_objs)
        db[obj_index] = Employee(
            index=obj_index,
            name=obj_in["name"],
            id=obj_in.id,
            max_slots=obj_in.max_slots,
            preferences=employee_preferences,
        )
    return db


def prepare_preferences(
    employee_data: pd.Series, days: Dict[str, Day]
) -> Dict[int, int]:
    employee_preferences: Dict[int, int] = {}
    for value in employee_data.index:
        try:
            day, shift = value.split("_")
            employee_preferences[
                days[day].shifts[shift].index
            ] = PREFERENCE_TO_SCORE[employee_data[value]]
        except ValueError:
            pass
        except KeyError:
            pass
    return employee_preferences


def create_lab(obj_in: pd.Series, db: Dict[str, Lab]) -> Dict[str, Lab]:
    existing_labs = db.keys()
    if obj_in["lab"] not in db:
        db[obj_in["lab"]] = Lab(
            index=len(existing_labs),
            name=obj_in["lab"],
            alias=obj_in["alias"],
        )
    return db


def find_active_shift(
    constraints: Constraints, lab: Lab, bench_name: str
) -> str:
    assert constraints.raw_bench_constraints is not None
    active_shift: str
    if bench_name:
        tmp_shift_var: pd.Series = constraints.raw_bench_constraints.loc[
            (constraints.raw_bench_constraints["lab"] == lab.name)
            & (
                constraints.raw_bench_constraints["benches"].apply(
                    lambda active_benches: bench_name in active_benches
                )
            ),
            "shift",
        ]
        active_shift = (
            "" if tmp_shift_var.empty else tmp_shift_var.values[0]
        )
    else:
        active_shift = ""
    return active_shift


def create_bench(
    name: str,
    lab: Lab,
    location: Location,
    active_shift: str,
    db: Dict[str, Bench],
) -> None:
    if name:
        existing_benches = db.keys()
        bench_name = f"{lab.name}-{name}"
        if bench_name not in db:
            db[bench_name] = Bench(
                index=len(existing_benches),
                full_name=bench_name,
                name=name,
                location=location,
                lab=lab,
                active_shift=active_shift,
            )
    else:
        pass
    return None


def prettify_schedule(
    schedule: Schedule,
    all_employees: Dict[int, Employee],
    all_shifts: Dict[int, Shift],
    all_days: Dict[str, Day],
) -> pd.DataFrame:
    nice_schedule: pd.DataFrame
    if schedule.shift_bench_schedule:
        nice_schedule = pd.DataFrame(
            {day: [] for day in all_days}.update(
                {
                    "score": [],
                    "number_of_shifts": [],
                    "max_desired_shifts": [],
                    "home_bench": [],
                }
            )
        )
    else:
        nice_schedule = pd.DataFrame(
            {day: [] for day in all_days}.update(
                {
                    "score": [],
                    "number_of_shifts": [],
                    "max_desired_shifts": [],
                }
            )
        )
    for employee in all_employees:
        employee_row = {}
        for shift in schedule.employee_schedule[employee]:
            if schedule.shift_bench_schedule:
                if shift.day in employee_row:
                    employee_row[shift.day] = (
                        f"AM+PM "
                        f"[{schedule.shift_bench_schedule[shift.index][employee].full_name}]"
                        f"({all_employees[employee].preferences[shift.index]})"
                    )
                else:
                    employee_row[shift.day] = (
                        f"{shift.label} "
                        f"[{schedule.shift_bench_schedule[shift.index][employee].full_name}]"
                        f"({all_employees[employee].preferences[shift.index]})"
                    )
            else:
                if shift.day in employee_row:
                    employee_row[shift.day] = (
                        f"AM+PM "
                        f"({all_employees[employee].preferences[shift.index]})"
                    )
                else:
                    employee_row[shift.day] = (
                        f"{shift.label} "
                        f"({all_employees[employee].preferences[shift.index]})"
                    )
        for day in all_days:
            if day not in employee_row:
                employee_row[day] = ""
        employee_row["score"] = str(schedule.score_per_employee[employee])
        employee_row["number_of_shifts"] = str(
            schedule.number_of_shifts_per_employee[employee]
        )
        employee_row["max_desired_shifts"] = str(
            all_employees[employee].max_slots
        )
        if schedule.shift_bench_schedule:
            employee_row["home_bench"] = all_employees[
                employee
            ].home_bench.full_name
        nice_schedule = nice_schedule.append(
            pd.Series(
                employee_row,
                name=all_employees[employee].name,
            )
        )

    AM_ROW: Dict[str, int] = {}
    PM_ROW: Dict[str, int] = {}
    FULL_ROW: Dict[str, int] = {}
    TOTAL_ROW: Dict[str, int] = {}
    for day in all_days:
        AM_ROW[day] = nice_schedule[day].str.startswith("AM").sum()
        PM_ROW[day] = nice_schedule[day].str.startswith("PM").sum()
        FULL_ROW[day] = nice_schedule[day].str.startswith("FULL").sum()
        TOTAL_ROW[day] = AM_ROW[day] + PM_ROW[day] + FULL_ROW[day]

    nice_schedule = nice_schedule.append(
        [
            pd.Series(AM_ROW, name="AM Swipes/Day"),
            pd.Series(PM_ROW, name="PM Swipes/Day"),
            pd.Series(FULL_ROW, name="FULL Day Swipes/Day"),
            pd.Series(TOTAL_ROW, name="TOTAL Daily Swipes/Day"),
        ]
    )
    if schedule.shift_bench_schedule:
        nice_schedule = nice_schedule.loc[
            :,
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
                "number_of_shifts",
                "max_desired_shifts",
                "score",
                "home_bench",
            ],
        ]
    else:
        nice_schedule = nice_schedule.loc[
            :,
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
                "number_of_shifts",
                "max_desired_shifts",
                "score",
            ],
        ]
    return nice_schedule


def get_table_download_link(
    df: pd.DataFrame, label: str, output_filename: str, index: bool = True
) -> str:
    """Generates a link allowing the data in a given panda dataframe to be
    downloaded
    in:  dataframe
    out: href string

    From:
    https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
    """
    csv = df.to_csv(index=index)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    href = (
        f'<a href="data:file/csv;base64,{b64}" '
        f'download="{output_filename}">{label}</a>'
    )
    return href


def get_excel_download_link(
    excel_file: str, label: str, output_filename: str
) -> str:
    """Generates a link allowing the data in a given excel file to be
    downloaded
    in:  excel filepath
    out: href string

    From:
    https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
    """
    excel_contents: bytes
    with open(excel_file, "rb") as F:
        excel_contents = F.read()
    b64 = base64.b64encode(
        excel_contents
    ).decode()  # some strings <-> bytes conversions necessary here
    href = (
        f'<a href="data:file/xlsx;base64,{b64}" '
        f'download="{output_filename}">{label}</a>'
    )
    return href


def create_demand_heatmap(roster: pd.DataFrame) -> go.Figure:
    roster_formatted_for_heatmap: pd.DataFrame = roster.T.reset_index()
    mask: pd.Series = (
        roster_formatted_for_heatmap["index"]
        .str.split("_")
        .str[0]
        .isin(
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
        )
    )
    roster_formatted_for_heatmap = roster_formatted_for_heatmap.loc[
        mask, :
    ]
    split_index: pd.Series = roster_formatted_for_heatmap[
        "index"
    ].str.split("_")
    roster_formatted_for_heatmap["Day"] = split_index.str[0]
    roster_formatted_for_heatmap["Shift"] = split_index.str[1]
    roster_formatted_for_heatmap = roster_formatted_for_heatmap.drop(
        columns=["index"]
    ).reset_index(drop=True)
    for col in roster_formatted_for_heatmap.columns.difference(
        ["Day", "Shift"]
    ):
        roster_formatted_for_heatmap.loc[
            :, col
        ] = roster_formatted_for_heatmap.loc[:, col].apply(int)
    roster_formatted_for_heatmap = roster_formatted_for_heatmap.applymap(
        lambda value: 21 if value == 0 else value
    )
    roster_formatted_for_heatmap[
        "Average_Preference"
    ] = roster_formatted_for_heatmap.mean(axis=1)
    roster_formatted_for_heatmap = roster_formatted_for_heatmap.drop(
        columns=roster_formatted_for_heatmap.columns.difference(
            ["Day", "Shift", "Average_Preference"]
        )
    )
    roster_formatted_for_heatmap = roster_formatted_for_heatmap.pivot(
        index="Shift", columns="Day", values="Average_Preference"
    ).loc[
        ["AM", "PM", "FULL"],
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=roster_formatted_for_heatmap,
            x=roster_formatted_for_heatmap.columns,
            y=roster_formatted_for_heatmap.index,
        ),
        layout=go.Layout(
            title="<b>Average Shift Preference (lower = more demand)</b>",
            # paper_bgcolor="rgb(248, 248, 255)",
            # plot_bgcolor="rgb(248, 248, 255)",
        ),
    )
    return fig


def create_score_boxplot(optimal_schedule: pd.DataFrame) -> go.Figure:
    fig = px.box(
        data_frame=optimal_schedule,
        y="score",
        points="all",
        hover_name=optimal_schedule.index,
        hover_data=["score", "number_of_shifts", "max_desired_shifts"],
        title="Distribution of Employee Scores",
    )
    return fig


# =======================================================================#
# Main Logic Functions
# =======================================================================#


def compute_optimal_schedule(
    all_employees: Dict[int, Employee],
    all_shifts: Dict[int, Shift],
    all_days: Dict[str, Day],
    schedule_parameters: ConstraintParameters,
) -> Schedule:
    model = CpModel()

    shifts = {}
    for employee in all_employees:
        for shift in all_shifts:
            shifts[(employee, shift)] = model.NewBoolVar(
                f"shift_e{employee}s{shift}"
            )

    # Each shift has max number of people
    # Up to input to calculate AM + FULL and PM + FULL true constraints
    for shift in all_shifts:
        model.Add(
            sum(shifts[(employee, shift)] for employee in all_employees)
            <= all_shifts[shift].max_capacity
        )

    # Each person has max slots
    for employee in all_employees:
        model.Add(
            sum(
                shifts[(employee, shift)]
                if all_shifts[shift].label != "FULL"
                else 2 * shifts[(employee, shift)]
                for shift in all_shifts
            )
            <= all_employees[employee].max_slots
        )

    # A person can only have one of AM, PM, or FULL per day
    for employee in all_employees:
        for day in all_days:
            model.Add(
                sum(
                    shifts[(employee, shift)]
                    for shift in all_shifts
                    if all_shifts[shift].day == day
                )
                <= 1
            )

    # A person might not be able to work all possible shifts
    for employee in all_employees:
        for shift in all_shifts:
            if all_employees[employee].preferences[shift] <= -100:
                model.Add(sum([shifts[(employee, shift)]]) < 1)

    # Workaround for minimizing variance of scores
    # Instead, minimize difference between max score and min score
    # From: https://stackoverflow.com/a/53363585
    employee_scores = {}
    for employee in all_employees:
        employee_scores[employee] = model.NewIntVar(
            -100, 100, f"employee_score_{employee}"
        )
        model.Add(
            employee_scores[employee]
            == sum(
                all_employees[employee].preferences[shift]
                * shifts[(employee, shift)]
                if all_shifts[shift].label != "FULL"
                else 2
                * all_employees[employee].preferences[shift]
                * shifts[(employee, shift)]
                for shift in all_shifts
            )
        )
    min_employee_score = model.NewIntVar(-100, 100, "min_employee_score")
    max_employee_score = model.NewIntVar(-100, 100, "max_employee_score")
    model.AddMinEquality(
        min_employee_score, [employee_scores[e] for e in all_employees]
    )
    model.AddMaxEquality(
        max_employee_score, [employee_scores[e] for e in all_employees]
    )
    # Max Unfairness constraint
    if schedule_parameters.max_unfairness >= 0:
        model.Add(
            max_employee_score - min_employee_score
            <= schedule_parameters.max_unfairness
        )

    # Create objective
    # Maximize points from requests
    # Maximize number of shifts filled
    # Minimize variance of scores per employee
    weights: Dict[str, int] = {
        "preferences": int(
            10
            * (
                1
                / (
                    1
                    + schedule_parameters.fill_schedule_weight
                    + schedule_parameters.fairness_weight
                )
            )
        ),
        "fill_schedule": int(
            10
            * (
                schedule_parameters.fill_schedule_weight
                / (
                    1
                    + schedule_parameters.fill_schedule_weight
                    + schedule_parameters.fairness_weight
                )
            )
        ),
        "fairness": int(
            10
            * (
                schedule_parameters.fairness_weight
                / (
                    1
                    + schedule_parameters.fill_schedule_weight
                    + schedule_parameters.fairness_weight
                )
            )
        ),
    }
    model.Maximize(
        weights["preferences"]
        * sum(
            all_employees[employee].preferences[shift]
            * shifts[(employee, shift)]
            if all_shifts[shift].label != "FULL"
            else 2
            * all_employees[employee].preferences[shift]
            * shifts[(employee, shift)]
            for employee in all_employees
            for shift in all_shifts
        )
        + weights["fill_schedule"]
        * sum(
            shifts[(employee, shift)]
            for employee in all_employees
            for shift in all_shifts
        )
        - weights["fairness"] * (max_employee_score - min_employee_score)
    )

    st.write("Constraints set up. Running Optimization...")
    solver = CpSolver()
    solver.parameters.max_time_in_seconds = 60
    solver.Solve(model)
    st.write("Finished Schedule Optimization!")

    # Prepare results
    shifts_per_employee: Dict[int, int] = {
        employee: sum(
            solver.Value(shifts[(employee, shift)])
            if all_shifts[shift].label != "FULL"
            else 2 * solver.Value(shifts[(employee, shift)])
            for shift in all_shifts
        )
        for employee in all_employees
    }
    score_per_employee: Dict[int, int] = {
        employee: sum(
            all_employees[employee].preferences[shift]
            * solver.Value(shifts[(employee, shift)])
            if all_shifts[shift].label != "FULL"
            else 2
            * all_employees[employee].preferences[shift]
            * solver.Value(shifts[(employee, shift)])
            for shift in all_shifts
        )
        for employee in all_employees
    }
    score_variance: float = statistics.variance(
        score_per_employee.values()
    )
    total_possible_shifts: int = sum(
        all_shifts[shift].max_capacity
        if all_shifts[shift].label != "FULL"
        else 2 * all_shifts[shift].max_capacity
        for shift in all_shifts
    )
    employee_schedule: Dict[int, List[Shift]] = {
        employee: [
            all_shifts[shift]
            for shift in all_shifts
            if solver.Value(shifts[(employee, shift)])
        ]
        for employee in all_employees
    }
    shift_schedule: Dict[int, List[Employee]] = {
        shift: [
            all_employees[employee]
            for employee in all_employees
            if solver.Value(shifts[(employee, shift)])
        ]
        for shift in all_shifts
    }

    schedule: Schedule = Schedule(
        objective_score=solver.ObjectiveValue(),
        time_to_solve=solver.WallTime(),
        number_of_shifts_per_employee=shifts_per_employee,
        score_per_employee=score_per_employee,
        score_variance=score_variance,
        min_employee_score=solver.Value(min_employee_score),
        max_employee_score=solver.Value(max_employee_score),
        total_possible_shifts=total_possible_shifts,
        total_shifts_filled=sum(shifts_per_employee.values()),
        employee_schedule=employee_schedule,
        shift_schedule=shift_schedule,
    )
    return schedule


def compute_optimal_bench_map(
    schedule: Schedule,
    all_employees: Dict[int, Employee],
    all_shifts: Dict[int, Shift],
    all_labs: Dict[str, Lab],
    all_benches: Dict[str, Bench],
    constraints: Constraints,
) -> Schedule:
    # Make sure employees have home_bench information
    for employee in all_employees:
        assert all_employees[employee].home_bench is not None

    model = CpModel()

    bench_placement = {}
    for shift in schedule.shift_schedule:
        for employee in schedule.shift_schedule[shift]:
            for bench in all_benches:
                bench_placement[
                    (shift, employee.index, bench)
                ] = model.NewBoolVar(
                    f"bench_s{shift}e{employee.index}b{bench}"
                )

    # 2 employees cannot have same bench
    for shift in schedule.shift_schedule:
        for bench in all_benches:
            model.Add(
                sum(
                    bench_placement[(shift, employee.index, bench)]
                    for employee in schedule.shift_schedule[shift]
                )
                <= 1
            )

    # All employees scheduled must have a bench for each shift
    # Other employees must not have a bench
    for shift in schedule.shift_schedule:
        for employee in schedule.shift_schedule[shift]:
            model.Add(
                sum(
                    bench_placement[(shift, employee.index, bench)]
                    for bench in all_benches
                )
                == 1
            )

    # Make sure appropriate benches are utilized during appropriate shift
    for shift in schedule.shift_schedule:
        for bench in all_benches:
            if (all_shifts[shift].label == "FULL") and (
                all_benches[bench].active_shift == "AM"
            ):
                # This works because we want FULL shifts to use AM benches
                # If proper condition, no constraint will be added
                # If anything else, it will be added
                # For example, FULL + PM will have constraint == 0
                pass  # important
            elif (
                all_shifts[shift].label != all_benches[bench].active_shift
            ):
                model.Add(
                    sum(
                        bench_placement[(shift, employee.index, bench)]
                        for employee in schedule.shift_schedule[shift]
                    )
                    == 0
                )

    # Create objective
    # Minimize distance from home bench to placed bench
    model.Minimize(
        sum(
            (
                (
                    employee.home_bench.location.x  # type: ignore
                    - all_benches[bench].location.x
                )
                ** 2
                + (
                    employee.home_bench.location.y  # type: ignore
                    - all_benches[bench].location.y
                )
                ** 2
                + (
                    constraints.raw_lab_separation.loc[  # type: ignore
                        employee.home_bench.lab.name,  # type: ignore
                        all_benches[bench].lab.name,
                    ]
                )
            )
            * bench_placement[(shift, employee.index, bench)]
            for shift in schedule.shift_schedule
            for employee in schedule.shift_schedule[shift]
            for bench in all_benches
        )
    )

    st.write("Running Location Optimization...")
    solver = CpSolver()
    solver.parameters.max_time_in_seconds = 60
    solver.Solve(model)
    st.write("Finished Location Optimization!")

    shift_bench_schedule: Dict[
        int, Dict[int, Bench]
    ] = {}  # {Shift.id: {Employee.id: Bench}}
    shift_bench_schedule = {
        shift: {
            employee.index: all_benches[bench]
            for employee in schedule.shift_schedule[shift]
            for bench in all_benches
            if solver.Value(
                bench_placement[(shift, employee.index, bench)]
            )
        }
        for shift in all_shifts
    }
    employee_bench_schedule: Dict[
        int, Dict[int, Bench]
    ] = {}  # {Employee.id: {Shift.id: Bench}}
    for shift in shift_bench_schedule:
        for employee in shift_bench_schedule[shift]:
            employee_bench_schedule[employee] = {}
    for shift in shift_bench_schedule:
        for employee in shift_bench_schedule[shift]:
            employee_bench_schedule[employee][
                shift
            ] = shift_bench_schedule[shift][employee]

    employee_bench_distances: Dict[
        int, float
    ] = {}  # {Employee.id: average distance}
    for employee in employee_bench_schedule:
        employee_bench_distances[employee] = statistics.mean(
            [
                (
                    (
                        employee_bench_schedule[employee][shift].location.x
                        - all_employees[employee].home_bench.location.x
                    )
                    ** 2
                    + (
                        employee_bench_schedule[employee][shift].location.y
                        - all_employees[employee].home_bench.location.y
                    )
                    ** 2
                )
                ** 0.5
                for shift in employee_bench_schedule[employee]
            ]
        )

    schedule.shift_bench_schedule = shift_bench_schedule
    schedule.employee_bench_schedule = employee_bench_schedule
    schedule.bench_objective_score = solver.ObjectiveValue()
    schedule.bench_optimization_average_distance = employee_bench_distances
    return schedule


def create_optimal_schedule(
    constraints: Constraints,
    roster: pd.DataFrame,
    parameters: ConstraintParameters,
) -> Schedule:
    days: Dict[str, Day] = {}
    constraints.slot_constraints.apply(
        lambda row: create_day(obj_in=row, db=days), axis=1
    )

    employees: Dict[int, Employee] = {}
    roster.apply(
        lambda row: create_employee(obj_in=row, db=employees, days=days),
        axis=1,
    )

    shifts: Dict[int, Shift] = {}
    for day in days:
        for shift in days[day].shifts:
            shifts[days[day].shifts[shift].index] = days[day].shifts[shift]

    optimal_schedule: Schedule = compute_optimal_schedule(
        all_employees=employees,
        all_shifts=shifts,
        all_days=days,
        schedule_parameters=parameters,
    )

    if parameters.include_location_optimization:
        assert constraints.raw_bench_constraints is not None
        assert constraints.raw_lab_maps is not None
        # Create labs
        labs: Dict[str, Lab] = {}
        constraints.raw_bench_constraints.apply(
            lambda row: create_lab(obj_in=row, db=labs), axis=1
        )

        # Create benches
        benches: Dict[str, Bench] = {}
        for lab in constraints.raw_lab_maps:
            constraints.raw_lab_maps[lab].apply(
                lambda row: pd.DataFrame(row).apply(
                    lambda col: create_bench(
                        name=col.values[0],
                        lab=labs[lab],
                        location=Location(x=row.name, y=col.name),
                        db=benches,
                        active_shift=find_active_shift(
                            constraints=constraints,
                            lab=labs[lab],
                            bench_name=col.values[0],
                        ),
                    ),
                    axis=1,
                ),
                axis=1,
            )

        # Update Employee Home Benches
        for employee in employees:
            bench_name: str = roster.loc[
                roster["id"] == employees[employee].id, "home_bench"
            ].values[0]
            employees[employee].home_bench = benches[bench_name]

        optimal_schedule = compute_optimal_bench_map(
            schedule=optimal_schedule,
            all_employees=employees,
            all_shifts=shifts,
            all_labs=labs,
            all_benches=benches,
            constraints=constraints,
        )

    optimal_schedule.pretty_employee_schedule = prettify_schedule(
        schedule=optimal_schedule,
        all_employees=employees,
        all_shifts=shifts,
        all_days=days,
    )

    return optimal_schedule


# =======================================================================#
# Main App
# =======================================================================#


def app():
    page = st.sidebar.radio(
        label="",
        options=["Play with demo", "Try it yourself!"],
    )

    st.sidebar.header("Optimization Settings")
    include_location_optimization = st.sidebar.checkbox(
        label="Include location optimization", value=True
    )
    fairness_weight = st.sidebar.slider(
        label="Fairness Weight", min_value=0, max_value=50, value=0, step=1
    )
    fill_schedule_weight = st.sidebar.slider(
        label="Fill Schedule Weight",
        min_value=0,
        max_value=50,
        value=5,
        step=1,
    )
    max_unfairness = st.sidebar.slider(
        label="Max Unfairness (Max Employee Score - Min Employee Score)",
        min_value=-1,
        max_value=50,
        value=-1,
        step=1,
    )
    parameters: ConstraintParameters = ConstraintParameters(
        max_unfairness=max_unfairness,
        fairness_weight=fairness_weight,
        fill_schedule_weight=fill_schedule_weight,
        include_location_optimization=include_location_optimization,
    )

    st.title("Scheduling App")
    st.subheader("Optimized using Google OR-Tools in Python")

    constraints: Constraints = Constraints(
        raw_slot_constraints=pd.DataFrame(),
        slot_constraints=pd.DataFrame(),
    )
    roster: pd.DataFrame = pd.DataFrame()

    if page == "Play with demo":
        constraints = read_constraints(
            constraints_xlsx="demo_constraints.xlsx",
            parameters=parameters,
        )
        roster = read_roster("demo_roster.csv")
    else:
        st.markdown(
            get_excel_download_link(
                excel_file="demo_constraints.xlsx",
                label="Download Example Constraints",
                output_filename="demo_constraints.xlsx",
            ),
            unsafe_allow_html=True,
        )
        constraints_file = st.file_uploader(
            label="Constraints File Upload", type="xlsx"
        )
        st.markdown(
            get_table_download_link(
                df=pd.read_csv("demo_roster.csv"),
                label="Download Example Roster",
                output_filename="demo_roster.csv",
                index=False,
            ),
            unsafe_allow_html=True,
        )
        roster_file = st.file_uploader(
            label="Roster File Upload", type="csv"
        )
        if constraints_file is not None:
            constraints_file.seek(0)
            constraints = read_constraints(
                constraints_xlsx=constraints_file,
                parameters=parameters,
            )
        if roster_file is not None:
            roster_file.seek(0)
            roster = read_roster(roster_file)

    if not constraints.slot_constraints.empty and not roster.empty:
        optimal_schedule: Schedule = create_optimal_schedule(
            constraints=constraints,
            roster=roster,
            parameters=parameters,
        )
        st.header("Schedule Constraints")
        st.write(constraints.raw_slot_constraints)

        st.header("Roster")
        st.write(roster)

        if parameters.include_location_optimization:
            st.header("Bench Constraints")
            st.write(constraints.raw_bench_constraints)
            st.header("Lab Separation")
            st.write(constraints.raw_lab_separation)
            for lab in constraints.raw_lab_maps:
                st.header(f"{lab}")
                st.write(constraints.raw_lab_maps[lab])

        demand_heatmap = create_demand_heatmap(roster=roster)
        st.plotly_chart(demand_heatmap)

        st.header("Optimal Schedule")
        st.write(
            f"Shift Optimization Score: {optimal_schedule.objective_score}"
        )
        st.write(
            f"Shifts Filled: {optimal_schedule.total_shifts_filled} "
            f"out of {optimal_schedule.total_possible_shifts}"
        )
        if parameters.include_location_optimization:
            st.write(
                f"Bench Optimization Score"
                f": {optimal_schedule.bench_objective_score}"
            )
            average_bench_distance: float = statistics.mean(
                optimal_schedule.bench_optimization_average_distance.values()
            )
            st.write(
                f"Average Distance b/w home bench and shift bench: "
                f"{average_bench_distance}"
            )
        st.write(optimal_schedule.pretty_employee_schedule)
        st.markdown(
            get_table_download_link(
                df=optimal_schedule.pretty_employee_schedule,
                label="Download Optimal Schedule",
                output_filename="optimal_employee_schedule.csv",
            ),
            unsafe_allow_html=True,
        )
        score_boxplot = create_score_boxplot(
            optimal_schedule=optimal_schedule.pretty_employee_schedule
        )
        st.plotly_chart(score_boxplot)
    else:
        st.write("Please upload constraints and roster")


if __name__ == "__main__":
    app()
