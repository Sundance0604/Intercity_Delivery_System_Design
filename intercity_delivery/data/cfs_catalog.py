"""Read-only catalog helpers for a CFS SQLite shipment store.

The GUI uses this module to inspect the database schema and enumerate the
bidirectional CFS-area pairs before an experiment starts.  Area descriptions
come from Table A2 of the official 2022 CFS PUMS User's Guide.  Unknown codes
remain usable and are displayed verbatim.
"""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from intercity_delivery.data.sqlite_store import TABLE_NAME, is_sqlite_path


# Table A2, 2022 CFS PUMS User's Guide.  Remainder-of-state labels are produced
# from STATE_NAMES below so the static mapping stays compact.
CFS_AREA_NAMES: Dict[str, str] = {
    "36-104": "Albany-Schenectady, NY",
    "13-122": "Atlanta-Athens-Clarke County-Sandy Springs, GA-AL (GA Part)",
    "01-142": "Birmingham-Hoover-Talladega, AL",
    "25-148": "Boston-Worcester-Providence, MA-RI-NH-CT (MA Part)",
    "33-148": "Boston-Worcester-Providence, MA-RI-NH-CT (NH Part)",
    "44-148": "Boston-Worcester-Providence, MA-RI-NH-CT (RI Part)",
    "36-160": "Buffalo-Cheektowaga-Olean, NY",
    "37-172": "Charlotte-Concord, NC-SC (NC Part)",
    "17-176": "Chicago-Naperville, IL-IN-WI (IL Part)",
    "18-176": "Chicago-Naperville, IL-IN-WI (IN Part)",
    "21-178": "Cincinnati-Wilmington-Maysville, OH-KY-IN (KY Part)",
    "39-178": "Cincinnati-Wilmington-Maysville, OH-KY-IN (OH Part)",
    "39-184": "Cleveland-Akron-Canton, OH",
    "39-198": "Columbus-Marion-Zanesville, OH",
    "48-204": "Corpus Christi-Kingsville-Alice, TX",
    "48-206": "Dallas-Fort Worth, TX-OK (TX Part)",
    "39-212": "Dayton-Springfield-Kettering, OH",
    "08-216": "Denver-Aurora, CO",
    "19-218": "Des Moines-Ames-West Des Moines, IA",
    "26-220": "Detroit-Warren-Ann Arbor, MI",
    "48-238": "El Paso-Las Cruces, TX-NM (TX Part)",
    "18-258": "Fort Wayne-Huntington-Auburn, IN",
    "06-260": "Fresno-Madera-Hanford, CA",
    "26-266": "Grand Rapids-Kentwood-Muskegon, MI",
    "37-268": "Greensboro-Winston-Salem-High Point, NC",
    "45-273": "Greenville-Spartanburg-Anderson, SC",
    "48-288": "Houston-The Woodlands, TX",
    "18-294": "Indianapolis-Carmel-Muncie, IN",
    "12-300": "Jacksonville-St. Marys-Palatka, FL-GA (FL Part)",
    "20-312": "Kansas City-Overland Park-Kansas City, MO-KS (KS Part)",
    "29-312": "Kansas City-Overland Park-Kansas City, MO-KS (MO Part)",
    "47-315": "Knoxville-Morristown-Sevierville, TN",
    "22-324": "Lake Charles-Jennings, LA",
    "32-332": "Las Vegas-Henderson, NV",
    "06-348": "Los Angeles-Long Beach, CA",
    "21-350": "Louisville/Jefferson County-Elizabethtown-Bardstown, KY-IN (KY Part)",
    "47-368": "Memphis-Forrest City, TN-MS-AR (TN Part)",
    "12-370": "Miami-Port St. Lucie-Fort Lauderdale, FL",
    "55-376": "Milwaukee-Racine-Waukesha, WI",
    "27-378": "Minneapolis-St. Paul, MN-WI (MN Part)",
    "01-380": "Mobile-Daphne-Fairhope, AL",
    "47-400": "Nashville-Davidson-Murfreesboro, TN",
    "22-406": "New Orleans-Metairie-Hammond, LA-MS (LA Part)",
    "09-408": "New York-Newark, NY-NJ-CT-PA (CT Part)",
    "34-408": "New York-Newark, NY-NJ-CT-PA (NJ Part)",
    "36-408": "New York-Newark, NY-NJ-CT-PA (NY Part)",
    "40-416": "Oklahoma City-Shawnee, OK",
    "31-420": "Omaha-Council Bluffs-Fremont, NE-IA (NE Part)",
    "12-422": "Orlando-Lakeland-Deltona, FL",
    "10-428": "Philadelphia-Reading-Camden, PA-NJ-DE-MD (DE Part)",
    "34-428": "Philadelphia-Reading-Camden, PA-NJ-DE-MD (NJ Part)",
    "42-428": "Philadelphia-Reading-Camden, PA-NJ-DE-MD (PA Part)",
    "04-429": "Phoenix-Mesa, AZ",
    "42-430": "Pittsburgh-New Castle-Weirton, PA-OH-WV (PA Part)",
    "41-440": "Portland-Vancouver-Salem, OR-WA (OR Part)",
    "53-440": "Portland-Vancouver-Salem, OR-WA (WA Part)",
    "37-450": "Raleigh-Durham-Cary, NC",
    "36-464": "Rochester-Batavia-Seneca Falls, NY",
    "06-472": "Sacramento-Roseville, CA",
    "17-476": "St. Louis-St. Charles-Farmington, MO-IL (IL Part)",
    "29-476": "St. Louis-St. Charles-Farmington, MO-IL (MO Part)",
    "49-482": "Salt Lake City-Provo-Orem, UT",
    "48-484": "San Antonio-New Braunfels-Pearsall, TX",
    "06-488": "San Jose-San Francisco-Oakland, CA",
    "13-496": "Savannah-Hinesville-Statesboro, GA",
    "53-500": "Seattle-Tacoma, WA",
    "39-534": "Toledo-Findlay-Tiffin, OH",
    "04-536": "Tucson-Nogales, AZ",
    "40-538": "Tulsa-Muskogee-Bartlesville, OK",
    "51-545": "Virginia Beach-Norfolk, VA-NC (VA Part)",
    "20-556": "Wichita-Winfield, KS",
    "42-10900": "Allentown-Bethlehem-Easton, PA-NJ (PA Part)",
    "48-12420": "Austin-Round Rock-Georgetown, TX",
    "24-12580": "Baltimore-Columbia-Towson, MD",
    "22-12940": "Baton Rouge, LA",
    "48-13140": "Beaumont-Port Arthur, TX",
    "45-16700": "Charleston-North Charleston, SC",
    "09-25540": "Hartford-East Hartford-Middletown, CT",
    "48-29700": "Laredo, TX",
    "51-40060": "Richmond, VA",
    "06-41740": "San Diego-Chula Vista-Carlsbad, CA",
    "12-45300": "Tampa-St. Petersburg-Clearwater, FL",
    "15-46520": "Urban Honolulu, HI",
    "11-47900": "Washington-Arlington-Alexandria, DC-VA-MD-WV (DC Part)",
    "24-47900": "Washington-Arlington-Alexandria, DC-VA-MD-WV (MD Part)",
    "51-47900": "Washington-Arlington-Alexandria, DC-VA-MD-WV (VA Part)",
}

STATE_NAMES = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
    "06": "California", "08": "Colorado", "09": "Connecticut", "10": "Delaware",
    "11": "District of Columbia", "12": "Florida", "13": "Georgia", "15": "Hawaii",
    "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa",
    "20": "Kansas", "21": "Kentucky", "22": "Louisiana", "23": "Maine",
    "24": "Maryland", "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
    "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska",
    "32": "Nevada", "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico",
    "36": "New York", "37": "North Carolina", "38": "North Dakota", "39": "Ohio",
    "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania", "44": "Rhode Island",
    "45": "South Carolina", "46": "South Dakota", "47": "Tennessee", "48": "Texas",
    "49": "Utah", "50": "Vermont", "51": "Virginia", "53": "Washington",
    "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming",
}


def cfs_area_name(code: str) -> str:
    """Return the official area description without the repeated CFS suffix."""

    code = str(code).strip()
    if code in CFS_AREA_NAMES:
        return CFS_AREA_NAMES[code]
    if code.endswith("-99999"):
        state = STATE_NAMES.get(code.split("-", 1)[0])
        if state:
            return f"Remainder of {state}"
    if code == "00-00000":
        return "Origin completely suppressed"
    if code.endswith("-00000"):
        return f"Origin metro area suppressed ({code[:2]})"
    return code


def cfs_area_filename_label(code: str) -> str:
    """Return a compact, Windows-safe area label for result filenames."""

    name = cfs_area_name(code)
    name = re.sub(r"\s*\([^)]*Part\)\s*$", "", name)
    name = name.split(",", 1)[0]
    fragment = re.sub(r"[^A-Za-z0-9-]+", "_", name).strip("_-")
    return fragment or re.sub(r"[^A-Za-z0-9-]+", "_", code).strip("_")


@dataclass(frozen=True)
class CFSCityPair:
    city_a: str
    city_b: str
    records_a_to_b: int
    records_b_to_a: int

    @property
    def display_label(self) -> str:
        return (
            f"{cfs_area_name(self.city_a)} [{self.city_a}]  ↔  "
            f"{cfs_area_name(self.city_b)} [{self.city_b}]  "
            f"({self.records_a_to_b:,}/{self.records_b_to_a:,})"
        )

    @property
    def filename_label(self) -> str:
        return (
            f"{cfs_area_filename_label(self.city_a)}_to_"
            f"{cfs_area_filename_label(self.city_b)}"
        )


@dataclass(frozen=True)
class CFSSQLiteCatalog:
    database_path: str
    columns: Tuple[Tuple[str, str], ...]
    city_pairs: Tuple[CFSCityPair, ...]


def inspect_cfs_sqlite(
    path: str | Path,
    *,
    min_records_per_direction: int = 1,
    metro_only: bool = True,
) -> CFSSQLiteCatalog:
    """Inspect columns and enumerate bidirectional OD pairs without loading rows."""

    database = Path(path).expanduser()
    if not database.is_file():
        raise FileNotFoundError(f"找不到 SQLite 文件：{database}")
    if not is_sqlite_path(database):
        raise ValueError("真实数据文件必须是 .sqlite、.sqlite3 或 .db。")
    if min_records_per_direction <= 0:
        raise ValueError("min_records_per_direction 必须大于 0。")

    uri = f"file:{database.resolve()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (TABLE_NAME,),
        ).fetchone()
        if table is None:
            raise ValueError(f"SQLite 中不存在表 {TABLE_NAME!r}。")
        columns = tuple(
            (str(row[1]), str(row[2] or ""))
            for row in connection.execute(f'PRAGMA table_info("{TABLE_NAME}")')
        )
        if not {"ORIG_CFS_AREA", "DEST_CFS_AREA"}.issubset(
            name for name, _type in columns
        ):
            raise ValueError("shipments 表缺少 ORIG_CFS_AREA 或 DEST_CFS_AREA。")

        directed = connection.execute(
            f'SELECT ORIG_CFS_AREA, DEST_CFS_AREA, COUNT(*) '
            f'FROM "{TABLE_NAME}" INDEXED BY idx_shipments_od '
            'WHERE ORIG_CFS_AREA IS NOT NULL AND DEST_CFS_AREA IS NOT NULL '
            'AND ORIG_CFS_AREA <> DEST_CFS_AREA '
            'GROUP BY ORIG_CFS_AREA, DEST_CFS_AREA'
        )
        counts: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0])
        for origin, destination, count in directed:
            origin = str(origin).strip()
            destination = str(destination).strip()
            if metro_only and (
                origin.endswith("-99999")
                or destination.endswith("-99999")
                or origin.endswith("-00000")
                or destination.endswith("-00000")
            ):
                continue
            if origin < destination:
                key, direction = (origin, destination), 0
            else:
                key, direction = (destination, origin), 1
            counts[key][direction] += int(count)

    pairs = [
        CFSCityPair(city_a, city_b, values[0], values[1])
        for (city_a, city_b), values in counts.items()
        if min(values) >= min_records_per_direction
    ]
    pairs.sort(
        key=lambda item: (
            -min(item.records_a_to_b, item.records_b_to_a),
            item.city_a,
            item.city_b,
        )
    )
    return CFSSQLiteCatalog(
        database_path=str(database.resolve()),
        columns=columns,
        city_pairs=tuple(pairs),
    )


def find_city_pair(
    catalog: CFSSQLiteCatalog, city_a: str, city_b: str
) -> CFSCityPair:
    """Find a pair regardless of which direction the caller supplied."""

    target = tuple(sorted((str(city_a).strip(), str(city_b).strip())))
    for pair in catalog.city_pairs:
        if (pair.city_a, pair.city_b) == target:
            return pair
    raise ValueError(f"SQLite 中不存在双向城市对 {target[0]} ↔ {target[1]}。")
