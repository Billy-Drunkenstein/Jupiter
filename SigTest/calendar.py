from pathlib import Path


def load_calendar(calendar_path: Path):
    calendar_path = Path(calendar_path)
    if not calendar_path.exists():
        raise FileNotFoundError(f"Calendar file not found: {calendar_path}")

    dates = []
    with open(calendar_path, "r") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
          
            if not line or line.startswith("#"):
                continue
            try:
                d = int(line)
            except ValueError:
                raise ValueError(
                    f"Non-integer entry on line {line_num}: '{raw_line.rstrip()}'"
                )
            if d < 10000000 or d > 99999999:
                raise ValueError(
                    f"Date on line {line_num} is not 8 digits: {d}"
                )
            dates.append(d)

    if not dates:
        raise ValueError(f"Calendar file is empty: {calendar_path}")

    if len(dates) != len(set(dates)):
        seen = set()
        for d in dates:
            if d in seen:
                raise ValueError(f"Duplicate date in calendar: {d}")
            seen.add(d)

    dates.sort()
    return dates

def assemble_dates(
    calendar: list[int],
    asof: int,
    window_size: int,
    direction: str,
):
    if direction not in ("IS", "OOS"):
        raise ValueError(f"direction must be 'IS' or 'OOS', got '{direction}'")

    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    try:
        idx = calendar.index(asof)
    except ValueError:
        raise ValueError(f"asof date {asof} not found in calendar")

    if direction == "IS":
        start = idx - window_size + 1
      
        if start < 0:
            available = idx + 1
            raise ValueError(
                f"Insufficient history: need {window_size} dates, "
                f"only {available} available before and including {asof}"
            )
          
        return calendar[start : idx + 1]

    else:  # OOS
        oos_start = idx + 1
        oos_end = oos_start + window_size
      
        if oos_end > len(calendar):
            available = len(calendar) - oos_start
            raise ValueError(
                f"Insufficient future: need {window_size} dates, "
                f"only {available} available after {asof}"
            )
          
        return calendar[oos_start : oos_end]
