import sys
import os
import math
import json
import threading
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, cast, List, Tuple, TypedDict

try:
    import ctypes

    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Orica.LeachIT.ExtractUpload")
except Exception:
    pass

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
import pandas as pd
import yaml
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class AuthTokens(TypedDict):
    access_token: str
    refresh_token: str
    token_type: str

# =========================
# Resources / packaging
# =========================
def app_base_dir() -> Path:
    """
    Return the directory the app should use for external support files.
    - In normal Python runs: the script folder
    - In PyInstaller builds: the folder containing the executable
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def resource_path(rel_path: str) -> Path:
    """
    Prefer external files beside the script/executable.
    Fall back to PyInstaller's internal bundle location if needed.
    """
    external = app_base_dir() / rel_path
    if external.exists():
        return external

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        bundled = Path(meipass) / rel_path
        if bundled.exists():
            return bundled

    return external


APP_NAME = "LeachIT"
APP_DIR = Path(os.getenv("APPDATA", str(Path.home()))) / APP_NAME
USER_SETTINGS = APP_DIR / "settings.yaml"
DEFAULT_SETTINGS = resource_path("settings_default.yaml")
LOGO_FILE = resource_path("Leachit.png")
ICON_FILE = resource_path("favicon.ico")


def _yaml_load(p: Path) -> dict[str, Any]:
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return cast(dict[str, Any], loaded or {})


def load_settings() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Returns (merged, defaults, user). Only ui_defaults are merged (shallow).
    """
    defaults = _yaml_load(DEFAULT_SETTINGS)
    user = _yaml_load(USER_SETTINGS)

    merged = deepcopy(defaults)
    d_ui = cast(dict[str, Any], defaults.get("ui_defaults", {}) or {})
    u_ui = cast(dict[str, Any], user.get("ui_defaults", {}) or {})
    merged["ui_defaults"] = {**d_ui, **u_ui}
    return merged, defaults, user


def save_user_settings(new_ui_defaults: dict[str, Any]) -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    with open(USER_SETTINGS, "w", encoding="utf-8") as f:
        yaml.safe_dump({"ui_defaults": new_ui_defaults}, f, sort_keys=False, allow_unicode=True)


def reset_user_settings() -> None:
    try:
        if USER_SETTINGS.exists():
            USER_SETTINGS.unlink()
    except Exception:
        pass


def load_config_text() -> str:
    if not DEFAULT_SETTINGS.exists():
        raise FileNotFoundError(
            f"Required config file not found: {DEFAULT_SETTINGS}\n"
            "Please keep settings_default.yaml beside the executable."
        )
    with open(DEFAULT_SETTINGS, "r", encoding="utf-8") as f:
        return f.read()


# =========================
# Excel extraction helpers
# =========================
def excel_col_to_idx(col_letter: str) -> int:
    col_letter = col_letter.strip().upper()
    n = 0
    for c in col_letter:
        n = n * 26 + (ord(c) - ord("A") + 1)
    return n - 1  # 0-based


def expand_ranges(rows_spec: Sequence[int | Sequence[int]]) -> list[int]:
    """Accepts [ints or [start,end]] -> explicit 1-based ints."""
    out: list[int] = []
    for item in rows_spec:
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2:
            start = int(item[0])
            end = int(item[1])
            out.extend(range(start, end + 1))
        elif isinstance(item, int):
            out.append(item)
        else:
            raise TypeError(f"Invalid rows_spec item: {item}")
    return out


def read_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")


def _collect_fields_from_rows(
    df: pd.DataFrame,
    rows_1based: List[int],
    label_overrides: Optional[Dict[int, str]],
    fallback_prefix: str,
) -> List[Tuple[str, int]]:
    fields: List[Tuple[str, int]] = []
    for r in rows_1based:
        row_idx0 = r - 1
        raw_label: Any = df.iat[row_idx0, 1]

        if raw_label is None:
            label = f"{fallback_prefix}{r}"
        elif isinstance(raw_label, float) and pd.isna(raw_label):
            label = f"{fallback_prefix}{r}"
        else:
            label = str(raw_label).strip()
            if label == "":
                label = f"{fallback_prefix}{r}"

        if label_overrides and r in label_overrides:
            label = label_overrides[r].strip()

        fields.append((label, r))
    return fields


def _collect_fields_from_explicit(explicit_fields: Sequence[Mapping[str, Any]]) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for item in explicit_fields:
        r = int(item["row"])
        name = str(item["name"]).strip()
        out.append((name, r))
    return out


def _dedupe_labels(labels: Sequence[str], suffix: str) -> list[str]:
    seen: dict[str, int] = {}
    final: list[str] = []
    for lab in labels:
        if lab not in seen:
            seen[lab] = 1
            final.append(lab)
        else:
            seen[lab] += 1
            final.append(f"{lab}{suffix}")
    return final


def _extract_sheet_columns(
    df: pd.DataFrame,
    first_data_col_letter: str,
    n_days: int,
    fields_lr: Sequence[tuple[str, int]],
) -> pd.DataFrame:
    """
    Vectorised extraction (prevents fragmentation).
    """
    start_col_idx = excel_col_to_idx(first_data_col_letter)
    row_idx0 = [r - 1 for _, r in fields_lr]
    labels = [lab for lab, _ in fields_lr]

    block = df.iloc[row_idx0, start_col_idx : start_col_idx + n_days]
    arr = block.to_numpy().T

    out = pd.DataFrame(arr, columns=labels).reset_index(drop=True)
    if len(out) != n_days:
        out = out.reindex(range(n_days)).reset_index(drop=True)
    return out


def build_extract(xlsx_path: Path, config_yaml_text: str) -> pd.DataFrame:
    cfg = cast(dict[str, Any], yaml.safe_load(config_yaml_text))
    sheets = cast(dict[str, Any], cfg["sheets"])
    sh_input = cast(dict[str, Any], sheets["input"])
    sh_calcs = cast(dict[str, Any], sheets["calcs"])

    # Load sheets
    df_input = read_sheet(xlsx_path, str(sh_input["name"]))
    df_calcs = read_sheet(xlsx_path, str(sh_calcs["name"]))

    # Dates (from Input sheet)
    date_row_idx0 = int(sh_input["date_row"]) - 1
    start_col_idx = excel_col_to_idx(str(sh_input["first_data_col"]))
    date_series = df_input.iloc[date_row_idx0, start_col_idx:]
    date_series = date_series.dropna(how="all")
    dates = pd.to_datetime(date_series, errors="coerce")
    n_days = len(dates)

    # ----- INPUT fields -----
    input_fields_cfg = sh_input.get("fields")
    input_label_overrides = {int(k): str(v) for k, v in cast(dict[Any, Any], sh_input.get("label_overrides") or {}).items()}
    if input_fields_cfg:
        input_fields = _collect_fields_from_explicit(cast(Sequence[Mapping[str, Any]], input_fields_cfg))
    else:
        rows_spec = cast(Sequence[int | Sequence[int]], sh_input["rows"])
        input_rows = expand_ranges(rows_spec)
        input_fields = _collect_fields_from_rows(
            df_input,
            input_rows,
            input_label_overrides,
            str(cfg["fallback_label_prefix"]),
        )

    # ----- CALCS fields -----
    calcs_fields_cfg = sh_calcs.get("fields")
    calcs_label_overrides = {int(k): str(v) for k, v in cast(dict[Any, Any], sh_calcs.get("label_overrides") or {}).items()}
    if calcs_fields_cfg:
        calcs_fields = _collect_fields_from_explicit(cast(Sequence[Mapping[str, Any]], calcs_fields_cfg))
    else:
        rows_spec = cast(Sequence[int | Sequence[int]], sh_calcs["rows"])
        calcs_rows = expand_ranges(rows_spec)
        calcs_fields = _collect_fields_from_rows(
            df_calcs,
            calcs_rows,
            calcs_label_overrides,
            str(cfg["fallback_label_prefix"]),
        )

    # Dedupe across both sheets
    all_labels = [lab for lab, _ in input_fields] + [lab for lab, _ in calcs_fields]
    deduped = _dedupe_labels(all_labels, str(cfg.get("dedupe_suffix", "_dup")))
    dedup_input = deduped[: len(input_fields)]
    dedup_calcs = deduped[len(input_fields) :]

    out = pd.DataFrame({"Date": dates.values})
    df_input_vals = _extract_sheet_columns(
        df_input,
        str(sh_input["first_data_col"]),
        n_days,
        list(zip(dedup_input, [r for _, r in input_fields])),
    )
    df_calcs_vals = _extract_sheet_columns(
        df_calcs,
        str(sh_calcs["first_data_col"]),
        n_days,
        list(zip(dedup_calcs, [r for _, r in calcs_fields])),
    )
    out = pd.concat([out, df_input_vals, df_calcs_vals], axis=1).copy()
    return out

# =========================
# Login and Refresh helpers
# =========================

def login_user(session: requests.Session, base: str, email: str, password: str) -> AuthTokens:
    r = session.post(
        f"{base}/auth/login",
        json={"email": email, "password": password},
        timeout=60,
    )
    if not r.ok:
        try:
            msg = r.json()
        except Exception:
            msg = r.text
        raise RuntimeError(f"Login failed {r.status_code}: {msg}")
    data = cast(dict[str, Any], r.json())
    return {
        "access_token": str(data["access_token"]),
        "refresh_token": str(data["refresh_token"]),
        "token_type": str(data.get("token_type", "bearer")),
    }


def refresh_login(session: requests.Session, base: str, refresh_token: str) -> AuthTokens:
    r = session.post(
        f"{base}/auth/refresh",
        json={"refresh_token": refresh_token},
        timeout=60,
    )
    if not r.ok:
        try:
            msg = r.json()
        except Exception:
            msg = r.text
        raise RuntimeError(f"Refresh failed {r.status_code}: {msg}")
    data = cast(dict[str, Any], r.json())
    return {
        "access_token": str(data["access_token"]),
        "refresh_token": str(data["refresh_token"]),
        "token_type": str(data.get("token_type", "bearer")),
    }


# =========================
# Upload / derivation logic
# =========================
REQUIRED = ["date", "cn", "do", "grade", "percent_solids", "throughput", "coarse", "fines", "ultrafines"]


def json_safe(v: Any) -> Any:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, (int, bool)):
        return v
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.date().isoformat()
    if isinstance(v, np.datetime64):
        try:
            return pd.to_datetime(v).date().isoformat()
        except Exception:
            return None
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in {"nan", "na", "none", ""}:
            return None
        return s
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): json_safe(val) for k, val in v.items()}
    return str(v)


def df_to_json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    obj = df.copy()
    if "date" in obj.columns:
        obj["date"] = pd.to_datetime(obj["date"], errors="coerce")
    obj = obj.astype(object)
    records_raw = obj.to_dict(orient="records")
    records = cast(list[dict[Any, Any]], records_raw)
    return [{str(k): json_safe(v) for k, v in row.items()} for row in records]


def clean_like_server(name: str) -> str:
    s = name.strip()
    s = s.replace("%", "pct").replace("/", " ").replace("-", " ")
    s = s.replace("(", " ").replace(")", " ")
    s = "_".join(s.split()).lower()
    return s


def cleanse_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out[out["date"].notna()].copy()

    # Only operate on numeric columns to avoid future downcast behaviour changes
    num_cols = out.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        out.loc[:, num_cols] = out.loc[:, num_cols].replace([np.inf, -np.inf], np.nan)

    return out


def _series_of_nan(index: pd.Index) -> pd.Series:
    return pd.Series(np.nan, index=index, dtype="float64")


def _coerce_series(value: Any, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.reindex(index)
    if isinstance(value, np.ndarray):
        return pd.Series(value, index=index)
    if isinstance(value, list):
        return pd.Series(value, index=index[: len(value)])
    return _series_of_nan(index)


def prefer_first(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                return s
    return _series_of_nan(df.index)


def normalise_triplet(coarse: pd.Series, fines: pd.Series, ultra: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    trip = pd.concat([coarse, fines, ultra], axis=1)
    trip.columns = ["coarse", "fines", "ultrafines"]
    for col in trip.columns:
        trip[col] = pd.to_numeric(trip[col], errors="coerce").clip(lower=0)
    s = trip.sum(axis=1)
    mask = s.between(95, 105)
    trip.loc[mask, ["coarse", "fines", "ultrafines"]] = (
        trip.loc[mask, ["coarse", "fines", "ultrafines"]].div(s[mask], axis=0) * 100.0
    )
    return trip["coarse"], trip["fines"], trip["ultrafines"]


def derive_required_fields(df: pd.DataFrame) -> pd.DataFrame:
    g = df
    out = pd.DataFrame(index=g.index)

    date_input = _coerce_series(g["date"], g.index) if "date" in g.columns else _series_of_nan(g.index)
    out["date"] = pd.to_datetime(date_input, errors="coerce")

    percent_solids_input = g["percent_solids"] if "percent_solids" in g.columns else _series_of_nan(g.index)
    out["percent_solids"] = pd.to_numeric(percent_solids_input, errors="coerce")

    out["cn"] = prefer_first(
        g,
        [
            "avg_free_cn_tank_ppm",
            "avg_free_cn_leach_ppm",
            "free_cn_leach_tank_1_ppm",
            "cyanide_profile_ppm_leach_tank_1",
            "free_cn_leach_tank_2_ppm",
        ],
    )

    out["do"] = prefer_first(
        g,
        [
            "avg_do_tank_ppm",
            "do_leach_tank_1_ppm_imputed",
            "do_leach_tank_1_ppm",
            "do_leach_tank_2_ppm",
        ],
    )

    out["grade"] = prefer_first(
        g,
        [
            "au_feed_grade_ppm",
            "leach_feed_grade_au_gt_day",
        ],
    )

    out["throughput"] = prefer_first(
        g,
        [
            "leach_feed_dry_t",
            "leach_feed_throughput_m3day",
        ],
    )

    if "pct_lt_75" in g.columns and "pct_lt_150" in g.columns:
        pct_lt_75 = pd.to_numeric(g["pct_lt_75"], errors="coerce")
        pct_lt_150 = pd.to_numeric(g["pct_lt_150"], errors="coerce")
        ultrafines = pct_lt_75 * 100.0
        fines = (pct_lt_150 - pct_lt_75) * 100.0
        coarse = (1.0 - pct_lt_150) * 100.0
    else:
        lt_75_src = g["leach_feed_lt_75um_day"] if "leach_feed_lt_75um_day" in g.columns else _series_of_nan(g.index)
        gt_75_src = g["leach_feed_gt_75um_day"] if "leach_feed_gt_75um_day" in g.columns else _series_of_nan(g.index)
        gt_150_src = g["leach_feed_gt_150um_day"] if "leach_feed_gt_150um_day" in g.columns else _series_of_nan(g.index)

        lt_75 = pd.to_numeric(lt_75_src, errors="coerce")
        gt_75 = pd.to_numeric(gt_75_src, errors="coerce")
        gt_150 = pd.to_numeric(gt_150_src, errors="coerce")

        ultrafines = lt_75
        fines = gt_75 - gt_150
        coarse = gt_150

    coarse, fines, ultrafines = normalise_triplet(coarse, fines, ultrafines)
    out["coarse"] = coarse
    out["fines"] = fines
    out["ultrafines"] = ultrafines

    for c in ["percent_solids", "cn", "do", "grade", "throughput", "coarse", "fines", "ultrafines"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(6)

    return out


def preflight(df_raw: pd.DataFrame, log: Callable[[str], None]) -> pd.DataFrame:
    orig_cols = list(df_raw.columns)
    cols = [clean_like_server(str(c)) for c in orig_cols]
    df = df_raw.copy()
    df.columns = cols

    log("== Preflight: source ==")
    log(f"Original columns ({len(orig_cols)}): {orig_cols[:10]}{' ...' if len(orig_cols) > 10 else ''}")
    log(f"Cleaned  columns ({len(cols)}): {cols[:10]}{' ...' if len(cols) > 10 else ''}")

    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        bad = int(dt.isna().sum())
        log(f"Date parse: {len(dt) - bad} ok, {bad} bad")

    derived = derive_required_fields(df)
    missing_by_row = derived[REQUIRED].isna().any(axis=1)
    n_missing = int(missing_by_row.sum())
    log("\n== Derived REQUIRED snapshot (first 3 rows) ==")
    log(derived.head(3).to_string())
    log(f"\nRows missing any REQUIRED fields: {n_missing} / {len(derived)}")
    if n_missing:
        sample = derived[missing_by_row].head(3)
        log("\nSample rows with missing REQUIRED fields:")
        log(sample.to_string())

    return df


def make_session_with_retry(
    total: int = 3,
    backoff: float = 0.5,
    status_forcelist: Iterable[int] = (502, 503, 504),
) -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=total,
        backoff_factor=backoff,
        status_forcelist=tuple(status_forcelist),
        allowed_methods=frozenset(["POST"]),
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def post_rows(
    session: requests.Session,
    base: str,
    headers: dict[str, str],
    customer: str,
    site: str,
    tag: str,
    row_batch: list[dict[str, Any]],
    req_mode: str,
) -> dict[str, Any]:
    payload = {
        "customer": customer,
        "site": site,
        "mode": req_mode,
        "tag": tag,
        "rows": row_batch,
    }
    json.dumps(payload, allow_nan=False)
    r = session.post(f"{base}/ingest/historical", json=payload, headers=headers, timeout=300)
    if not r.ok:
        try:
            msg = r.json()
        except Exception:
            msg = r.text
        raise RuntimeError(f"Upload failed {r.status_code}: {msg}")
    return cast(dict[str, Any], r.json())


# =========================
# GUI
# =========================
APP_TITLE = "LeachIT: Excel Extract & Upload"


class BusyDialog(tk.Toplevel):
    pb: ttk.Progressbar

    def __init__(self, parent: tk.Tk, text: str = "Working…") -> None:
        super().__init__(parent)
        self.title("Working")
        self.resizable(False, False)
        self.transient(cast(tk.Wm, parent))
        self.grab_set()

        frm = ttk.Frame(self, padding=16)
        frm.pack(fill="both", expand=True)                                                                                      

        ttk.Label(frm, text=text).pack(anchor="w", pady=(0, 8))
        self.pb = ttk.Progressbar(frm, mode="indeterminate", length=260)
        self.pb.pack(fill="x")
        self.pb.start(12)

        self.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")


class App(tk.Tk):
    xlsx_path: Optional[Path]
    busy: Optional[BusyDialog]
    cached_extract_df: Optional[pd.DataFrame]

    def _auth_headers(self, session: requests.Session, base: str) -> dict[str, str]:
        if not self.access_token:
            if not self.refresh_token:
                raise ValueError("No access token available. Please sign in again.")
            tokens = refresh_login(session, base, self.refresh_token)
            self.access_token = tokens["access_token"]
            self.refresh_token = tokens["refresh_token"]

        return {"Authorization": f"Bearer {self.access_token}"}

    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)

        # Window / taskbar icon
        self._icon_refs: list[tk.PhotoImage] = []
        try:
            self.iconbitmap(default=str(ICON_FILE))
        except Exception:
            try:
                png_icon = tk.PhotoImage(file=str(LOGO_FILE))
                self.iconphoto(True, png_icon)
                self._icon_refs.append(png_icon)
            except Exception:
                pass

        self.geometry("1180x820")
        self.minsize(1080, 780)

        self.var_password = tk.StringVar()
        self.var_logged_in_as = tk.StringVar(value="Not signed in")

        self.access_token: str = ""
        self.refresh_token: str = ""

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 12))
        style.configure("Muted.TLabel", foreground="#6b7280")

        # --- Header with logo ---
        header = ttk.Frame(self, padding=(12, 8))
        header.pack(fill="x")
        self.logo_img: Optional[tk.PhotoImage] = None
        try:
            img = tk.PhotoImage(file=str(LOGO_FILE))
            if img.width() > 220:
                factor = max(1, int(round(img.width() / 220)))
                img = img.subsample(factor, factor)
            self.logo_img = img
            ttk.Label(header, image=self.logo_img).pack(side="left", padx=(2, 10))
        except Exception:
            ttk.Label(header, text="LeachIT", font=("Segoe UI Semibold", 18)).pack(side="left", padx=(2, 10))
        ttk.Label(header, text="Excel Extract & Upload Tool", font=("Segoe UI", 16)).pack(side="left")

        outer = ttk.Frame(self, padding=12)
        outer.pack(fill="both", expand=True)

        # -------- Left column --------
        left = ttk.Frame(outer, width=420)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)

        # Quick Actions
        qa = ttk.LabelFrame(left, text="Quick Actions", padding=12)
        qa.pack(fill="x", pady=(0, 8))
        self.btn_pick = ttk.Button(qa, text="①  Choose Excel File…", command=self.pick_file)
        self.btn_pick.pack(fill="x", pady=4)
        self.btn_preview = ttk.Button(qa, text="②  Preview / Preflight", command=self.start_preview, state="disabled")
        self.btn_preview.pack(fill="x", pady=4)
        self.btn_upload = ttk.Button(qa, text="③  Extract → Upload", command=self.start_upload, state="disabled")
        self.btn_upload.pack(fill="x", pady=4)
        self.lbl_file = ttk.Label(qa, text="No file selected", style="Muted.TLabel")
        self.lbl_file.pack(anchor="w", pady=(6, 0))

        # Collapsible Advanced
        adv_wrap = ttk.LabelFrame(left, text="Advanced Settings", padding=8)
        adv_wrap.pack(fill="x", pady=(8, 8))
        self.adv_open = tk.BooleanVar(value=True)   
        self._adv_toggle_btn = ttk.Button(adv_wrap, text="Show ▾", command=self.toggle_advanced)
        self._adv_toggle_btn.pack(anchor="w")

        self.adv_frame = ttk.Frame(adv_wrap, padding=(2, 6))
        self.adv_frame.pack(fill="x", pady=(6, 4))
        self._adv_toggle_btn.config(text="Hide ▴")

        # Load merged settings for defaults
        try:
            settings_merged, settings_defaults, settings_user = load_settings()
            ui_defaults = cast(dict[str, Any], settings_merged.get("ui_defaults", {}) or {})
        except Exception:
            settings_defaults = {}
            settings_user = {}
            ui_defaults = {}

        # Vars
        self.var_email = tk.StringVar(value=str(ui_defaults.get("email", "")))
        self.var_base = tk.StringVar(
            value=str(ui_defaults.get("base", "http://localhost:8000/api"))
        )
        self.var_cust = tk.StringVar(value=str(ui_defaults.get("customer", "")))
        self.var_site = tk.StringVar(value=str(ui_defaults.get("site", "")))
        self.var_tag = tk.StringVar(value=str(ui_defaults.get("tag", "bulk_upload")))
        self.mode_var = tk.StringVar(value=str(ui_defaults.get("mode", "append")))
        self.var_batch = tk.IntVar(value=int(ui_defaults.get("batch", 5000)))
        self.var_dry = tk.BooleanVar(value=bool(ui_defaults.get("dry_run", False)))
        self.var_save_csv = tk.BooleanVar(value=bool(ui_defaults.get("save_csv", False)))

        LABEL_W = 14

        # Build Advanced UI
        def add_row(label: str, var: tk.Variable, show: Optional[str] = None) -> None:
            frm = ttk.Frame(self.adv_frame)
            frm.pack(fill="x", pady=3)

            ttk.Label(frm, text=label, width=LABEL_W, anchor="w").pack(side="left", padx=(0, 8))

            entry_kwargs: dict[str, Any] = {"textvariable": var}
            if show is not None:
                entry_kwargs["show"] = show

            entry = ttk.Entry(frm, **entry_kwargs)
            entry.pack(side="left", fill="x", expand=True)

        add_row("BASE URL", self.var_base)
        add_row("Email", self.var_email)
        add_row("Password", self.var_password, show="*")
        auth_row = ttk.Frame(self.adv_frame)
        auth_row.pack(fill="x", pady=(4, 6))

        ttk.Button(auth_row, text="Sign In", command=self.start_login).pack(side="left")
        ttk.Button(auth_row, text="Sign Out", command=self.sign_out).pack(side="left", padx=8)

        ttk.Label(
            self.adv_frame,
            textvariable=self.var_logged_in_as,
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 6))
        add_row("Customer", self.var_cust)
        add_row("Site", self.var_site)
        add_row("Tag", self.var_tag)

        row2 = ttk.Frame(self.adv_frame)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text="Mode", width=LABEL_W, anchor="w").pack(side="left")
        ttk.Combobox(row2, textvariable=self.mode_var, values=["replace", "append"], width=12, state="readonly").pack(
            side="right"
        )

        row3 = ttk.Frame(self.adv_frame)
        row3.pack(fill="x", pady=2)
        ttk.Label(row3, text="Batch size", width=LABEL_W, anchor="w").pack(side="left")
        ttk.Spinbox(row3, from_=100, to=50000, increment=100, textvariable=self.var_batch, width=10).pack(side="right")

        opts = ttk.Frame(self.adv_frame)
        opts.pack(fill="x", pady=(4, 2))
        ttk.Checkbutton(opts, text="Dry run (no upload)", variable=self.var_dry).pack(anchor="w", pady=(0, 2))
        ttk.Checkbutton(opts, text="Also save extracted CSV", variable=self.var_save_csv).pack(anchor="w")

        # Save / Reset buttons
        btn_row = ttk.Frame(self.adv_frame)
        btn_row.pack(fill="x", pady=(6, 2))
        ttk.Button(btn_row, text="Save Settings", command=self._save_settings).pack(side="left")
        ttk.Button(btn_row, text="Reset to Defaults", command=self._reset_settings).pack(side="left", padx=8)

        # -------- Right panel: Log --------
        right = ttk.Frame(outer)
        right.pack(side="right", fill="both", expand=True)

        logcard = ttk.LabelFrame(right, text="Log / Diagnostics", padding=12)
        logcard.pack(fill="both", expand=True)
        self.log_box = ScrolledText(logcard, height=28, wrap="word")
        self.log_box.pack(fill="both", expand=True, pady=(4, 0))

        try:
            d_ui = cast(dict[str, Any], settings_defaults.get("ui_defaults", {}) or {})
            u_ui = cast(dict[str, Any], settings_user.get("ui_defaults", {}) or {})
            self.log(f"Default settings file: {DEFAULT_SETTINGS}")
            self.log(f"User settings file: {USER_SETTINGS}")
            self.log(f"Default BASE URL: {d_ui.get('base', '')}")
            self.log(f"User BASE URL override: {u_ui.get('base', '')}")
            self.log(f"Effective BASE URL: {self.var_base.get()}")
        except Exception:
            pass

        # State
        self.xlsx_path = None
        self.cancel_requested = False
        self.busy = None
        self.cached_extract_df = None
        self._toasts: list[tk.Toplevel] = []

        # Hotkeys / close
        self.bind("<Escape>", lambda e: self.destroy())
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # =========================
    # Login actions and worker
    # ========================

    def start_login(self) -> None:
        self.run_in_thread(self._do_login, "Signing in…")


    def sign_out(self) -> None:
        self.access_token = ""
        self.refresh_token = ""
        self.var_password.set("")
        self.var_logged_in_as.set("Not signed in")
        self.toast("Signed out", "info")
        self.log("Signed out.")


    def _do_login(self) -> None:
        try:
            base = self.var_base.get().strip().rstrip("/")
            email = self.var_email.get().strip()
            password = self.var_password.get()

            if not base:
                raise ValueError("BASE URL must not be empty.")
            if not email:
                raise ValueError("Email must not be empty.")
            if not password:
                raise ValueError("Password must not be empty.")

            session = make_session_with_retry(total=2, backoff=0.5)
            tokens = login_user(session, base, email, password)

            self.log(f"Attempting login against: {base}/auth/login")

            self.access_token = tokens["access_token"]
            self.refresh_token = tokens["refresh_token"]
            self.var_logged_in_as.set(f"Signed in as {email}")

            try:
                save_user_settings(self._collect_ui_defaults())
                self.log(f"Saved settings → {USER_SETTINGS}")
            except Exception as save_err:
                self.log(f"Warning: could not save settings: {save_err}")

            self.log("Login successful.")
            self.toast("Signed in", "success")

        except Exception as e:
            self.access_token = ""
            self.refresh_token = ""
            self.var_logged_in_as.set("Not signed in")
            messagebox.showerror("Login error", str(e))
            self.toast("Sign in failed", "error")


    # ---------- simple toast ----------
    def toast(self, msg: str, tone: str = "info", ttl_ms: int = 2400) -> None:
        colors = {
            "success": "#22c55e",
            "info": "#38bdf8",
            "warn": "#f59e0b",
            "error": "#ef4444",
            "default": "#e5e7eb",
        }
        fg = colors.get(tone, colors["default"])
        tw = tk.Toplevel(self)
        self._toasts.append(tw)
        tw.overrideredirect(True)
        tw.configure(bg="#0f172a")
        lbl = tk.Label(tw, text=msg, fg=fg, bg="#0f172a", padx=12, pady=8, font=("Segoe UI", 10))
        lbl.pack()
        self.update_idletasks()
        x = self.winfo_rootx() + self.winfo_width() - (tw.winfo_reqwidth() + 24)
        y = self.winfo_rooty() + self.winfo_height() - (tw.winfo_reqheight() + 24)
        tw.geometry(f"+{x}+{y}")

        def _close() -> None:
            if tw in self._toasts:
                self._toasts.remove(tw)
            try:
                tw.destroy()
            except Exception:
                pass

        tw.after(ttl_ms, _close)

    # ---------- collapse/expand ----------
    def toggle_advanced(self) -> None:
        opened = self.adv_open.get()
        if opened:
            self.adv_frame.pack_forget()
            self.adv_open.set(False)
            self._adv_toggle_btn.config(text="Show ▾")
        else:
            self.adv_frame.pack(fill="x", pady=(6, 4))
            self.adv_open.set(True)
            self._adv_toggle_btn.config(text="Hide ▴")

    # ---------- logging helpers ----------
    def log(self, text: str) -> None:
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.update_idletasks()

    def set_actions_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        preview_state = "normal" if enabled and self.xlsx_path is not None else "disabled"
        upload_state = "normal" if enabled and self.xlsx_path is not None else "disabled"
        self.btn_preview.config(state=preview_state)
        self.btn_upload.config(state=upload_state)
        self.btn_pick.config(state=state)

    # ---------- settings helpers ----------
    def _collect_ui_defaults(self) -> dict[str, Any]:
        return {
            "base": self.var_base.get().strip(),
            "token": "",
            "email": self.var_email.get().strip(),
            "customer": self.var_cust.get().strip(),
            "site": self.var_site.get().strip(),
            "tag": self.var_tag.get().strip(),
            "mode": self.mode_var.get().strip(),
            "batch": int(self.var_batch.get()),
            "dry_run": bool(self.var_dry.get()),
            "save_csv": bool(self.var_save_csv.get()),
        }

    def _save_settings(self) -> None:
        try:
            save_user_settings(self._collect_ui_defaults())
            self.toast("Settings saved", "success")
            self.log(f"Saved settings → {USER_SETTINGS}")
        except Exception as e:
            self.toast("Could not save settings", "error")
            messagebox.showerror("Save error", str(e))

    def _reset_settings(self) -> None:
        if not messagebox.askyesno("Reset settings", "Revert to defaults for this user?"):
            return
        reset_user_settings()
        merged, _defaults, _user = load_settings()
        ui = cast(dict[str, Any], merged.get("ui_defaults", {}) or {})
        self.var_base.set(str(ui.get("base", "http://localhost:8000/api")))
        self.var_email.set(str(ui.get("email", "")))
        self.var_password.set("")
        self.var_logged_in_as.set("Not signed in")
        self.access_token = ""
        self.refresh_token = ""
        self.var_cust.set(str(ui.get("customer", "shanta")))
        self.var_site.set(str(ui.get("site", "new_luika")))
        self.var_tag.set(str(ui.get("tag", "bulk_upload")))
        self.mode_var.set(str(ui.get("mode", "append")))
        self.var_batch.set(int(ui.get("batch", 5000)))
        self.var_dry.set(bool(ui.get("dry_run", False)))
        self.var_save_csv.set(bool(ui.get("save_csv", False)))
        self.toast("Reset to defaults", "info")

    def _on_close(self) -> None:
        try:
            save_user_settings(self._collect_ui_defaults())
        except Exception:
            pass

        self.var_password.set("")
        self.destroy()

    # ---------- UI callbacks ----------
    def pick_file(self) -> None:
        f = filedialog.askopenfilename(title="Select workbook", filetypes=[("Excel files", "*.xlsx")])
        if not f:
            return
        self.xlsx_path = Path(f)
        self.lbl_file.config(text=f"Selected: {self.xlsx_path.name}")
        self.btn_preview.config(state="normal")
        self.btn_upload.config(state="normal")
        self.toast("Workbook selected", "success")
        self.log(f"Workbook selected: {self.xlsx_path}")

    def request_cancel(self) -> None:
        self.cancel_requested = True
        self.log("Cancel requested; finishing current step…")

    def start_preview(self) -> None:
        if not self.xlsx_path:
            messagebox.showwarning("No file", "Please choose a workbook first.")
            return
        self.run_in_thread(self._do_preview, "Preflighting…")

    def start_upload(self) -> None:
        if not self.xlsx_path:
            messagebox.showwarning("No file", "Please choose a workbook first.")
            return
        dry = self.var_dry.get()
        if self.mode_var.get().lower() == "replace" and not dry:
            resp = messagebox.askyesno(
                "Confirm REPLACE",
                "This will REPLACE all existing history on the server for this site.\n\nContinue?",
            )
            if not resp:
                return
        self.run_in_thread(self._do_upload, "Extracting and uploading…")

    # ---------- background execution ----------
    def run_in_thread(self, target: Callable[[], None], label: str) -> None:
        self.cancel_requested = False
        self.set_actions_enabled(False)
        self.busy = BusyDialog(self, label)
        t = threading.Thread(target=target, daemon=True)
        t.start()
        self.after(120, lambda: self._poll_thread(t))

    def _poll_thread(self, thread: threading.Thread) -> None:
        if thread.is_alive():
            self.after(160, lambda: self._poll_thread(thread))
        else:
            if self.busy is not None:
                try:
                    self.busy.grab_release()
                except Exception:
                    pass
                try:
                    self.busy.destroy()
                except Exception:
                    pass
                self.busy = None
            self.set_actions_enabled(True)

    # ---------- core work ----------
    def _extract_frame(self) -> pd.DataFrame:
        if self.xlsx_path is None:
            raise ValueError("No workbook selected.")
        self.log(f"Loading config from: {DEFAULT_SETTINGS}")
        cfg_text = load_config_text()
        self.log("Extracting from Excel…")
        df = build_extract(self.xlsx_path, cfg_text)
        self.log(f"Extracted shape: {df.shape}")
        cols = [c if c != "Date" else "date" for c in df.columns]
        df.columns = cols
        return df

    def _preflight_only(self, df_extract: pd.DataFrame) -> None:
        df_wide = df_extract.copy()
        df_wide.columns = [clean_like_server(str(c)) for c in df_wide.columns]
        df_wide = cleanse_df(df_wide)

        if self.var_save_csv.get():
            suggested = f"extract_{datetime.now():%Y%m}.csv"
            save_path = filedialog.asksaveasfilename(
                title="Save extracted CSV",
                defaultextension=".csv",
                initialfile=suggested,
                filetypes=[("CSV", "*.csv")],
            )
            if save_path:
                save_path_obj = Path(save_path)
                save_path_obj.parent.mkdir(parents=True, exist_ok=True)
                df_wide.to_csv(save_path_obj, index=False)
                self.log(f"Saved extracted CSV → {save_path_obj}")

        _ = preflight(df_wide, self.log)

    def _do_preview(self) -> None:
        try:
            df = self._extract_frame()
            self._preflight_only(df)
            self.toast("Preview complete", "success")
            self.log("\nPreview complete.")
        except Exception as e:
            messagebox.showerror("Preview error", str(e))
            self.toast("Preview failed", "error")

    def _do_upload(self) -> None:
        try:
            df = self._extract_frame()

            if self.var_save_csv.get():
                suggested = f"extract_{datetime.now():%Y%m}.csv"
                save_path = filedialog.asksaveasfilename(
                    title="Save extracted CSV",
                    defaultextension=".csv",
                    initialfile=suggested,
                    filetypes=[("CSV", "*.csv")],
                )
                if save_path:
                    save_path_obj = Path(save_path)
                    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
                    tmp = df.copy()
                    tmp.columns = [clean_like_server(str(c)) for c in tmp.columns]
                    tmp = cleanse_df(tmp)
                    tmp.to_csv(save_path_obj, index=False)
                    self.log(f"Saved extracted CSV → {save_path_obj}")

            # Preflight + build wide frame
            df.columns = [clean_like_server(str(c)) for c in df.columns]
            df = cleanse_df(df)
            self.log("Running preflight checks…")
            _ = preflight(df, self.log)

            if self.var_dry.get():
                self.log("\nDry run enabled — no upload performed.")
                self.toast("Dry run complete", "info")
                return

            if df.empty:
                self.log("No rows to upload after cleansing.")
                self.toast("Nothing to upload", "warn")
                return

            base = self.var_base.get().strip()

            # Normalise base URL (no forced rewriting)
            base = base.rstrip("/")

            cust = self.var_cust.get().strip()
            site = self.var_site.get().strip()
            tag = self.var_tag.get().strip()
            mode = self.mode_var.get().strip().lower()
            batch = max(100, int(self.var_batch.get()))

            if not base:
                raise ValueError("BASE URL must not be empty.")
            if not cust or not site:
                raise ValueError("Customer and Site must not be empty.")
            if not self.access_token and not self.refresh_token:
                raise ValueError("Please sign in before uploading.")

            session = make_session_with_retry(total=3, backoff=0.8)

            headers = self._auth_headers(session, base)

            rows = df_to_json_records(df)
            total = len(rows)
            self.log(f"\nUploading {total} rows in batches of {batch} (mode={mode})…")
            sent = 0
            req_mode = mode

            self.log(f"POST target: {base}/ingest/historical")

            if self.busy is not None:
                try:
                    self.busy.pb.stop()
                    self.busy.pb.config(mode="determinate", maximum=total)
                    self.busy.pb["value"] = 0
                except Exception:
                    pass

            while sent < total:
                if self.cancel_requested:
                    self.log("\nUpload cancelled by user.")
                    self.toast("Upload cancelled", "warn")
                    return

                batch_rows = rows[sent : sent + batch]

                try:
                    headers = self._auth_headers(session, base)
                    resp = post_rows(session, base, headers, cust, site, tag, batch_rows, req_mode)
                except Exception as upload_err:
                    err_text = str(upload_err)
                    if "401" in err_text and self.refresh_token:
                        self.log("Access token may have expired; attempting refresh.")
                        self.access_token = ""
                        headers = self._auth_headers(session, base)
                        resp = post_rows(session, base, headers, cust, site, tag, batch_rows, req_mode)
                    else:
                        self.log(f"Batch {sent}..{sent + len(batch_rows)} failed: {upload_err}")
                        self.toast("Upload failed", "error")
                        raise

                self.log(f"Sent {sent}..{sent + len(batch_rows)} -> OK {resp}")
                sent += len(batch_rows)
                req_mode = "append"  # subsequent batches always append

                if self.busy is not None:
                    try:
                        self.busy.pb["value"] = sent
                    except Exception:
                        pass

            # Save current settings after a successful run
            try:
                save_user_settings(self._collect_ui_defaults())
                self.log(f"Saved settings → {USER_SETTINGS}")
            except Exception:
                pass

            self.log("\nUpload complete.")
            self.toast("Upload complete", "success")
            messagebox.showinfo("Done", f"Uploaded {total} rows successfully.")

        except Exception as e:
            messagebox.showerror("Upload error", f"{e}")
            self.toast("Upload error", "error")


# =========================
# Entry point
# =========================
if __name__ == "__main__":
    App().mainloop()
