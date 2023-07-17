#!/usr/bin/env python3

import sys

MINIMUM_PYTHON_VERSION = (3, 10)
if sys.version_info < MINIMUM_PYTHON_VERSION:
    raise Exception(
        "Minimal Python version required: " + ".".join(map(str, MINIMUM_PYTHON_VERSION))
    )

import argparse
from io import TextIOWrapper
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, DefaultDict, Generator, Sequence
import unicodedata
from ctypes import (
    c_char_p,
    c_int,
    c_uint32,
    cdll,
)
from ctypes.util import find_library

################################################################################
# xkbcommon handling
################################################################################

# Try to load xkbcommon
if xkbcommon_path := find_library("xkbcommon"):
    HAS_XKBCOMMON = True
    xkbcommon = cdll.LoadLibrary(xkbcommon_path)

    xkb_keysym_t = c_uint32
    xkbcommon.xkb_keysym_from_name.argtypes = [c_char_p, c_int]
    xkbcommon.xkb_keysym_from_name.restype = xkb_keysym_t

    xkbcommon.xkb_keysym_to_utf32.argtypes = [xkb_keysym_t]
    xkbcommon.xkb_keysym_to_utf32.restype = c_uint32

    XKB_KEY_NoSymbol = 0
    XKB_KEYSYM_NO_FLAGS = 0

    def xkb_keysym_from_name(keysym_name: str) -> int:
        return xkbcommon.xkb_keysym_from_name(
            keysym_name.encode("utf-8"), XKB_KEYSYM_NO_FLAGS
        )

    def keysym_to_char(keysym_name: str) -> str:
        keysym = xkb_keysym_from_name(keysym_name)
        if keysym == XKB_KEY_NoSymbol:
            raise ValueError(f"Unsupported keysym: “{keysym_name}”")
        codepoint = xkbcommon.xkb_keysym_to_utf32(keysym)
        if codepoint == 0:
            raise ValueError(
                f"Keysym cannot be translated to character: “{keysym_name}”"
            )
        return chr(codepoint)

else:
    HAS_XKBCOMMON = False


################################################################################
# Keysyms headers
################################################################################

DEFAULT_KEYSYMS_HEADERS_PREFIX = Path("/usr")
DEFAULT_KEYSYMS_HEADERS = [
    Path("include/X11/keysymdef.h"),
    Path("include/X11/XF86keysym.h"),
    Path("include/X11/Sunkeysym.h"),
    Path("include/X11/DECkeysym.h"),
    Path("include/X11/HPkeysym.h"),
]

KEYSYM_ENTRY_PATTERN = re.compile(
    r"""
    ^\#define\s+
    (?:(?P<prefix>\w+)?XK|XKB_KEY)_(?P<name>\w+)\s+
    (?P<evdev>_EVDEVK\()?
    (?P<value>0x[0-9a-fA-F]+)
    (?(evdev)\)|)\s*
    (?:/\*\s*
        (?:
            (?P<deprecated>deprecated)|
            \(U\+(?P<unicode>[0-9a-fA-F]{4,})(?:\s|\w|-)+\)|
            .*
        )
    )?
    """,
    re.VERBOSE,
)
EXTRA_DEPRECATED_KEYSYMS = ("Ext16bit_L", "Ext16bit_R")


def parse_keysyms_header(
    path: Path, keysyms: dict[int, str], keysyms_names: dict[str, str]
):
    with path.open("rt", encoding="utf-8") as fd:
        pending_multine_comment = False
        for n, line in enumerate(map(lambda l: l.strip(), fd)):
            if not line:
                # Empty line
                continue
            elif pending_multine_comment:
                if line.endswith("*/"):
                    pending_multine_comment = False
                continue
            elif line.startswith("/*"):
                if not line.endswith("*/"):
                    pending_multine_comment = True
                continue
            elif any(
                line.startswith(s)
                for s in ("#ifdef", "#ifndef", "#endif", "#define _", "#undef")
            ):
                continue
            elif m := KEYSYM_ENTRY_PATTERN.match(line):
                if m.group("evdev"):
                    # _EVDEVK macro
                    keysym = 0x10081000 + int(m.group("value"), 16)
                else:
                    keysym = int(m.group("value"), 16)
                name = (m.group("prefix") or "") + m.group("name")
                if ref := keysyms.get(keysym):
                    # Deprecated, because there is a previous definition with other name.
                    # Ensure that the replacement keysym is supported by xkbcommon.
                    if (
                        not HAS_XKBCOMMON
                        or xkb_keysym_from_name(ref) != XKB_KEY_NoSymbol
                    ):
                        keysyms_names[name] = ref
                        continue
                    else:
                        print(
                            f"[WARNING] Line {n}: Keep deprecated keysym “{name}”; reference keysym “{ref}” is not supported by available xkbcommon."
                        )
                else:
                    # Reference keysym
                    keysyms[keysym] = name
                if (
                    m.group("deprecated")
                    or m.group("unicode")
                    or m.group("name") in EXTRA_DEPRECATED_KEYSYMS
                ):
                    # Explicitely deprecated
                    keysyms_names[name] = ""
                else:
                    # Reference keysym
                    keysyms_names[name] = name
            else:
                raise ValueError(f"Cannot parse header “{path}” line: {line}")


def parse_keysyms_headers(paths: Sequence[Path]) -> dict[str, str]:
    keysyms: dict[int, str] = {}
    keysyms_names: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            print(f"[ERROR] Cannot open keysym header file: {path}")
        else:
            print(f" Processing header file: {path} ".center(80, "="), file=sys.stderr)
            parse_keysyms_header(path, keysyms, keysyms_names)
    return keysyms_names


################################################################################
# Compose files
################################################################################

COMPOSE_ENTRY_PATTERN = re.compile(
    r"""^(?P<sequence>[^:]+)
        :\s+
        "(?P<string>(?:\\"|[^"])+)"
        (?:\s+(?P<keysym>\w+))?
        (?:(?P<space>\s+)\#\s*(?P<comment>.*))?
    """,
    re.VERBOSE,
)
"""A pattern for Compose entries"""

UNICODE_KEYSYM_PATTERN = re.compile(r"\bU(?P<codepoint>[0-9A-Fa-f]+)\b")
KEYSYM_PATTERN = re.compile(r"<(\w+)>")


def _unescape(s: str) -> Generator[str, Any, None]:
    """Unescape a Compose file string"""
    pending_escape = False
    for c in s:
        # WARNING: probably incomplete, but sufficient for now
        if pending_escape:
            match c:
                case "\\":
                    yield c
                    pending_escape = False
                    break
                case '"':
                    yield c
                    pending_escape = False
                    break
                case _:
                    raise ValueError(f"Invalid escape sequence: “{s}”")
        elif c == "\\":
            pending_escape = True
        else:
            yield c
    if pending_escape:
        raise ValueError(f"Incomplete escape sequence: “{s}”")


def unescape(s: str) -> str:
    return "".join(_unescape(s))


def unicode_name(c: str, is_first: bool) -> str:
    # TODO: we should use Unicode *corrected* names!
    #       But the Python API does not propose those.
    name = unicodedata.name(c, None)
    if name is None:
        raise ValueError(f"Cannot find Unicode name for: “{c}” (U+{ord(c):0>4X})")
    # RULE: remove “ACCENT” from the name, when the character is combining and
    #       is not in first position
    if not is_first and "COMBINING" in name and name.endswith("ACCENT"):
        return name[:-7]
    else:
        return name


def make_comment(s: str) -> str:
    """Make the comment of a Compose sequence, based on its result."""
    return " plus ".join(unicode_name(c, k == 0) for k, c in enumerate(s))


def check_keysym(deprecated_keysyms: dict[str, str], n: int, keysym_name: str) -> str:
    if m := UNICODE_KEYSYM_PATTERN.match(keysym_name):
        # Reformat Unicode keysym
        codepoint = int(m.group("codepoint"), 16)
        return f"U{codepoint:0>4X}"
    ref = deprecated_keysyms.get(keysym_name)
    if keysym_name == ref:
        # Reference keysym
        return keysym_name
    elif ref is None:
        print(f"[ERROR] Line {n}: Unsupported keysym “{keysym_name}”")
        return keysym_name
    elif ref == "":
        # Deprecated: keep keysym
        print(f"[WARNING] Line {n}: Deprecated keysym “{keysym_name}”.")
        return keysym_name
    else:
        # Deprecated alias: return reference keysym
        print(
            f"[WARNING] Line {n}: Deprecated keysym “{keysym_name}”. Please use “{ref}” instead."
        )
        return ref


def check_keysym_sequence(
    deprecated_keysyms: dict[str, str], n: int, sequence: str
) -> str:
    subsitutions: dict[str, str] = {}
    for keysym_name in KEYSYM_PATTERN.findall(sequence):
        keysym_nameʹ = check_keysym(deprecated_keysyms, n, keysym_name)
        if keysym_nameʹ != keysym_name:
            subsitutions[keysym_name] = keysym_nameʹ
    if subsitutions:
        pattern = re.compile("|".join(re.escape(k) for k in subsitutions.keys()))
        return pattern.sub(lambda x: subsitutions[x.group()], sequence)
    else:
        return sequence


def process_lines(fd: TextIOWrapper, keysyms_names: dict[str, str]):
    multi_line_comment = False
    for n, line in enumerate(fd, start=1):
        # Handle pending multi-line comment
        if multi_line_comment:
            if line.strip().endswith("*/"):
                multi_line_comment = False
            yield line
        # Handle single-line comment & include
        elif not line.strip() or any(
            line.startswith(s) for s in ("XCOMM", "#", "include")
        ):
            yield line
        # Handle start of a multi-line comment
        elif line.startswith("/*"):
            # Check if one-liner
            if not line.strip().endswith("*/"):
                multi_line_comment = True
            yield line
        # Handle compose sequence
        elif m := COMPOSE_ENTRY_PATTERN.match(line):
            string = unescape(m.group("string"))
            rewrite = False
            # Check sequence keysyms
            if keysyms_names:
                sequence = check_keysym_sequence(keysyms_names, n, m.group("sequence"))
                if sequence != m.group("sequence"):
                    rewrite = True
            else:
                sequence = m.group("sequence")
            # Check result keysym
            if keysym := m.group("keysym"):
                if HAS_XKBCOMMON:
                    keysym_char = keysym_to_char(m.group("keysym"))
                    if string != keysym_char:
                        print(
                            f"[ERROR] Line {n}: The keysym does not correspond to the character: expected “{string}”, got “{keysym_char}”.",
                            file=sys.stderr,
                        )
                if keysyms_names:
                    keysym = check_keysym(keysyms_names, n, m.group("keysym"))
                    if keysym != m.group("keysym"):
                        rewrite = True
            expected_comment = make_comment(string)
            # Check if we have the expected comment
            # NOTE: Some APL sequences provide the combo of composed characters
            if not (
                m.group("comment") == expected_comment
                or (m.group("comment") and m.group("comment")[4:] == expected_comment)
            ):
                print(
                    f"[WARNING] Line {n}: Expected “{expected_comment}” comment, "
                    f"got: “{m.group('comment')}”",
                    file=sys.stderr,
                )
                rewrite = True
            # Rewrite entry if necessary
            if rewrite:
                keysym = "" if keysym is None else f"\t{keysym}"
                assert (len(string) == 1 and m.group("keysym") is not None) ^ (
                    len(string) > 1 and m.group("keysym") is None
                )
                comment_space = " " if len(string) == 1 else m.group("space") or "\t"
                yield f"""{sequence}: "{m.group('string')}"{keysym}{comment_space}# {expected_comment}\n"""
            else:
                yield line
        else:
            raise ValueError(f"Cannot parse line: “{line}”")


def process_file(path: Path, keysyms_names: dict[str, str]):
    with path.open("rt", encoding="utf-8") as fd:
        yield from process_lines(fd, keysyms_names)


def run(paths: Sequence[Path], write: bool, keysyms_headers: Sequence[Path]):
    # Keysyms headers
    keysyms_names = parse_keysyms_headers(keysyms_headers)
    # Compose file
    for path in paths:
        print(f" Processing Compose file: {path} ".center(80, "="), file=sys.stderr)
        if write:
            with tempfile.NamedTemporaryFile("wt") as fd:
                # Write to a temporary file
                fd.writelines(process_file(path, keysyms_names))
                fd.flush()
                # No error: now ovewrite the original file
                shutil.copyfile(fd.name, path)
        else:
            for _ in process_file(path, keysyms_names):
                pass


def parse_args():
    parser = argparse.ArgumentParser(description="Add comment to compose sequence")
    parser.add_argument("input", type=Path, nargs="+", help="Compose file to process")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--no-keysyms", action="store_true", help="Do not check keysyms")
    group.add_argument(
        "--keysyms", type=Path, action="append", help="Add a keysym header to parse"
    )
    group.add_argument(
        "--keysyms-prefix",
        type=Path,
        default=DEFAULT_KEYSYMS_HEADERS_PREFIX,
        help="Keysym header prefix for default keysyms header files (default: %(default)s)",
    )
    parser.add_argument("--write", action="store_true", help="Write the compose file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.no_keysyms:
        keysyms = []
    elif args.keysyms:
        keysyms = args.keysyms
    else:
        keysyms = list(args.keysyms_prefix / path for path in DEFAULT_KEYSYMS_HEADERS)
    run(args.input, args.write, keysyms)
