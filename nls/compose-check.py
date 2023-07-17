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
from typing import Any, Generator, Sequence
import unicodedata


SEQUENCE_PATTERN = re.compile(
    r"""^(?P<sequence>[^:]+)
        :\s+
        "(?P<string>(?:\\"|[^"])+)"
        (?:\s+(?P<keysym>\w+))?
        (?:(?P<space>\s+)\#\s*(?P<comment>.*))?
    """,
    re.VERBOSE,
)
"""A pattern for Compose entries"""


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


# TODO: we probably also want to check that the keysyms are correct and not deprecated
def process_lines(fd: TextIOWrapper):
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
        elif m := SEQUENCE_PATTERN.match(line):
            string = unescape(m.group("string"))
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
                keysym = "" if m.group("keysym") is None else f"\t{m.group('keysym')}"
                assert (len(string) == 1 and m.group("keysym") is not None) ^ (
                    len(string) > 1 and m.group("keysym") is None
                )
                comment_space = " " if len(string) == 1 else m.group("space") or "\t"
                yield f"""{m.group('sequence')}: "{m.group('string')}"{keysym}{comment_space}# {expected_comment}\n"""
            else:
                yield line
        else:
            raise ValueError(f"Cannot parse line: “{line}”")


def process_file(path: Path):
    with path.open("rt", encoding="utf-8") as fd:
        yield from process_lines(fd)


def run(paths: Sequence[Path], write: bool):
    for path in paths:
        print(f" Processing Compose file: {path} ".center(80, "="), file=sys.stderr)
        if write:
            with tempfile.NamedTemporaryFile("wt") as fd:
                # Write to a temporary file
                fd.writelines(process_file(path))
                fd.flush()
                # No error: now ovewrite the original file
                shutil.copyfile(fd.name, path)
        else:
            for _ in process_file(path):
                pass


def parse_args():
    parser = argparse.ArgumentParser(description="Add comment to compose sequence")
    parser.add_argument("input", type=Path, nargs="+", help="Compose file to process")
    parser.add_argument("--write", action="store_true", help="Write the compose file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.write)
