#!/usr/bin/env python3

from __future__ import annotations

import sys

MINIMUM_PYTHON_VERSION = (3, 10)
if sys.version_info < MINIMUM_PYTHON_VERSION:
    raise Exception(
        "Minimal Python version required: " + ".".join(map(str, MINIMUM_PYTHON_VERSION))
    )

import argparse
import ctypes
import ctypes.util
from enum import Enum, IntFlag, unique
from functools import partial
import logging
import re
import shutil
import tempfile
import unicodedata
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Generator, Sequence


################################################################################
# Utils
################################################################################


@dataclass
class Configuration:
    keysyms_names: dict[str, str]
    unicode_name_aliases: dict[str, str]
    prefer_unicode_keysym: bool
    convert_xcomm: bool


logger = logging.getLogger(__name__)
verbosity: int = 0


def file_only(category: str, path: Path):
    """Check file"""
    if not path.is_file():
        logger.error(f"Invalid {category} file: {path}")
    return path.is_file()


def processing_file_message(category: str, path: Path):
    return f"=== Processing {category} file: {path} ==="


################################################################################
# xkbcommon handling
################################################################################


xkb_keysym_t = ctypes.c_uint32


class Xkbcommon:
    XKB_KEY_NoSymbol = 0
    XKB_KEYSYM_NO_FLAGS = 0

    def __init__(self, xkbcommon_path):
        self._lib = ctypes.cdll.LoadLibrary(xkbcommon_path)

        self._lib.xkb_keysym_from_name.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self._lib.xkb_keysym_from_name.restype = xkb_keysym_t

        self._lib.xkb_keysym_to_utf32.argtypes = [xkb_keysym_t]
        self._lib.xkb_keysym_to_utf32.restype = ctypes.c_uint32

        self._lib.xkb_utf32_to_keysym.argtypes = [ctypes.c_uint32]
        self._lib.xkb_utf32_to_keysym.restype = xkb_keysym_t

        self._lib.xkb_keysym_get_name.argtypes = [
            xkb_keysym_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self._lib.xkb_keysym_get_name.restype = int

    @classmethod
    def load(cls) -> Xkbcommon | None:
        """Try to load xkbcommon"""
        if xkbcommon_path := ctypes.util.find_library("xkbcommon"):
            return cls(xkbcommon_path)
        else:
            return None

    def keysym_from_name(self, keysym_name: str) -> int:
        return self._lib.xkb_keysym_from_name(
            keysym_name.encode("utf-8"), self.XKB_KEYSYM_NO_FLAGS
        )

    def is_invalid_keysym(self, keysym: int) -> bool:
        return keysym == self.XKB_KEY_NoSymbol

    def is_invalid_keysym_name(self, name: str) -> bool:
        return self.is_invalid_keysym(self.keysym_from_name(name))

    def keysym_to_char(self, keysym_name: str) -> str:
        keysym = self.keysym_from_name(keysym_name)
        if self.is_invalid_keysym(keysym):
            raise ValueError(f"Unsupported keysym: “{keysym_name}”")
        codepoint = self._lib.xkb_keysym_to_utf32(keysym)
        if codepoint == 0:
            raise ValueError(
                f"Keysym cannot be translated to character: “{keysym_name}”"
            )
        return chr(codepoint)

    def keysym_get_name(self, keysym: int) -> str:
        buf_len = 90
        buf = ctypes.create_string_buffer(buf_len)
        n = self._lib.xkb_keysym_get_name(keysym, buf, ctypes.c_size_t(buf_len))
        if n < 0:
            raise ValueError(f"Unsupported keysym: 0x{keysym:0>4X})")
        elif n >= buf_len:
            raise ValueError(f"Buffer is not big enough: expected at least {n}.")
        else:
            return buf.value.decode("utf-8")

    def char_to_keysym(self, char: str) -> str:
        keysym = self._lib.xkb_utf32_to_keysym(ord(char))
        if self.is_invalid_keysym(keysym):
            return ""
        else:
            return self.keysym_get_name(keysym)


libxkbcommon = Xkbcommon.load()

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
    (?:/\*(?P<comment>.+)\*/)?
    \s*$
    """,
    re.VERBOSE,
)
KEYSYM_DEPRECATION_COMMENT_PATTERN = re.compile(
    r"""
    # Explicit alias: do not deprecate
    alias\s+for\s+(?P<alias>\w+)|
    # Explicitly deprecated
    (?P<deprecated>deprecated)|
    # Inexact Unicode match
    \(U\+(?P<inexact_unicode>[0-9a-fA-F]{4,})(?:\s|\w|-)+\)
    """,
    re.VERBOSE | re.IGNORECASE,
)
EXTRA_DEPRECATED_KEYSYMS = ("Ext16bit_L", "Ext16bit_R")


@unique
class Deprecation(IntFlag):
    NONE = 0
    ALIAS = 1 << 0
    EXPLICIT = 1 << 2
    IMPLICIT = 1 << 3


def handle_keysym_match(
    keysyms: dict[int, str],
    keysyms_names: dict[str, str],
    line_nbr: int,
    m: re.Match[str],
):
    if m.group("evdev"):
        # _EVDEVK macro
        keysym = 0x10081000 + int(m.group("value"), 16)
    else:
        keysym = int(m.group("value"), 16)
    name = (m.group("prefix") or "") + m.group("name")
    alias = None
    if (comment := m.group("comment")) and (
        comment_match := KEYSYM_DEPRECATION_COMMENT_PATTERN.match(comment.strip())
    ):
        if comment_match.group("deprecated") or comment_match.group("inexact_unicode"):
            # Explicitly deprecated
            deprecated = Deprecation.EXPLICIT
        elif alias := comment_match.group("alias"):
            # Explicit alias: do not deprecate
            deprecated = Deprecation.ALIAS
        else:
            # Normal comment
            deprecated = Deprecation.NONE
    elif name in EXTRA_DEPRECATED_KEYSYMS:
        # Explicitly deprecated
        deprecated = Deprecation.EXPLICIT
    else:
        deprecated = Deprecation.NONE

    if name in keysyms_names:
        # Duplicate keysym: skip
        if verbosity:
            logger.warning(
                f"Line {line_nbr}: Keysym “{name}” 0x{keysym:0>4x} already defined; skipping."
            )
        return
    elif ref := keysyms.get(keysym):
        if deprecated & Deprecation.ALIAS:
            # Check alias has same value
            if keysyms.get(keysym) != alias:
                if verbosity:
                    if alias in keysyms_names:
                        logger.warning(
                            f"Line {line_nbr}: Keysym {name} is declared as alias of {alias}, but they have different values."
                        )
                    else:
                        logger.warning(
                            f"Line {line_nbr}: Keysym “{name}” is declared as alias of “{alias}”, but the alias does not exists. Typo?"
                        )
            keysyms_names[name] = name
        else:
            # Deprecated, because there is a previous definition with other name.
            # Ensure that the replacement keysym is supported by xkbcommon.
            deprecated |= Deprecation.IMPLICIT
            if libxkbcommon and libxkbcommon.is_invalid_keysym_name(ref):
                if verbosity:
                    logger.warning(
                        f"Line {line_nbr}: Keep deprecated keysym “{name}”; reference keysym “{ref}” is not supported by available xkbcommon."
                    )
                keysyms_names[name] = name
            else:
                keysyms_names[name] = ref
    else:
        # Reference keysym
        if deprecated & Deprecation.ALIAS:
            if verbosity:
                logger.error(
                    f"Line {line_nbr}: Explicit alias “{name}” for “{alias}” is invalid."
                )
            keysyms_names[name] = name
        elif deprecated & Deprecation.EXPLICIT:
            keysyms_names[name] = ""
        else:
            assert deprecated is Deprecation.NONE
            keysyms_names[name] = name
        keysyms[keysym] = name


def parse_keysyms_header(
    path: Path, keysyms: dict[int, str], keysyms_names: dict[str, str]
):
    with path.open("rt", encoding="utf-8") as fd:
        pending_multiline_comment = False
        for line_nbr, line in enumerate(map(lambda l: l.strip(), fd)):
            if not line:
                # Skip empty line
                pass
            elif pending_multiline_comment:
                # Continuation of a multiline comment.
                # Check if it ends on this line.
                if line.endswith("*/"):
                    pending_multiline_comment = False
            elif line.startswith("/*"):
                # Start of a multiline comment
                if not line.endswith("*/"):
                    pending_multiline_comment = True
            elif any(
                line.startswith(s)
                for s in ("#ifdef", "#ifndef", "#endif", "#define _", "#undef")
            ):
                # Skip C macros
                pass
            elif m := KEYSYM_ENTRY_PATTERN.match(line):
                # Valid keysym entry
                handle_keysym_match(keysyms, keysyms_names, line_nbr, m)
            else:
                raise ValueError(f"Cannot parse header “{path}” line: {line}")


def parse_keysyms_headers(paths: Sequence[Path]) -> dict[str, str]:
    keysyms: dict[int, str] = {}
    keysyms_names: dict[str, str] = {}
    for path in paths:
        if verbosity:
            logger.info(processing_file_message("keysym header", path))
        parse_keysyms_header(path, keysyms, keysyms_names)
    return keysyms_names


################################################################################
# Unicode names
################################################################################


def parse_unicode_name_aliases(path: Path) -> dict[str, str]:
    aliases: dict[str, str] = {}
    if verbosity:
        logger.info(processing_file_message("Unicode name aliases", path))
    with path.open("rt", encoding="utf-8") as fd:
        for line in map(lambda s: s.strip(), fd):
            # Empty line or comment
            if not line or line.startswith("#"):
                continue
            line = line.split("#")[0]
            raw_codepoint, alias, category, *_ = map(
                lambda s: s.strip(), line.split(";")
            )
            char = chr(int(raw_codepoint, 16))
            if category == "correction":
                aliases[char] = alias
    return aliases


def unicode_name(unicode_name_aliases: dict[str, str], c: str, is_first: bool) -> str:
    # We want to use Unicode *corrected* names, but the Python API does not
    # propose those. So we process the UCD by ourselves.
    name = unicode_name_aliases.get(c) or unicodedata.name(c, None)
    if name is None:
        raise ValueError(f"Cannot find Unicode name for: “{c}” (U+{ord(c):0>4X})")
    # RULE: remove “ACCENT” from the name, when the character is combining and
    #       is not in first position
    if not is_first and "COMBINING" in name and name.endswith("ACCENT"):
        return name.removesuffix(" ACCENT")
    else:
        return name


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
COMPOSE_TAG_PATTERN = re.compile(r"\s*\{([^}]+)\}")
COMPOSE_APL_PATTERN = re.compile(r"\S \S APL ")


UNICODE_KEYSYM_PATTERN = re.compile(r"\bU(?P<codepoint>[0-9A-Fa-f]+)\b")
KEYSYM_PATTERN = re.compile(r"<(\w+)>")


@unique
class ComposeTag(Enum):
    PRESERVE_COMMENT = "preserve comment"


COMPOSE_TAGS_MAPPING = {t.value: t for t in ComposeTag}


def parse_compose_tags(s: str) -> tuple[set[ComposeTag], set[str]]:
    valid: set[ComposeTag] = set()
    invalid: set[str] = set()
    for t in COMPOSE_TAG_PATTERN.findall(s):
        if tag := COMPOSE_TAGS_MAPPING.get(t):
            valid.add(tag)
        else:
            invalid.add(t)
    return valid, invalid


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


def make_comment(unicode_name_aliases: dict[str, str], s: str) -> str:
    """Make the comment of a Compose sequence, based on its result."""
    return " plus ".join(
        unicode_name(unicode_name_aliases, c, k == 0) for k, c in enumerate(s)
    )


def check_keysym(config: Configuration, n: int, keysym_name: str) -> str:
    if m := UNICODE_KEYSYM_PATTERN.match(keysym_name):
        # Reformat Unicode keysym
        codepoint = int(m.group("codepoint"), 16)
        unicode_keysym = f"U{codepoint:0>4X}"
        if libxkbcommon:
            # Find the canonical keysym name using xkbcommon
            keysym_name = libxkbcommon.char_to_keysym(chr(codepoint))
            # We keep our normalized Unicode in case xkbcommon returns a long
            # Unicode keysym, or we explicitly prefer Unicode keysyms, or
            # the named keysym is deprecated.
            if (
                unicode_keysym == keysym_name
                or keysym_name.startswith("U0")
                or config.prefer_unicode_keysym
                or config.keysyms_names.get(keysym_name) != keysym_name
            ):
                return unicode_keysym
            else:
                return keysym_name
        else:
            return unicode_keysym
    ref = config.keysyms_names.get(keysym_name)
    if keysym_name == ref:
        # Reference keysym
        return keysym_name
    elif ref is None:
        logger.error(f"Line {n}: Unsupported keysym “{keysym_name}”")
        return keysym_name
    elif ref == "":
        # Deprecated: keep keysym
        logger.warning(f"Line {n}: Deprecated keysym “{keysym_name}”.")
        return keysym_name
    else:
        # Deprecated alias: return reference keysym
        logger.warning(
            f"Line {n}: Deprecated keysym “{keysym_name}”. Please use “{ref}” instead."
        )
        return ref


def check_keysym_sequence(config: Configuration, n: int, sequence: str) -> str:
    substitutions: dict[str, str] = {}
    for keysym_name in KEYSYM_PATTERN.findall(sequence):
        keysym_nameʹ = check_keysym(config, n, keysym_name)
        if keysym_nameʹ != keysym_name:
            substitutions[keysym_name] = keysym_nameʹ
    if substitutions:
        pattern = re.compile("|".join(re.escape(k) for k in substitutions.keys()))
        return pattern.sub(lambda x: substitutions[x.group()], sequence)
    else:
        return sequence


def handle_compose_entry_match(
    config: Configuration, line_nbr: int, m: re.Match[str]
) -> str | None:
    string = unescape(m.group("string"))
    rewrite = False
    # Check sequence keysyms
    if config.keysyms_names:
        sequence = check_keysym_sequence(config, line_nbr, m.group("sequence"))
        if sequence != m.group("sequence"):
            rewrite = True
    else:
        sequence = m.group("sequence")
    # Check result keysym
    if keysym := m.group("keysym"):
        if libxkbcommon:
            keysym_char = libxkbcommon.keysym_to_char(m.group("keysym"))
            if string != keysym_char:
                logger.error(
                    f"Line {line_nbr}: The keysym does not correspond to the character: expected “{string}”, got “{keysym_char}”.",
                )
        if config.keysyms_names:
            keysym = check_keysym(config, line_nbr, m.group("keysym"))
            if keysym != m.group("keysym"):
                rewrite = True
    expected_comment = make_comment(config.unicode_name_aliases, string)
    # Check the comment
    if comment := m.group("comment"):
        # Check tags
        tags, invalid = parse_compose_tags(comment)
        if invalid:
            logger.error(f"Line {line_nbr}: Invalid tags {invalid}")
        if tags:
            logger.info(f"Line {line_nbr}: preserving comment")
            expected_comment = comment
        # Check if we have the expected comment
        # NOTE: Some APL sequences provide the combo of composed characters
        elif comment == expected_comment or (
            COMPOSE_APL_PATTERN.match(comment)
            and COMPOSE_APL_PATTERN.sub("APL ", comment) == expected_comment
        ):
            expected_comment = comment
        else:
            logger.warning(
                f"Line {line_nbr}: Expected “{expected_comment}”, "
                f"got: “{m.group('comment')}”",
            )
            rewrite = True
    else:
        # No comment: require to write it
        rewrite = True
    # Rewrite entry if necessary
    if rewrite:
        keysym = "" if keysym is None else f"\t{keysym}"
        assert (len(string) == 1 and m.group("keysym") is not None) ^ (
            len(string) > 1 and m.group("keysym") is None
        )
        comment_space = " " if len(string) == 1 else m.group("space") or "\t"
        return f"""{sequence}: "{m.group('string')}"{keysym}{comment_space}# {expected_comment}\n"""
    else:
        return None


def process_compose_lines(fd: TextIOWrapper, config: Configuration):
    multi_line_comment = False
    for line_nbr, line in enumerate(fd, start=1):
        # Handle pending multi-line comment
        if multi_line_comment:
            if line.strip().endswith("*/"):
                multi_line_comment = False
            yield line
        # Handle XCOMM comments
        elif line.startswith("XCOMM"):
            if config.convert_xcomm:
                yield "#" + line.removeprefix("XCOMM")
            else:
                yield line
        # Handle single-line comment & include
        elif not line.strip() or any(line.startswith(s) for s in ("#", "include")):
            yield line
        # Handle start of a multi-line comment
        elif line.startswith("/*"):
            # Check if one-liner
            if not line.strip().endswith("*/"):
                multi_line_comment = True
            yield line
        # Handle compose sequence
        elif m := COMPOSE_ENTRY_PATTERN.match(line):
            if lineʹ := handle_compose_entry_match(config, line_nbr, m):
                yield lineʹ
            else:
                yield line
        else:
            raise ValueError(f"Cannot parse line: “{line}”")


def process_compose_file(path: Path, config: Configuration):
    with path.open("rt", encoding="utf-8") as fd:
        yield from process_compose_lines(fd, config)


def run(
    paths: Sequence[Path],
    write: bool,
    keysyms_headers: Sequence[Path],
    name_aliases_path: Path | None,
    prefer_named_keysyms: bool,
    keep_xcomm: bool,
):
    # Keysyms headers
    keysyms_names = parse_keysyms_headers(keysyms_headers)
    # Unicode files
    unicode_name_aliases = (
        parse_unicode_name_aliases(name_aliases_path) if name_aliases_path else {}
    )
    # Set config
    config = Configuration(
        keysyms_names=keysyms_names,
        unicode_name_aliases=unicode_name_aliases,
        prefer_unicode_keysym=not prefer_named_keysyms,
        convert_xcomm=not keep_xcomm,
    )
    # Compose files
    for path in paths:
        logger.info(processing_file_message("Compose", path))
        if write:
            with tempfile.NamedTemporaryFile("wt") as fd:
                # Write to a temporary file
                fd.writelines(process_compose_file(path, config))
                fd.flush()
                # No error: now overwrite the original file
                shutil.copyfile(fd.name, path)
        else:
            for _ in process_compose_file(path, config):
                pass


def parse_args():
    parser = argparse.ArgumentParser(description="Fix Compose file formatting")
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
    parser.add_argument(
        "--unicode-name-aliases",
        type=Path,
        help="Name aliases file from the Unicode Character Database. Latest version available at: https://www.unicode.org/Public/UCD/latest/ucd/NameAliases.txt",
    )
    parser.add_argument(
        "--prefer-named-keysyms",
        action="store_true",
        help="Prefer named keysyms over Unicode keysyms",
    )
    parser.add_argument(
        "--keep-xcomm",
        action="store_true",
        help="Do NOT convert XCOMM comments to # comments",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity. Useful to see log entry of keysyms headers & Unicode files.",
    )
    parser.add_argument("--write", action="store_true", help="Write the compose file")
    return parser.parse_args()


if __name__ == "__main__":
    # Logging setup
    logFormatter = logging.Formatter("[%(levelname)s] %(message)s")
    logHandler = logging.StreamHandler(sys.stderr)
    logHandler.setFormatter(logFormatter)
    logger.addHandler(logHandler)
    logger.setLevel(logging.INFO)

    # Parse CLI args
    args = parse_args()
    verbosity = args.verbose
    if args.no_keysyms:
        keysyms = []
    elif args.keysyms:
        keysyms = args.keysyms
    else:
        keysyms = list(args.keysyms_prefix / path for path in DEFAULT_KEYSYMS_HEADERS)
    unicode_name_aliases = (
        args.unicode_name_aliases
        if args.unicode_name_aliases
        and file_only("Unicode name aliases", args.unicode_name_aliases)
        else None
    )

    run(
        list(filter(partial(file_only, "Compose"), args.input)),
        args.write,
        list(filter(partial(file_only, "keysyms header"), keysyms)),
        unicode_name_aliases,
        args.prefer_named_keysyms,
        args.keep_xcomm,
    )
