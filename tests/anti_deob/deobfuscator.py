"""Core deobfuscation logic."""

import asyncio
import collections
import concurrent.futures
import contextlib
import dataclasses
import enum
import functools
import itertools
import logging
import multiprocessing
import re
import struct
import threading
import time
import typing
import warnings

import capstone
from ida_taskr import get_logger
from ida_taskr.utils import (
    AsyncEventEmitter,
    IntervalSet,
    Range,
    log_execution_time,
    make_chunks,
    resolve_overlaps,
    shm_buffer,
)

# PyQt 5.15/6.2/6.3/6.4:
# https://riverbankcomputing.com/news/SIP_v6.7.12_Released
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=(
        r"sipPyTypeDict\(\) is deprecated, the extension module should use "
        r"sipPyTypeDictRef\(\) instead"
    ),
)

# maximum length of any stage-1 pattern (you said 129 bytes)
MAX_PATTERN_LEN = 129
MIN_PATTERN_LEN = 12
LOG_LEVEL = logging.INFO
logger = get_logger(__name__)


class PatternCategory(enum.Enum):
    MULTI_PART = enum.auto()
    SINGLE_PART = enum.auto()
    JUNK = enum.auto()


@dataclasses.dataclass
class RegexPatternMetadata:
    category: PatternCategory
    pattern: bytes  # The regex pattern as a bytes literal
    description: typing.Optional[str] = None
    compiled: typing.Optional[typing.Pattern] = None

    def compile(self, flags=0):
        """Compile the regex if not already done, and return the compiled object."""
        if self.compiled is None:
            self.compiled = re.compile(self.pattern, flags)
        return self.compiled

    @property
    def group_names(self):
        """Return the dictionary mapping group names to their indices."""
        return self.compile().groupindex


@dataclasses.dataclass
class MultiPartPatternMetadata(RegexPatternMetadata):
    category: PatternCategory = dataclasses.field(
        default=PatternCategory.MULTI_PART, init=False
    )

    def __post_init__(self):
        # Compile to ensure group names are available.
        _ = self.compile(re.DOTALL | re.VERBOSE)
        required_groups = {"first_jump", "padding", "second_jump"}
        missing = required_groups - set(self.group_names)
        if missing:
            raise ValueError(f"MultiPart pattern is missing required groups: {missing}")


@dataclasses.dataclass
class SinglePartPatternMetadata(RegexPatternMetadata):
    category: PatternCategory = dataclasses.field(
        default=PatternCategory.SINGLE_PART, init=False
    )

    def __post_init__(self):
        _ = self.compile(re.DOTALL | re.VERBOSE)
        required_groups = {"prefix", "padding", "jump"}
        missing = required_groups - set(self.group_names)
        if missing:
            raise ValueError(
                f"SinglePart pattern is missing required groups: {missing}"
            )


@dataclasses.dataclass
class JunkPatternMetadata(RegexPatternMetadata):
    category: PatternCategory = dataclasses.field(
        default=PatternCategory.JUNK, init=False
    )

    def __post_init__(self):
        _ = self.compile(re.DOTALL | re.VERBOSE)
        required_groups = {"junk"}
        missing = required_groups - set(self.group_names)
        if missing:
            raise ValueError("Junk pattern must have a 'junk' group.")


# fmt: off
# (We do not wrap this in a named group here so that we can reuse it inside other groups.)
PADDING_PATTERN = rb"""
    (?:
        \xC0[\xE0-\xFF]\x00  # 3-byte SHL reg, 0 with random register encoding
        |                     # OR
        (?:\x86|\x8A)        # 2-byte XCHG or MOV instruction
        [\xC0\xC9\xD2\xDB\xE4\xED\xF6\xFF]  # Specific register encodings
    )
"""
    
# Multi-part jump patterns: pairs of conditional jumps with optional padding
MULTI_PART_PATTERNS = [
    MultiPartPatternMetadata(rb"(?P<first_jump>\x70.)(?P<padding>" + PADDING_PATTERN + rb")*(?P<second_jump>\x71.)", "JO ... JNO"),
    MultiPartPatternMetadata(rb"(?P<first_jump>\x71.)(?P<padding>" + PADDING_PATTERN + rb")*(?P<second_jump>\x70.)", "JNO ... JO"),
    MultiPartPatternMetadata(rb"(?P<first_jump>\x72.)(?P<padding>" + PADDING_PATTERN + rb")*(?P<second_jump>\x73.)", "JB ... JAE"),
    MultiPartPatternMetadata(rb"(?P<first_jump>\x73.)(?P<padding>" + PADDING_PATTERN + rb")*(?P<second_jump>\x72.)", "JAE ... JB"),
    MultiPartPatternMetadata(rb"(?P<first_jump>\x74.)(?P<padding>" + PADDING_PATTERN + rb")*(?P<second_jump>\x75.)", "JE ... JNE"),
    MultiPartPatternMetadata(rb"(?P<first_jump>\x75.)(?P<padding>" + PADDING_PATTERN + rb")*(?P<second_jump>\x74.)", "JNE ... JE"),
    MultiPartPatternMetadata(rb"(?P<first_jump>\x76.)(?P<padding>" + PADDING_PATTERN + rb")*(?P<second_jump>\x77.)", "JBE ... JA"),
    MultiPartPatternMetadata(rb"(?P<first_jump>\x77.)(?P<padding>" + PADDING_PATTERN + rb")*(?P<second_jump>\x76.)", "JA ... JBE"),
    MultiPartPatternMetadata(rb"(?P<first_jump>\x78.)(?P<padding>" + PADDING_PATTERN + rb")*(?P<second_jump>\x79.)", "JS ... JNS"),
    MultiPartPatternMetadata(rb"(?P<first_jump>\x79.)(?P<padding>" + PADDING_PATTERN + rb")*(?P<second_jump>\x78.)", "JNS ... JS"),
    MultiPartPatternMetadata(rb"(?P<first_jump>\x7A.)(?P<padding>" + PADDING_PATTERN + rb")*(?P<second_jump>\x7B.)", "JP ... JNP"),
    MultiPartPatternMetadata(rb"(?P<first_jump>\x7B.)(?P<padding>" + PADDING_PATTERN + rb")*(?P<second_jump>\x7A.)", "JNP ... JP"),
    MultiPartPatternMetadata(rb"(?P<first_jump>\x7C.)(?P<padding>" + PADDING_PATTERN + rb")*(?P<second_jump>\x7D.)", "JL ... JGE"),
    MultiPartPatternMetadata(rb"(?P<first_jump>\x7D.)(?P<padding>" + PADDING_PATTERN + rb")*(?P<second_jump>\x7C.)", "JGE ... JL"),
    MultiPartPatternMetadata(rb"(?P<first_jump>\x7E.)(?P<padding>" + PADDING_PATTERN + rb")*(?P<second_jump>\x7F.)", "JLE ... JG"),
    MultiPartPatternMetadata(rb"(?P<first_jump>\x7F.)(?P<padding>" + PADDING_PATTERN + rb")*(?P<second_jump>\x7E.)", "JG ... JLE"),
]

# Single-part jump patterns: prefix instruction + optional padding + conditional jump
SINGLE_PART_PATTERNS = [
    SinglePartPatternMetadata(rb"(?P<prefix>\x0C\x00)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x71.)", "OR AL, 0x00 ... JNO"),
    SinglePartPatternMetadata(rb"(?P<prefix>\x0C\x00)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x73.)", "OR AL, 0x00 ... JAE"),
    SinglePartPatternMetadata(rb"(?P<prefix>\x24\xFF)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x71.)", "AND AL, 0xFF ... JNO"),
    SinglePartPatternMetadata(rb"(?P<prefix>\x24\xFF)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x73.)", "AND AL, 0xFF ... JAE"),
    SinglePartPatternMetadata(rb"(?P<prefix>\x34\x00)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x71.)", "XOR AL, 0x00 ... JNO"),
    SinglePartPatternMetadata(rb"(?P<prefix>\x34\x00)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x73.)", "XOR AL, 0x00 ... JAE"),
    SinglePartPatternMetadata(rb"(?P<prefix>\x80[\xC8-\xCF]\x00)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x71.)", "OR r/m8, 0x00 ... JNO"),
    SinglePartPatternMetadata(rb"(?P<prefix>\x80[\xC8-\xCF]\x00)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x73.)", "OR r/m8, 0x00 ... JAE"),
    SinglePartPatternMetadata(rb"(?P<prefix>\x80[\xE0-\xE7]\xFF)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x71.)", "AND r/m8, 0xFF ... JNO"),
    SinglePartPatternMetadata(rb"(?P<prefix>\x80[\xE0-\xE7]\xFF)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x73.)", "AND r/m8, 0xFF ... JAE"),
    SinglePartPatternMetadata(rb"(?P<prefix>\x80[\xF0-\xF7]\x00)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x71.)", "XOR r/m8, 0x00 ... JNO"),
    SinglePartPatternMetadata(rb"(?P<prefix>\x80[\xF0-\xF7]\x00)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x73.)", "XOR r/m8, 0x00 ... JAE"),
    SinglePartPatternMetadata(rb"(?P<prefix>\x84.)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x71.)", "TEST r/m8, r8 ... JNO"),
    SinglePartPatternMetadata(rb"(?P<prefix>\x84.)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x73.)", "TEST r/m8, r8 ... JAE"),
    SinglePartPatternMetadata(rb"(?P<prefix>\x85.)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x71.)", "TEST r/m32, r32 ... JNO"),
    SinglePartPatternMetadata(rb"(?P<prefix>\x85.)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x73.)", "TEST r/m32, r32 ... JAE"),
    SinglePartPatternMetadata(rb"(?P<prefix>\xA8.)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x71.)", "TEST AL, imm8 ... JNO"),
    SinglePartPatternMetadata(rb"(?P<prefix>\xA8.)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x73.)", "TEST AL, imm8 ... JAE"),
    SinglePartPatternMetadata(rb"(?P<prefix>\xA9....)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x71.)", "TEST EAX, imm32 ... JNO"),
    SinglePartPatternMetadata(rb"(?P<prefix>\xA9....)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x73.)", "TEST EAX, imm32 ... JAE"),
    SinglePartPatternMetadata(rb"(?P<prefix>\xF6..)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x71.)", "TEST r/m8, imm8 ... JNO"),
    SinglePartPatternMetadata(rb"(?P<prefix>\xF6..)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x73.)", "TEST r/m8, imm8 ... JAE"),
    SinglePartPatternMetadata(rb"(?P<prefix>\xF7.....)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x71.)", "TEST r/m32, imm32 ... JNO"),
    SinglePartPatternMetadata(rb"(?P<prefix>\xF7.....)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x73.)", "TEST r/m32, imm32 ... JAE"),
    SinglePartPatternMetadata(rb"(?P<prefix>\xF8)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x73.)", "CLC ... JAE"),
    SinglePartPatternMetadata(rb"(?P<prefix>\xF9)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x72.)", "STC ... JB"),
    SinglePartPatternMetadata(rb"(?P<prefix>\xF9)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>\x76.)", "STC ... JBE"),
    # CMP ESP,0x1C00h – always “above” on a real Win x64 stack, then JA short
    # # 81 FC 00 44 00 00
    # # 81 FC 00 3F 00 00
    # # 81 FC 00 44 00 00
    SinglePartPatternMetadata(rb"(?P<prefix>[\x80\x81\x83]\xFC\x00...)(?P<padding>" + PADDING_PATTERN + rb")?(?P<jump>(\x77|\x73).)", "CMP ESP,0x1C00 … JA/E"),
]



# Helper sets for checking specific registers based on the regex patterns
# These patterns [\xC0-\xC3\xC5-\xC7] and [\xD8-\xDB\xDD-\xDF] and [\xE8-\xEB\xED-\xEF]
# correspond to ModR/M byte where MOD=11 (register) and R/M is 0-3 (EAX, ECX, EDX, EBX)
# or 5-7 (EBP, ESI, EDI). R/M=4 is ESP, which is skipped by these ranges.
# For 8-bit, these are AL, CL, DL, BL, BPL, SIL, DIL.
# Since REX prefixes are confirmed *not* to be used with these patterns,
# we only need to check for the 8-bit and 32-bit registers.
REG_32_SET = {
    capstone.x86.X86_REG_EAX,
    capstone.x86.X86_REG_ECX,
    capstone.x86.X86_REG_EDX,
    capstone.x86.X86_REG_EBX,
    capstone.x86.X86_REG_EBP,
    capstone.x86.X86_REG_ESI,
    capstone.x86.X86_REG_EDI,
}

REG_8_SET = {
    capstone.x86.X86_REG_AL,
    capstone.x86.X86_REG_CL,
    capstone.x86.X86_REG_DL,
    capstone.x86.X86_REG_BL,
    # AH (C4) is skipped by the regex range
    capstone.x86.X86_REG_CH,  # C5
    capstone.x86.X86_REG_DH,  # C6
    capstone.x86.X86_REG_BH,  # C7
    # Low bytes of SI, DI, BP, SP are only accessible with REX in 64-bit
    # but the regex implies non-REX. The original set included BPL, SIL, DIL
    # which correspond to ModR/M 101, 110, 111 when MOD != 11.
    # The regex range C0-C3, C5-C7 *specifically* uses MOD=11.
    # So, the correct registers are AL, CL, DL, BL, CH, DH, BH.
    # Let's redefine REG_8_SET based *only* on the registers implied by
    # the specific ModR/M bytes in the regexes when MOD=11.
}

# Helper to check if the first operand is a register from the allowed set
# based on the ModR/M ranges implied by the regexes.
# Assumes no REX prefixes are used with these specific junk patterns,
# so we only check against the 8-bit and 32-bit sets.
def is_allowed_reg(operands):
    if not operands or operands[0].type != capstone.CS_OP_REG:
        return False
    reg = operands[0].reg
    # Check if the register is one of the 8-bit or 32-bit registers
    # corresponding to the ModR/M R/M field 0-3, 5-7 when MOD=11,
    # which is the set derived from ModR/M=11 ranges
    return reg in REG_8_SET or reg in REG_32_SET

# Helper to check if there's an immediate operand
def has_imm_operand(operands):
    return any(op.type == capstone.CS_OP_IMM for op in operands)

# Function to check if a byte is a valid REX prefix (0x40-0x4F)
def is_rex_prefix(byte):
    return 0x40 <= byte <= 0x4F

# Function to check if a byte is a valid ModR/M byte (0x80-0xBF)
def is_valid_modrm(byte):
    return 0x80 <= byte <= 0xBF
# fmt: on


class SegmentType(enum.Enum):
    STAGE1_MULTIPLE = enum.auto()
    STAGE1_SINGLE = enum.auto()
    JUNK = enum.auto()
    BIG_INSTRUCTION = enum.auto()


@dataclasses.dataclass
class MatchSegment:
    start: int
    length: int
    description: str
    matched_bytes: bytes
    segment_type: SegmentType
    matched_groups: dict = dataclasses.field(default_factory=dict)


class MatchChain:
    def __init__(self, base_address: int, segments: list[MatchSegment] | None = None):
        self.base_address = base_address
        self.segments = segments or []

    def add_segment(self, segment: MatchSegment):
        self.segments.append(segment)

    def overall_start(self) -> int:
        return self.segments[0].start + self.base_address if self.segments else 0

    def overall_length(self) -> int:
        if not self.segments:
            return 0
        first = self.segments[0]
        last = self.segments[-1]
        return (last.start + last.length) - first.start

    def overall_matched_bytes(self) -> bytes:
        return b"".join(seg.matched_bytes for seg in self.segments)

    def append_junk(
        self, junk_start: int, junk_len: int, junk_desc: str, junk_bytes: bytes
    ):
        seg = MatchSegment(
            start=junk_start,
            length=junk_len,
            description=junk_desc,
            matched_bytes=junk_bytes,
            segment_type=SegmentType.JUNK,
        )
        self.add_segment(seg)

    @property
    def description(self) -> str:
        desc = []
        for idx, seg in enumerate(self.segments):
            if idx == 0:
                desc.append(f"{seg.description}")
            else:
                desc.append(f" -> {seg.description}")
        return "".join(desc)

    def update_description(self, new_desc: str):
        if self.segments:
            self.segments[0].description = new_desc

    # New properties for junk analysis
    @property
    def stage1_type(self) -> SegmentType:
        return self.segments[0].segment_type

    @property
    def junk_segments(self) -> list:
        """
        Returns a list of segments considered as junk based on their segment_type.
        """
        return [seg for seg in self.segments if seg.segment_type == SegmentType.JUNK]

    @property
    def junk_starts_at(self) -> typing.Optional[int]:
        """
        Returns the starting address of the junk portion.
        This is computed as base_address + the offset of the first junk segment.
        If no junk segments exist, returns None.
        """
        js = self.junk_segments
        if js:
            return self.base_address + js[0].start
        return None

    @property
    def junk_length(self) -> int:
        """
        Returns the total length of the junk portion.
        This is computed as the difference between the end (start + length) of the last junk segment
        and the start of the first junk segment.
        If there are no junk segments, returns 0.
        """
        js = self.junk_segments
        if not js:
            return 0
        first = js[0]
        last = js[-1]
        return (last.start + last.length) - first.start

    def __lt__(self, other):
        return self.overall_start() < other.overall_start()

    def __repr__(self):
        r = [
            f"{self.description.rjust(32, ' ')} @ 0x{self.overall_start():X} - "
            f"{self.overall_matched_bytes().hex()[:16]}"
            f"{'...' if self.overall_length() > 16 else ''}",
            "  |",
        ]
        for seg in self.segments:
            _grps = f"{' - ' + str(seg.matched_groups) if seg.matched_groups else ''}"
            r.append(
                f"  |_ {seg.description} @ 0x{self.base_address + seg.start:X} - {seg.matched_bytes.hex()}{_grps}"
            )
        return "\n".join(r)


# TODO: inherit from list?
class MatchChains:
    def __init__(self):
        self.chains: list[MatchChain] = []

    def add_chain(self, chain: MatchChain):
        self.chains.append(chain)

    def __iter__(self):
        yield from self.chains

    def sort(self):
        self.chains.sort(key=lambda x: x.overall_start())

    def __len__(self):
        return len(self.chains)

    def __repr__(self):
        lines = []
        for c in self.chains:
            desc = c.description
            off = c.overall_start()
            bhex = c.overall_matched_bytes().hex()[:16]
            tail = "…" if c.overall_length() > 16 else ""
            lines.append(f"{desc.rjust(32)} @ 0x{off:X} - {bhex}{tail}")
        return "\n".join(lines)


def _stage1_scan_one(job: tuple[RegexPatternMetadata, int, memoryview]) -> MatchChains:
    """
    job = (pattern_bytes, description, segment_type, base_ea, buf)
    returns MatchChains
    """
    rgx, base_ea, buf = job
    prog = rgx.compile()
    hits = MatchChains()

    for m in prog.finditer(buf):
        s = m.start()
        e = m.end()
        mb = buf[s:e]
        match_len = e - s

        groups = {
            k: v.hex()
            for k, v in m.groupdict().items()
            if (v is not None and k != "padding")
        }

        # compute jump targets
        if "jump" in groups:
            offset = struct.unpack("<b", mb[-1:])[0]
            tgt = base_ea + s + match_len + offset
            groups["target"] = hex(tgt)
        elif "first_jump" in groups:
            off1 = struct.unpack("<b", mb[1:2])[0]
            groups["first_target"] = hex(base_ea + s + 2 + off1)
            off2 = struct.unpack("<b", mb[-1:])[0]
            groups["second_target"] = hex(base_ea + s + match_len + off2)

        seg = MatchSegment(
            start=s,
            length=match_len,
            description=rgx.description or "",
            matched_bytes=mb.tobytes(),
            segment_type=(
                SegmentType.STAGE1_MULTIPLE
                if rgx.category == PatternCategory.MULTI_PART
                else SegmentType.STAGE1_SINGLE
            ),
            matched_groups=groups,
        )
        hits.add_chain(MatchChain(base_address=base_ea, segments=[seg]))

    return hits


def stage1_find_patterns(
    buf: memoryview, base_ea: int, mp: bool = False
) -> list[MatchChain]:
    """
    Parallel regex-based Stage 1. Returns List[MatchChain].
    """
    jobs = [
        (rgx, base_ea, buf)
        for rgx in itertools.chain(MULTI_PART_PATTERNS, SINGLE_PART_PATTERNS)
    ]

    if mp:
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(mp_context=ctx) as exe:
            all_groups = exe.map(_stage1_scan_one, jobs)
    else:
        all_groups = list(map(_stage1_scan_one, jobs))

    # flatten
    out = [chain for group in all_groups for chain in group]
    # sort
    out.sort(key=lambda c: c.overall_start())
    return out


# ─── Stage 2: peel off junk via Capstone ────────────────────────────────────


@dataclasses.dataclass
class JunkInstruction:
    """Holds information about a single peeled junk instruction."""

    #: Offset relative to the start of the peeled buffer
    start_offset: int
    #: Length of the peeled junk instruction
    length: int
    #: Description of the peeled junk instruction
    description: str
    #: Bytes of the peeled junk instruction
    matched_bytes: bytes


class CapstoneDisasmContext:
    """
    Context manager and iterable for Capstone disassembly.

    Usage:
        with CapstoneDisasmContext(is_64, buf) as disasm_ctx:
            for insn in disasm_ctx:
                ...

    On error, iteration yields nothing.
    """

    _EMPTY_SET = set()

    def __init__(self, is_64: bool):
        self.is_64 = is_64

    @functools.cached_property
    def md(self):
        """
        Cached property for the Capstone disassembler instance.
        Returns None if initialization fails.
        """
        try:
            md = capstone.Cs(
                capstone.CS_ARCH_X86,
                capstone.CS_MODE_64 if self.is_64 else capstone.CS_MODE_32,
            )
            md.detail = True
            return md
        except Exception as e:
            logger.error(f"Failed to initialize Capstone: {e}")
            return None

    def __enter__(self):
        # No setup needed; iteration is handled in __iter__
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # The __exit__ method is not a generator and should not yield.
        # Instead, handle exceptions by logging them if present.
        PROPOGATE = False
        SUPPRESS = True

        match exc_type:
            case None:
                return PROPOGATE
            case capstone.CsError:
                logger.error(f"Capstone disassembly error: {exc_val}", exc_info=True)
                return SUPPRESS
            case _:
                logger.error(
                    f"Unexpected {exc_type.__name__} during Capstone disassembly: {exc_val}",
                    exc_info=True,
                )
                return SUPPRESS

    def disasm(self, buf: bytes, start_ea: int, **kwargs):
        """
        Disassemble the buffer and return an iterator of instructions.
        If initialization failed, returns an empty iterator.
        start_ea = 0 means is relative to the start of the input buffer 'buf'
        """
        if not self.md:
            yield from self._EMPTY_SET
            return

        for insn in self.md.disasm(buf, start_ea, **kwargs):
            yield insn


@dataclasses.dataclass
class BasicDecodedInstruction:
    """Holds standardized information about a decoded instruction."""

    address: int
    size: int
    is_jump: bool = False
    jump_target: typing.Optional[int] = None
    is_nop: bool = False
    dead_opaque_predicate: bool = False


class InstructionDecoder(typing.Protocol):
    """Protocol defining the expected signature for decoder functions."""

    def __init__(self, is_x64: bool): ...

    def decode(
        self, ea: int, mem_bytes_at_ea: bytes
    ) -> typing.Optional[BasicDecodedInstruction]:
        """
        Decodes the instruction at virtual address 'ea' using the provided memory bytes.

        Args:
            ea: The virtual address of the instruction to decode.
            mem_bytes_at_ea: A bytes object containing memory starting from 'ea'.
                             The implementation should only consume the bytes
                             needed for the single instruction at 'ea'.

        Returns:
            An InstructionInfo object if decoding is successful, otherwise None.
        """
        ...


class CapstoneInstructionDecoder(InstructionDecoder):

    def __init__(self, is_x64: bool):
        self.is_x64 = is_x64
        self.md = capstone.Cs(
            capstone.CS_ARCH_X86, capstone.CS_MODE_64 if is_x64 else capstone.CS_MODE_32
        )
        self.md.detail = True

    def get_next_insn(
        self, mem_bytes_at_ea: bytes, ea: int
    ) -> typing.Optional[capstone.CsInsn]:
        try:
            # Use list comprehension and next to get the first instruction or None
            insn = next(self.md.disasm(mem_bytes_at_ea, ea, count=1), None)
        except capstone.CsError as e:
            logger.error(f"Capstone decoding error at 0x{ea:X}: {e}")
            return None
        if insn is None:
            return None
        logger.debug(
            "Decoded instruction: %s %s (%X bytes) at offset %s - bytes: %s",
            insn.mnemonic,
            insn.op_str,
            insn.size,
            hex(ea),
            insn.bytes.hex(),
        )
        return insn

    def decode(
        self, ea: int, mem_bytes_at_ea: bytes
    ) -> typing.Optional[BasicDecodedInstruction]:
        """
        Decodes instruction at ea using IDA's disassembler.
        Ignores mem_bytes_at_ea, uses IDA's database.
        Conforms to DecoderProtocol.
        """
        # Decode using Capstone
        insn = self.get_next_insn(mem_bytes_at_ea, ea)
        if insn is None:
            return None

        decoded = BasicDecodedInstruction(address=ea, size=insn.size)
        if insn.id == capstone.x86.X86_INS_NOP:
            decoded.is_nop = True
        # Check for 'xchg r8, r8' as a NOP pattern (0x90 is 'nop', i.e. 0x87 C9 is 'xchg cl, cl')
        elif insn.id in (
            capstone.x86.X86_INS_XCHG,
            capstone.x86.X86_INS_MOV,
            capstone.x86.X86_GRP_CMOV,
        ):
            op1, op2 = insn.operands
            if op1.type == op2.type and op1.size == op2.size and op1.reg == op2.reg:
                decoded.is_nop = True
        elif insn.id == capstone.x86.X86_INS_PUSH and len(insn.operands) > 0:
            # we have encountered this dead code:
            # .text:0000000180188FB2 50                                                  push    rax
            # .text:0000000180188FB3 EB FF                                               jmp     short near ptr loc_180188FB3+1
            # .text:0000000180188FB5 C0 58 ? ?                                           rcr     byte ptr [rax-?], ?
            if insn.operands[0].reg == capstone.x86.X86_REG_RAX:
                next_insn = self.get_next_insn(
                    mem_bytes_at_ea, insn.address + insn.size
                )
                if next_insn is not None and self._is_self_recursive_jump(insn):
                    next_next_insn = self.get_next_insn(
                        mem_bytes_at_ea, next_insn.address + next_insn.size
                    )
                    if next_next_insn is not None and next_next_insn.bytes.startswith(
                        b"\xc0\x58"
                    ):
                        decoded.dead_opaque_predicate = True
                        decoded.size = insn.size + next_insn.size + 2
                    return decoded
                else:
                    decoded.is_nop = True
                    return decoded
        elif capstone.CS_GRP_JUMP in insn.groups:
            if (
                len(insn.operands) > 0
                and insn.operands[0].type == capstone.x86.X86_OP_IMM
            ):
                decoded.is_jump = True
                decoded.jump_target = insn.operands[0].imm
        return decoded

    def _is_self_recursive_jump(self, insn: capstone.CsInsn) -> bool:
        """
        Heuristic detection of self-recursive jumps for Capstone.

        Args:
            insn: Capstone instruction object

        Returns:
            True if this appears to be a self-recursive jump
        """
        jump_source = insn.address
        # Pattern detection for common dead opaque predicates
        # EB FF - jump back 1 byte (into same instruction)
        if (
            insn.id == capstone.x86.X86_INS_JMP
            and len(insn.bytes) == 2
            and insn.bytes[0] == 0xEB
            and insn.bytes[1] == 0xFF
        ):
            logger.debug(f"Self-recursive jump detected: EB FF at 0x{jump_source:X}")
            return True

        return False


@dataclasses.dataclass
class JumpTargetAnalyzer:
    # Input parameters for processing jumps.
    match_bytes: bytes  # The bytes in which we're matching jump instructions.
    match_start: int  # The address where match_bytes starts.
    block_end: int  # End address of the allowed region.
    start_ea: int  # Base address of the memory block (used for bounds checking).

    # Internal structures.
    jump_targets: collections.Counter = dataclasses.field(
        init=False, default_factory=collections.Counter
    )
    jump_details: list = dataclasses.field(
        init=False, default_factory=list
    )  # List of (jump_ea, final_target, stage1_type)
    target_type: dict = dataclasses.field(
        init=False, default_factory=dict
    )  # final_target -> stage1_type

    def follow_jump_chain(
        self,
        mem: bytes,
        current_ea: int,
        match_end: int,
        decoder: InstructionDecoder,
        visited: set[int] | None = None,
        depth: int = 0,
    ) -> typing.Optional[int]:
        """
        Follow a chain of 2-byte jumps starting from current_ea using the provided decoder.

        Args:
            mem: Memory object containing the relevant byte data. Its 'base' attribute
                 defines the absolute address corresponding to the start of its buffer.
            current_ea: The absolute starting virtual address for tracing.
            match_end: The absolute end address (exclusive) of the 'stage1' area.
            decoder: A function conforming to DecoderProtocol used for disassembly.
            visited: Set of visited addresses to prevent loops (internal use).
            depth: Recursion depth for logging (internal use).

        Returns:
            The absolute virtual address where the jump chain ends, or None.
        """
        indent = "  " * depth + "|_ "
        if visited is None:
            visited = set()

        # Get an efficient view of the memory buffer
        mem_view = mem
        mem_start_ea = self.start_ea  # Absolute start address of the buffer
        mem_len = len(mem_view)
        mem_end_ea = mem_start_ea + mem_len  # Absolute end address (exclusive)

        if current_ea in visited:
            logger.debug(
                "%sJump chain stopped: Already visited 0x%X", indent, current_ea
            )
            return None
        # Check if start address is within the bounds defined by the Memory object
        if not (mem_start_ea <= current_ea < mem_end_ea):
            logger.debug(
                "%sJump chain stopped: Start address 0x%X is outside Memory bounds [0x%X, 0x%X)",
                indent,
                current_ea,
                mem_start_ea,
                mem_end_ea,
            )
            return None

        visited.add(current_ea)

        trace_ea = current_ea
        while True:
            # Check if the current tracing address is still within the Memory bounds
            if not (mem_start_ea <= trace_ea < mem_end_ea):
                logger.debug(
                    "%sStopping trace: Address 0x%X is outside Memory bounds [0x%X, 0x%X). Returning last valid start: 0x%X",
                    indent,
                    trace_ea,
                    mem_start_ea,
                    mem_end_ea,
                    current_ea,
                )
                return current_ea  # Return the start address of the sequence that led out of bounds

            decoded_insn = None
            # Calculate offset relative to the start of the Memory object's buffer
            offset = trace_ea - mem_start_ea
            logger.debug("%soffset: %X", indent, offset)
            # We already know offset is >= 0 because trace_ea >= mem_start_ea
            # We need to ensure we have enough bytes left for *potential* instructions

            # Get bytes starting from the offset using the memoryview slice
            # Convert the slice to bytes for the decoder interface
            bytes_for_decoder = mem_view[offset:]
            if (
                not bytes_for_decoder
            ):  # Should not happen if bounds check is correct, but defensive check
                logger.warning(
                    "%sNo bytes available for decoding at offset %X (address 0x%X). Stopping trace.",
                    indent,
                    offset,
                    trace_ea,
                )
                return current_ea

            try:
                # Call the passed-in decoder function
                decoded_insn = decoder.decode(trace_ea, bytes_for_decoder)
            except Exception as e:
                logger.error(
                    "%sDecoder function raised exception at 0x%X: %s",
                    indent,
                    trace_ea,
                    e,
                )
                decoded_insn = None  # Treat as decode failure

            # If decoding failed or decoder returned None
            if not decoded_insn:
                logger.debug(
                    "%sFailed to decode instruction at 0x%X. Stopping trace. Returning start: 0x%X",
                    indent,
                    trace_ea,
                    current_ea,
                )
                return current_ea  # Return start of the sequence

            # --- Process the decoded instruction ---
            if decoded_insn.is_nop:
                logger.debug(
                    "%sNOP found at 0x%X (size %X). Skipping.",
                    indent,
                    trace_ea,
                    decoded_insn.size,
                )
                trace_ea += decoded_insn.size
                continue  # Continue the while loop to the next instruction

            if decoded_insn.dead_opaque_predicate:
                logger.debug(
                    "%sDead opaque predicate found at 0x%X (size %X). Returning start: 0x%X.",
                    indent,
                    trace_ea,
                    decoded_insn.size,
                    current_ea,
                )
                return current_ea + decoded_insn.size

            if not decoded_insn.is_jump or decoded_insn.size != 2:
                logger.debug(
                    "%sChain stopped at 0x%X: Instruction is not a 2-byte jump. Returning start: 0x%X",
                    indent,
                    trace_ea,
                    current_ea,
                )
                return current_ea  # Return the start address of the sequence that ended

            # --- We have a 2-byte jump ---
            target = decoded_insn.jump_target  # This is an absolute address
            if target is not None:
                logger.debug(
                    "%s  -> Found 2-byte jump at 0x%X targeting 0x%X",
                    indent,
                    trace_ea,
                    target,
                )

                # --- Decide action based on the jump target (using absolute addresses) ---
                # 1. Target is within the 'followable' range [match_start, match_end )
                if self.match_start <= target < match_end:
                    logger.debug(
                        "%sFollowing jump from 0x%X to 0x%X (recursive call)",
                        indent,
                        trace_ea,
                        target,
                    )
                    # Pass the same Memory object and decoder down recursively
                    return self.follow_jump_chain(
                        mem, target, match_end, decoder, visited, depth + 1
                    )

                # 3. Target is within the overall Memory block, but *before* match_start.
                elif mem_start_ea <= target < self.match_start:
                    logger.debug(
                        "%sJump chain ends: Target 0x%X is within Memory bounds [0x%X,0x%X) but outside followable range [0x%X, 0x%X). Returning target.",
                        indent,
                        target,
                        mem_start_ea,
                        mem_end_ea,
                        self.match_start,
                        match_end,
                    )
                    if depth == 0:  # this is a bs jump, ignore it.
                        return None
                    return target  # Return the target address itself

            # 4. Target is out of the overall Memory bounds or otherwise unexpected.
            logger.debug(
                "%sJump chain stopped: Target 0x%X is outside allowed ranges. Returning start address 0x%X",
                indent,
                target,
                current_ea,
            )
            if depth == 0:  # this is a bs jump, ignore it.
                return None
            return current_ea  # Return the start address of the sequence containing the invalid jump

    def _decode_stream(self, decoder, start, match_bytes):
        offset = 0
        n = len(match_bytes)

        while offset < n:
            try:
                # hand the decoder only the bytes we haven’t consumed yet
                insn = decoder.decode(start + offset, match_bytes[offset:])
            except Exception as e:
                logger.error("Decode error @0x%X: %s", start + offset, e)
                return

            if not insn:
                return

            yield insn
            offset += insn.size

    def process(self, mem, chain, is_x64: bool):
        """
        Process each jump match in match_bytes.
        'chain' is expected to have attributes:
          - junk_length: int
          - stage1_type: SegmentType
        """
        decoder = CapstoneInstructionDecoder(is_x64)
        match_end = chain.overall_start() + MAX_PATTERN_LEN
        logger.debug(
            "Processing jumps for chain @ 0x%X, match_end=0x%X",
            chain.overall_start(),
            match_end,
        )
        match chain.stage1_type:
            case SegmentType.STAGE1_SINGLE:
                jump_offset = chain.segments[0].length - (
                    len(chain.segments[0].matched_groups["jump"]) // 2
                )
                jump_ea = self.match_start + jump_offset
            case SegmentType.STAGE1_MULTIPLE:
                jump_offset = 0
                jump_ea = self.match_start + jump_offset
            case _:
                raise ValueError(f"Invalid stage1_type: {chain.stage1_type}")

        final_target = self.follow_jump_chain(mem, jump_ea, match_end, decoder)
        if not final_target:
            logger.debug(
                "  Skipping jump at 0x%X: Invalid final target 0x%X",
                jump_ea,
                final_target if final_target else 0,
            )
        else:
            self.jump_targets[final_target] += 1
            if final_target not in self.target_type:
                self.target_type[final_target] = chain.stage1_type
            self.jump_details.append((jump_ea, final_target, chain.stage1_type))
            logger.debug("Found jump @0x%X → 0x%X", jump_ea, final_target)
        return self
        # for insn in self._decode_stream(
        #     decoder, chain.overall_start(), self.match_bytes
        # ):
        #     # 1) filter out non jumps or non 2-byte jumps
        #     if not insn.is_jump or insn.size != 2:
        #         continue

        #     final_target = self.follow_jump_chain(mem, insn.address, match_end, decoder)
        #     if not final_target:
        #         logger.debug("Bad target @0x%X", insn.address)
        #         continue

        #     if abs(final_target - match_end) > 6:
        #         logger.debug("Out of range @0x%X", insn.address)
        #         continue

        #     # 2) process the hit
        #     self.jump_targets[final_target] += 1
        #     if final_target not in self.target_type:
        #         self.target_type[final_target] = chain.stage1_type
        #     self.jump_details.append((insn.address, final_target, chain.stage1_type))
        #     logger.debug("Found jump @0x%X → 0x%X", insn.address, final_target)

        return self

    def __iter__(self):
        """
        Iterate over the most likely targets.
        For each candidate, if a jump exists whose starting address equals candidate + 1,
        yield its final target instead.

        Sorting is by count descending, then by final_target descending.
        """
        # Prepare a list of (final_target, count) tuples
        results = list(self.jump_targets.items())
        # Sort by count descending, then by final_target descending
        results.sort(key=lambda x: (x[1], x[0]), reverse=True)
        for candidate, count in results:
            final_candidate = candidate
            for jump_ea, target, stype in self.jump_details:
                if jump_ea == candidate + 1:
                    final_candidate = target
                    break
            yield final_candidate


def _analyze_chain(
    chain: MatchChain,
    mem: bytes,
    start_ea: int,
    is_x64: bool,
    max_size: int = MAX_PATTERN_LEN,
) -> list[Range]:
    """
    Filter out false positive anti-disassembly patterns and handle overlaps.
    Integrates with existing big instruction detection code.

    Args:
        chains: List of MatchChain objects
        mem: Memory object containing binary data
        start_ea: Starting effective address
        max_size: Maximum valid size for an anti-disassembly routine (default: MAX_PATTERN_LEN)

    Returns:
        A single validated MatchChain object
    """

    # Find the big instruction
    match_start = chain.overall_start()
    chain_end = match_start + max_size
    ranges = []

    logger.info(f"Analyzing match: {chain.description} @ 0x{match_start:X}")

    # Determine possible jump targets - using your existing code
    jump_targets = JumpTargetAnalyzer(
        chain.overall_matched_bytes(), match_start, chain_end, start_ea
    ).process(mem=mem, chain=chain, is_x64=is_x64)

    for target in jump_targets:
        if target <= match_start:  # sanity-check
            continue
        logger.info(f"most_likely_target: 0x{target:X}, block_end: 0x{chain_end:X}")
        ranges.append(Range(match_start, target))
    return ranges


# ─── Async deobfuscator ───────────────────────────────────


def process_chunk(args):
    """
    Entire 4-stage pipeline over one overlapping chunk.
    Returns only those chains whose start is in the chunk's core region.
    """
    shm_name, padded_start, padded_end, core_start, core_end, base_ea, is_64 = args

    core_valid = []

    # attach shared memory
    with shm_buffer(shm_name) as shm:
        # zero-copy view of the chunk
        full_buf_mv = memoryview(shm.buf)[padded_start:padded_end]  # type: ignore
        try:
            # — Stage 1
            s1_chains = stage1_find_patterns(full_buf_mv, base_ea + padded_start)

            # ─── TRACEPOINT: Stage 1 ───────────────────────────────────
            TARGET_EA = 0x140005131
            for chain in s1_chains:
                s = chain.overall_start()
                e = s + chain.overall_length()
                if s <= TARGET_EA < e:
                    logger.warning("[TRACE][Stage1] covers 0x%X: %s", TARGET_EA, chain)
                    for seg in chain.segments:
                        abs_s = chain.base_address + seg.start
                        logger.warning(
                            "    seg @0x%X len=%d desc=%s",
                            abs_s,
                            seg.length,
                            seg.description,
                        )
                    break
            # ────────────────────────────────────────────────────────────

            for chain in s1_chains:
                # IMPORTANT: buf starts at padded_start, so bump base_ea accordingly
                ranges = _analyze_chain(
                    chain, full_buf_mv, base_ea + padded_start, is_64
                )
                core_valid.extend(ranges)

        finally:
            del full_buf_mv
    return core_valid


@dataclasses.dataclass
class AsyncDeobfuscator(AsyncEventEmitter):
    shm_name: str
    data_size: int
    start_ea: int
    is_64bit: bool
    max_workers: int = 0
    executor: concurrent.futures.Executor | None = None

    def __post_init__(self):
        super().__post_init__()
        self.pause_evt = asyncio.Event()
        self.stop_evt = asyncio.Event()
        self.max_workers = self.max_workers or max(1, multiprocessing.cpu_count())
        ctx = multiprocessing.get_context("spawn")
        self.executor = self.executor or concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers, mp_context=ctx
        )
        logger.info(f"executor pool created with {self.max_workers} workers")

    @log_execution_time
    async def run(self):
        await self.emit("run_started")

        # 1) define exactly max_workers chunks over the shared buffer
        buf_len = self.data_size
        chunks = list(
            make_chunks(
                buf_len,
                self.max_workers,
                max_pat=MAX_PATTERN_LEN * 2,
            )
        )
        logger.debug(
            "Splitting buffer of %d bytes into %d chunks:", buf_len, len(chunks)
        )
        for idx, (ps, pe, cs, ce) in enumerate(chunks):
            logger.info(
                "  chunk %2d: core=[0x%X..0x%X) padded=[0x%X..0x%X)",
                idx,
                cs + self.start_ea,
                ce + self.start_ea,
                ps + self.start_ea,
                pe + self.start_ea,
            )

        # 2) fire one full-pipeline task per chunk
        loop = asyncio.get_running_loop()
        jobs = [
            (
                self.shm_name,
                padded_start,
                padded_end,
                core_start,
                core_end,
                self.start_ea,
                self.is_64bit,
            )
            for padded_start, padded_end, core_start, core_end in chunks
        ]
        futures = [
            loop.run_in_executor(self.executor, process_chunk, job) for job in jobs
        ]

        # 3) wait, flatten, resolve overlaps globally
        per_chunk = await asyncio.gather(*futures)
        all_ranges = [r for grp in per_chunk for r in grp]
        final: IntervalSet = resolve_overlaps(all_ranges)

        await self.emit("run_finished", final)
        return final

    async def shutdown(self):
        self.stop_evt.set()
        if self.executor:
            self.executor.shutdown(wait=True)
        await self.emit("stopped")


class WorkerController:
    """Wrap AsyncDeobfuscator in its own event loop"""

    def __init__(self, deob: AsyncDeobfuscator):
        self.deob = deob
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._result = None
        self._started = False  # Track if start() has been called

    def _run_loop(self):
        # set and run the loop
        asyncio.set_event_loop(self.loop)
        try:
            self._result = self.loop.run_until_complete(self.deob.run())
        except Exception as e:
            logger.error(f"Exception in worker thread loop: {e}", exc_info=True)
            # Store exception or indicate error?
            self._result = None  # Or some error sentinel

    def start(self):
        """Launch the pipeline in its own thread."""
        if self._started:
            logger.warning("Start called on an already started worker controller.")
            return
        self._thread.start()
        self._started = True  # Mark as started

    def pause(self):
        """Pause after finishing the current iteration."""
        if not self._started:
            logger.warning("Pause called before worker controller was started.")
            return
        logger.info("▶️  Pausing...")
        self.loop.call_soon_threadsafe(self.deob.pause_evt.set)

    def resume(self):
        """Resume if previously paused."""
        if not self._started:
            logger.warning("Resume called before worker controller was started.")
            return
        logger.info("▶️  Resuming...")
        self.loop.call_soon_threadsafe(self.deob.pause_evt.clear)

    def stop(self):
        """Stop the pipeline as soon as possible."""
        if not self._started:
            logger.warning("Stop called before worker controller was started.")
            # Even if not started, set stop event for consistency if needed
            # self.loop.call_soon_threadsafe(self.deob.stop_evt.set) # Maybe not necessary if loop never runs
            return
        logger.info("🛑  Stopping...")
        # Use call_soon_threadsafe as the loop might be running
        self.loop.call_soon_threadsafe(self.deob.stop_evt.set)

    def join(self):
        """Block until the pipeline finishes, return the final chains."""
        if not self._started:
            logger.warning(
                "Join called before worker controller was started. Returning current result (None)."
            )
            return self._result  # Return None or whatever _result is initially

        # Check if the thread is actually alive before joining
        # is_alive() is True from the time start() returns until shortly after run() completes
        if self._thread.is_alive():
            self._thread.join()
        else:
            # Thread was started but might have finished already or crashed
            logger.info(
                "Worker thread was not alive when join was called (already finished or failed?)."
            )
        self._started = False
        return self._result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        PROPOGATE = False
        SUPPRESS = True
        if not exc_type:
            return SUPPRESS

        logger.error(
            "Worker thread raised an exception: %s %s %s", exc_type, exc_val, exc_tb
        )
        self.stop()
        return PROPOGATE
