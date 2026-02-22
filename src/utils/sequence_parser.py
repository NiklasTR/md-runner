"""Parse sequence lines from example_cycles.txt and similar files.

Linear (simple format): plain one-letter sequence e.g. AA, ARIP, GYDPETGTWG.
Cyclic (bracket format): [#HeadTailLactam][P][dL][P][dV][L][dA][F][dP] - header
followed by bracketed residues. [X] = L-amino acid, [dX] = D-amino acid,
[mX] = N-methyl-L-X, [Sar] = sarcosine (N-methylglycine).
"""

import re

_CYCLIC_HEADER = "[#HeadTailLactam]"

D_AMINO_ACID_CCD: dict[str, str] = {
    "A": "DAL",
    "R": "DAR",
    "N": "DSG",
    "D": "DAS",
    "C": "DCY",
    "E": "DGL",
    "Q": "DGN",
    "H": "DHI",
    "I": "DIL",
    "L": "DLE",
    "K": "DLY",
    "M": "MED",
    "F": "DPN",
    "P": "DPR",
    "S": "DSN",
    "T": "DTH",
    "W": "DTR",
    "Y": "DTY",
    "V": "DVA",
}

N_METHYL_CCD: dict[str, str] = {
    "G": "SAR",
    "A": "MAA",
    "V": "MVA",
    "L": "MLE",
    "I": "MIL",
    "F": "MEF",
}


def line_to_name(line: str) -> str:
    """Derive a filename-safe name from a sequence line.

    Cyclic bracket format gets a 'cyclo-' prefix with the mixed-case one-letter sequence.
    Linear plain format is returned as-is.
    """
    parsed = parse_sequence_line(line)
    if parsed is None:
        return line.strip()
    sequence, is_cyclic, _ = parsed
    return f"cyclo-{sequence}" if is_cyclic else sequence


def d_positions_and_ccds(sequence: str) -> list[tuple[int, str]]:
    """Return (position, ccd) for each D-amino acid. Positions are 1-indexed."""
    return [
        (i + 1, D_AMINO_ACID_CCD[c.upper()])
        for i, c in enumerate(sequence)
        if c.islower() and c.upper() in D_AMINO_ACID_CCD
    ]


def parse_sequence_line(line: str) -> tuple[str, bool, list[tuple[int, str]]] | None:
    """Parse a line from a sequence file.

    Returns:
        (sequence, is_cyclic, modifications) for parseable lines, None for comments.
        Sequence: uppercase for L, lowercase for D. Modifications: (position, ccd)
        for D-amino acids and N-methyl variants. Positions are 1-indexed.
    """
    line = line.strip()
    if not line:
        return None

    if line.startswith(_CYCLIC_HEADER + "["):
        parts = re.findall(r"\[([^]]+)\]", line)
        parts = [p for p in parts if p != "#HeadTailLactam"]
        seq_chars = []
        modifications: list[tuple[int, str]] = []
        for i, p in enumerate(parts):
            pos = i + 1
            if p == "Sar" or (p.startswith("m") and len(p) == 2 and p[1].upper() == "G"):
                seq_chars.append("G")
                modifications.append((pos, "SAR"))
            elif p.startswith("m") and len(p) == 2 and p[1].upper() in N_METHYL_CCD:
                base = p[1].upper()
                seq_chars.append(base)
                modifications.append((pos, N_METHYL_CCD[base]))
            elif p.startswith("d") and len(p) == 2 and p[1].upper() in D_AMINO_ACID_CCD:
                seq_chars.append(p[1].lower())
                modifications.append((pos, D_AMINO_ACID_CCD[p[1].upper()]))
            elif len(p) == 1:
                seq_chars.append(p.upper())
            else:
                seq_chars.append(p[0].upper())
        return ("".join(seq_chars), True, modifications)

    if re.match(r"^[A-Za-z]+$", line):
        return (line, False, [])
    return None
