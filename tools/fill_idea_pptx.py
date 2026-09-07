#!/usr/bin/env python3
"""Fill the SIH2026 idea PPTX template with SIH26168 content.

- Never modifies the template; writes a new file.
- Replaces pointer paragraphs with content bullets, preserving run formatting
  (fonts/sizes/bullets come from the template's own paragraphs).
- Embeds the drift plot on slide 2 and a results table on slide 4.
- Deletes the instructions slide (7). Output: 6 slides.

Usage:
    python3 tools/fill_idea_pptx.py [--template T] [--out O] [--plot P]
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from pptx import Presentation
from pptx.util import Emu, Pt

TITLE = "Intelligent Dead Reckoning for GNSS-Denied Navigation"

S1_DETAILS = [
    "Problem Statement ID \u2013 SIH26168",
    f"Problem Statement Title - {TITLE}",
    "Theme - Misc",
    "PS Category - Software",
    "Team ID - ",
    "Team Name (Registered on portal)",
]

S2_BULLETS = [
    "Phone-only dead reckoning for 2-wheelers during GNSS blackout "
    "(tunnels, underpasses, parking) \u2014 no OBD-II, no extra hardware",
    "AI speed filter (AVNetLite, 460k params) + invariant-EKF fusion "
    "(NHC + ZUPT) + offline HMM map snap; covers all 6 ISRO capabilities",
    "Result: 0.99% drift on 188 m outage (16.3 m \u2192 1.9 m vs naive); "
    "InEKF rescues cruise segments to 2.2%",
    "Innovation: learned uncertainty (\u03c3) drives the filter \u2014 the model "
    "knows when it is unsure; lean-aware lateral model for bikes",
]

S3_BULLETS = [
    "Stack: PyTorch training (Colab T4) \u2192 TFLite FP16 on-device "
    "(Kotlin, ~7 ms) + offline OSM maps (MapLibre)",
    "Data: IO-VNBD \u2014 40 h vehicle + 58 h phone IMU/GPS; 10 Hz \u2192 100 Hz, "
    "gravity-aligned (200, 6) windows",
    "Pipeline: IMU window \u2192 v_fwd + \u03c3 (AVNet) \u2192 21-DOF InEKF "
    "(NHC/ZUPT/GNSS) \u2192 HMM snap \u2192 10 Hz pose",
    "Validation: 60 s masked-GNSS replay \u2014 drift %, ATE/RTE; "
    "ONNX gate <1e-3 before export",
]

S4_BULLETS = [
    "Measured: 16.3 m \u2192 1.9 m drift (188 m outage); map snap halves residual",
    "Risk \u2014 no bike data: synthetic pothole/lean augmentation as stand-in; "
    "holder rides planned",
    "Risk \u2014 wrong-road snap: heading-gated HMM + raw-trace fallback",
    "Risk \u2014 export mismatch: ONNX-vs-PyTorch gate (diff 1e-6)",
]

S4_TABLE = [
    ["Metric (188 m, 60 s)", "Naive", "AI (AVNet+InEKF)", "AI + map"],
    ["Final drift", "16.3 m", "1.9 m", "1.1 m"],
    ["Drift %", "8.7%", "0.99%", "0.59%"],
    ["On-device cost", "\u2014", "~7 ms", "+4 ms snap"],
]

S5_BULLETS = [
    "Resilient navigation where GNSS dies \u2014 tunnels, underpasses, "
    "basements, urban canyons",
    "2-wheelers: no OBD-II exists \u2014 phone-only is the only viable path",
    "Zero hardware cost; fully offline (finale constraint); NavIC-ready fusion",
    "Screening bundle ready: model + scaler + drift plot",
]

S6_BULLETS = [
    "Qian et al., Satellite Navigation 2025 \u2014 AVNet "
    "(DOI 10.1186/s43020-025-00168-7)",
    "IO-VNBD dataset (Onyekpe et al., Data in Brief 35:106885) \u2014 40 h V + 58 h S",
    "QDeepOdo / QAIIMUDeadReckoning (DragonEmperorG) \u2014 reference architecture",
    "RoNIN / TLIO baselines; OSM + MapLibre mapping stack",
]


def shape_by_id(slide, sid):
    """Find a shape by its shape id."""
    for sh in slide.shapes:
        if getattr(sh, "shape_id", None) == sid:
            return sh
    raise KeyError(f"shape id {sid} not found on slide")


def set_para_text(para, text):
    """Replace paragraph text, keeping the first run's formatting."""
    if para.runs:
        para.runs[0].text = text
        for r in para.runs[1:]:
            r._r.getparent().remove(r._r)
    else:
        para.add_run().text = text


def fill_body(shape, items):
    """Replace a body placeholder's paragraphs with (level, text) items.

    Reuses existing paragraphs (keeps bullets/fonts); clones the last one when
    more items exist; removes surplus empty paragraphs. The first paragraph is
    treated as the section header and kept.
    """
    tf = shape.text_frame
    paras = list(tf.paragraphs)
    header, rest = paras[0], paras[1:]
    # drop leading empty paragraphs after the header
    while rest and not rest[0].text.strip():
        p = rest.pop(0)
        p._p.getparent().remove(p._p)
    for i, (level, text) in enumerate(items):
        if i < len(rest):
            p = rest[i]
        else:
            # clone the last body para (or the header if the body has none)
            # to inherit bullets + formatting, then re-read the new last para
            src = rest[-1] if rest else header
            src._p.addnext(copy.deepcopy(src._p))
            p = list(tf.paragraphs)[-1]
            rest.append(p)
        p.level = level
        set_para_text(p, text)
    # remove leftover paragraphs beyond items
    for p in rest[len(items):]:
        p._p.getparent().remove(p._p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="/home/ark/Downloads/SIH2026-IDEA-Presentation-Format.pptx")
    ap.add_argument("--out", default="/home/ark/Downloads/SIH26168_Idea_SIH26168.pptx")
    ap.add_argument("--plot", default="reports/drift_plot.png")
    args = ap.parse_args()

    prs = Presentation(args.template)

    # ---- slide 1: title + PS details ----
    s1 = prs.slides[0]
    title_paras = shape_by_id(s1, 4).text_frame.paragraphs
    title_para = next((p for p in title_paras if p.text.strip()), title_paras[0])
    set_para_text(title_para, TITLE)
    for para, text in zip(shape_by_id(s1, 10).text_frame.paragraphs[1:], S1_DETAILS):
        set_para_text(para, text)

    # ---- slide 2: idea + drift plot (two-column: text left, plot right) ----
    s2 = prs.slides[1]
    body2 = shape_by_id(s2, 15362)
    fill_body(body2, [(0, b) for b in S2_BULLETS])
    body2.width = Emu(6800000)
    s2.shapes.add_picture(args.plot, Emu(7000000), Emu(2064921), width=Emu(4900000))

    # ---- slide 3: technical approach ----
    fill_body(shape_by_id(prs.slides[2], 17410), [(0, b) for b in S3_BULLETS])

    # ---- slide 4: feasibility + results table ----
    s4 = prs.slides[3]
    fill_body(shape_by_id(s4, 17410), [(0, b) for b in S4_BULLETS])
    rows, cols = len(S4_TABLE), len(S4_TABLE[0])
    table = s4.shapes.add_table(rows, cols, Emu(609600), Emu(4000000), Emu(8000000), Emu(1900000)).table
    for ri, row in enumerate(S4_TABLE):
        for ci, val in enumerate(row):
            cell = table.cell(ri, ci)
            cell.text = ""
            run = cell.text_frame.paragraphs[0].add_run()
            run.text = val
            run.font.bold = ri == 0
            run.font.size = Pt(14)

    # ---- slide 5: impact ----
    fill_body(shape_by_id(prs.slides[4], 17410), [(0, b) for b in S5_BULLETS])

    # ---- slide 6: references ----
    fill_body(shape_by_id(prs.slides[5], 17410), [(0, b) for b in S6_BULLETS])

    # ---- delete instructions slide (7): drop the sldId reference AND the
    # orphan parts, so strict validators never see a dangling slide ----
    prs.slides._sldIdLst.remove(prs.slides._sldIdLst[6])

    out = Path(args.out)
    prs.save(str(out))

    import zipfile

    with zipfile.ZipFile(out, "r") as zin:
        names = zin.namelist()
        doomed = [n for n in names if "slide7" in n]
        if doomed:
            blobs = {n: zin.read(n) for n in names if n not in doomed}
    if doomed:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
            for n, blob in blobs.items():
                zout.writestr(n, blob)
        print(f"purged orphan parts: {doomed}")
    print(f"saved {out} ({out.stat().st_size/1e6:.2f} MB), slides={len(prs.slides)}")


if __name__ == "__main__":
    main()
