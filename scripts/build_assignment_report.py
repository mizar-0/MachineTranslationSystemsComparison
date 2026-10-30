"""Create the assignment report DOCX from tracked experiment results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
REPORT_DIR = ROOT / "report"
OUTPUT = REPORT_DIR / "FGAI_Assignment3_Report.docx"

BLUE = "1F4E78"
PALE_BLUE = "D9EAF7"
PALE_GRAY = "F2F2F2"
BORDER = "D9D9D9"
WHITE = "FFFFFF"
BLACK = RGBColor(0, 0, 0)


def set_font(run, name="Aptos", size=11, bold=False, color=BLACK):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)
    rfonts.set(qn("w:eastAsia"), "Nirmala UI")
    rfonts.set(qn("w:cs"), "Nirmala UI")
    return run


def set_cell_fill(cell, color):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), color)


def set_cell_margins(cell, top=100, start=120, bottom=100, end=120):
    tc_pr = cell._tc.get_or_add_tcPr()
    margins = tc_pr.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        tc_pr.append(margins)
    for edge, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        element = margins.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            margins.append(element)
        element.set(qn("w:w"), str(value))
        element.set(qn("w:type"), "dxa")


def set_table_borders(table, color=BORDER, size=6):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = borders.find(qn(f"w:{edge}"))
        if tag is None:
            tag = OxmlElement(f"w:{edge}")
            borders.append(tag)
        tag.set(qn("w:val"), "single")
        tag.set(qn("w:sz"), str(size))
        tag.set(qn("w:space"), "0")
        tag.set(qn("w:color"), color)


def repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def format_table(table, widths=None, header_fill=BLUE):
    table.autofit = False
    table.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_table_borders(table)
    repeat_header(table.rows[0])
    for r_idx, row in enumerate(table.rows):
        if r_idx == 0:
            fill = header_fill
        else:
            fill = PALE_BLUE if r_idx % 2 == 0 else WHITE
        for c_idx, cell in enumerate(row.cells):
            if widths:
                cell.width = Inches(widths[c_idx])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_fill(cell, fill)
            set_cell_margins(cell)
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1.05
                paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    set_font(run, size=9.2, bold=(r_idx == 0), color=RGBColor(255, 255, 255) if r_idx == 0 else BLACK)


def add_table(doc, headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for i, text in enumerate(headers):
        table.rows[0].cells[i].text = str(text)
    for row_values in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row_values):
            cells[i].text = str(value)
    format_table(table, widths)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_hyperlink(paragraph, text, url):
    part = paragraph.part
    rel_id = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), rel_id)
    run = OxmlElement("w:r")
    r_pr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), BLUE)
    r_pr.append(color)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    r_pr.append(underline)
    run.append(r_pr)
    text_node = OxmlElement("w:t")
    text_node.text = text
    run.append(text_node)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def add_page_number(paragraph):
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instruction, separate, end])


def add_caption(doc, text):
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(8)
    set_font(paragraph.add_run(text), size=9, bold=False, color=RGBColor(64, 64, 64))


def add_figure(doc, filename, caption, width=6.25):
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(FIGURES / filename), width=Inches(width))
    add_caption(doc, caption)


def add_body(doc, text, bold_lead=None):
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(7)
    paragraph.paragraph_format.line_spacing = 1.12
    if bold_lead:
        set_font(paragraph.add_run(bold_lead), bold=True)
    set_font(paragraph.add_run(text))
    return paragraph


def add_bullets(doc, items):
    for item in items:
        paragraph = doc.add_paragraph(style="List Bullet")
        paragraph.paragraph_format.space_after = Pt(4)
        paragraph.paragraph_format.left_indent = Inches(0.25)
        for run in paragraph.runs:
            set_font(run)
        if not paragraph.runs:
            set_font(paragraph.add_run(item))
        else:
            paragraph.runs[0].text = item


def read_results():
    with (RESULTS / "model_comparison.csv").open(encoding="utf-8", newline="") as handle:
        comparison = list(csv.DictReader(handle))
    qualitative = json.loads((RESULTS / "qualitative_samples.json").read_text(encoding="utf-8"))
    return comparison, qualitative


def build_report():
    comparison, qualitative = read_results()
    by_model = {row["model"]: row for row in comparison}

    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(0.72)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(11)
    normal.font.color.rgb = BLACK
    normal.paragraph_format.space_after = Pt(7)
    normal.paragraph_format.line_spacing = 1.12
    for style_name, size, before, after in (
        ("Title", 24, 0, 12),
        ("Heading 1", 16, 14, 7),
        ("Heading 2", 12.5, 10, 5),
        ("Heading 3", 11.5, 8, 4),
    ):
        style = styles[style_name]
        style.font.name = "Aptos Display" if style_name != "Normal" else "Aptos"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = BLACK
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
    title_ppr = styles["Title"].element.get_or_add_pPr()
    title_border = title_ppr.find(qn("w:pBdr"))
    if title_border is not None:
        title_ppr.remove(title_border)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(footer.add_run("English to Hindi Machine Translation Systems Comparison   |   "), size=8.5, color=RGBColor(89, 89, 89))
    add_page_number(footer)

    # Cover page
    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(78)
    set_font(title.add_run("English to Hindi Machine Translation Systems Comparison"), name="Aptos Display", size=24, bold=True)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(42)
    set_font(subtitle.add_run("FGAI Assignment 3 Experimental Report"), size=14, color=RGBColor(64, 64, 64))
    metadata = [
        ("Student name", "________________________________________"),
        ("Registration number", "________________________________________"),
        ("Developed system", "Comparative English to Hindi neural machine translation pipeline"),
        ("Dataset", "IIT Bombay English-Hindi Parallel Corpus"),
        ("Repository", "https://github.com/mizar-0/MachineTranslationSystemsComparison"),
        ("Report date", "31 October 2026"),
    ]
    cover_table = doc.add_table(rows=len(metadata), cols=2)
    cover_table.style = "Table Grid"
    for i, (label, value) in enumerate(metadata):
        cover_table.rows[i].cells[0].text = label
        cover_table.rows[i].cells[1].text = value
    format_table(cover_table, [1.75, 4.55], header_fill=BLUE)
    # Cover table has no header; restyle all rows in a quiet two-column format.
    for i, row in enumerate(cover_table.rows):
        for j, cell in enumerate(row.cells):
            set_cell_fill(cell, PALE_BLUE if i % 2 == 0 else WHITE)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for run in p.runs:
                    set_font(run, size=10.3, bold=(j == 0), color=BLACK)
    doc.add_page_break()

    doc.add_heading("Executive Summary", level=1)
    add_body(doc, "This report evaluates four English-to-Hindi machine translation systems on identical fixed data splits: a Transformer trained from scratch, full fine-tuning of mT5-small, and LoRA adaptation of LLaMA 3.2 1B and Gemma 3 270M. Every system was evaluated on the same 2,507-sentence test set with deterministic greedy decoding.")
    add_body(doc, "LLaMA 3.2 1B LoRA produced the strongest result, with corpus BLEU 0.845 and chrF 9.680, while updating only 5.64 million parameters. The scratch Transformer was fastest to train and decode. Absolute quality remained low across all systems, and sentence-level inspection exposed repetition, omissions, empty outputs, and fluent hallucinations. The models are suitable for experimental comparison but not for high-stakes deployment.")
    add_table(
        doc,
        ["System", "BLEU", "chrF", "Train hours", "Sec per sentence"],
        [[row["model"], f"{float(row['test_bleu']):.3f}", f"{float(row['test_chrf']):.3f}", f"{float(row['training_hours']):.2f}", f"{float(row['seconds_per_test_sentence']):.3f}"] for row in comparison],
        [2.15, 0.75, 0.75, 1.05, 1.25],
    )
    doc.add_heading("Report Structure", level=2)
    add_bullets(doc, [
        "Sections 1 to 4 define the task, dataset, systems, and experimental protocol.",
        "Sections 5 and 6 compare quantitative and qualitative outcomes.",
        "Sections 7 to 10 address responsible AI, limitations, reproducibility, and conclusions.",
        "The appendix records run identifiers and core hyperparameters for auditability.",
    ])

    doc.add_heading("1 Problem and Objective", level=1)
    add_body(doc, "Machine translation quality depends not only on model size but also on architecture, pretraining, tokenization, and the way a model is adapted. This project controls the dataset and test protocol so those design choices can be compared on English-to-Hindi translation.")
    add_body(doc, "The objective is to train or adapt four required model families, measure BLEU, chrF, time, memory, and parameter counts, retain sentence-level predictions, and interpret the results with explicit attention to failure modes and responsible use.")

    doc.add_heading("2 Dataset and Split Protocol", level=1)
    add_body(doc, "The public IIT Bombay English-Hindi Parallel Corpus provides aligned English and Hindi sentences. A reproducible reservoir sample of 30,000 training pairs was selected with seed 1729. The complete official development and test splits were retained. No normalization, deduplication, length filtering, or language identification was applied, which preserves the source distribution but also preserves noise and domain imbalance.")
    add_table(doc, ["Split", "Pairs", "Selection"], [["Train", "30,000", "Seeded reservoir sample"], ["Validation", "520", "Complete official development split"], ["Test", "2,507", "Complete official test split"]], [1.15, 1.0, 4.0])
    paragraph = doc.add_paragraph()
    set_font(paragraph.add_run("Dataset source: "), bold=True)
    add_hyperlink(paragraph, "IIT Bombay English-Hindi Parallel Corpus", "https://www.cfilt.iitb.ac.in/iitb_parallel/")
    add_body(doc, "The generated manifest records split sizes, source files, and checksums. All four runs consumed the same saved data files, preventing accidental test-set drift.")

    doc.add_heading("3 Systems and Adaptation", level=1)
    add_table(
        doc,
        ["System", "Architecture", "Tokenizer", "Adaptation"],
        [
            ["Scratch Transformer", "4 encoder and 4 decoder layers; d=256; 4 heads", "8,000-token SentencePiece BPE trained on selected train split", "Full training"],
            ["mT5-small", "Multilingual encoder-decoder", "Pretrained mT5 SentencePiece vocabulary", "Full fine-tuning"],
            ["LLaMA 3.2 1B", "Pretrained decoder-only causal LM", "Native pretrained tokenizer", "LoRA rank 8 on all linear layers"],
            ["Gemma 3 270M", "Instruction-tuned decoder-only causal LM", "Native pretrained tokenizer", "LoRA rank 8 on all linear layers"],
        ],
        [1.35, 1.75, 2.0, 1.55],
    )
    doc.add_heading("3.1 Scratch Transformer", level=2)
    add_body(doc, "The scratch baseline uses a standard Transformer encoder-decoder with learned token embeddings and a jointly trained SentencePiece tokenizer. Its 11.51 million parameters are all trainable. Because it receives no multilingual pretraining, the 30,000-pair subset must teach both language representation and translation behavior.")
    doc.add_heading("3.2 mT5 Small", level=2)
    add_body(doc, "mT5-small is a multilingual text-to-text encoder-decoder. All 300.18 million parameters were fine-tuned with BF16, gradient checkpointing, AdamW, and an effective batch size of 16. Its pretrained vocabulary provides Hindi coverage without a new tokenizer.")
    doc.add_heading("3.3 LLaMA and Gemma LoRA", level=2)
    add_body(doc, "The causal language models were adapted with low-rank matrices applied to all linear layers. Only 0.454% of LLaMA parameters and 0.703% of Gemma parameters were trainable. Both used the prompt 'Translate the following English sentence into Hindi' and masked prompt and padding tokens in the loss. The local snapshots came from Unsloth mirrors of the original model families; this avoids gated repository access while preserving the named base architectures.")
    add_figure(doc, "parameter_comparison.png", "Figure 1. Total and trainable parameter counts on a logarithmic scale.")

    doc.add_heading("4 Experimental Setup", level=1)
    add_body(doc, "Training ran on one NVIDIA GeForce RTX 5060 Laptop GPU with 8 GB VRAM using Python 3.14, PyTorch 2.11.0, CUDA 12.8, Transformers, PEFT, SentencePiece, and SacreBLEU. Seed 1729 was used throughout. Run folders preserve configuration, step logs, epoch logs, status, checkpoints, summaries, and all test predictions.")
    add_table(
        doc,
        ["System", "Epochs", "Batch / accumulation", "Max lengths", "Learning rate", "Precision"],
        [
            ["Scratch Transformer", "16 of 30; early stop", "32 / 1", "128 shared", "3e-4 with reductions", "FP32"],
            ["mT5-small", "5", "4 / 4", "128 / 128", "5e-5 linear", "BF16"],
            ["LLaMA LoRA", "2", "1 / 16", "96 / 96", "2e-4 linear", "BF16"],
            ["Gemma LoRA", "2", "1 / 16", "96 / 96", "2e-4 linear", "BF16"],
        ],
        [1.45, 1.15, 1.2, 1.05, 1.15, 0.8],
    )
    add_body(doc, "Evaluation used greedy decoding with beam size 1 and no temperature sampling. Corpus BLEU measures exact n-gram precision with a brevity penalty; chrF measures character n-gram similarity and is more tolerant of morphology and spelling variation. Both are reported because neither alone captures semantic adequacy.")
    add_figure(doc, "validation_loss.png", "Figure 2. Validation loss by epoch. Loss values are not directly comparable across different tokenizers and objectives.")

    doc.add_heading("5 Quantitative Results", level=1)
    add_table(
        doc,
        ["Model", "BLEU", "chrF", "Train h", "Infer s", "s/sent", "Peak GiB"],
        [[row["model"], f"{float(row['test_bleu']):.3f}", f"{float(row['test_chrf']):.3f}", f"{float(row['training_hours']):.2f}", f"{float(row['inference_seconds']):.1f}", f"{float(row['seconds_per_test_sentence']):.3f}", f"{float(row['peak_gpu_memory_gib']):.2f}"] for row in comparison],
        [1.75, 0.65, 0.65, 0.75, 0.85, 0.8, 0.8],
    )
    add_figure(doc, "quality_comparison.png", "Figure 3. Corpus BLEU and chrF on the common 2,507-sentence test set.")
    add_body(doc, "LLaMA ranks first on both metrics, followed by Gemma on BLEU and mT5 on chrF. The gap between BLEU and chrF is important: partial character overlap can remain even when full phrase ordering and adequacy are poor. The low absolute BLEU values align with the visible repetition and hallucination in sentence-level outputs.")
    add_figure(doc, "runtime_comparison.png", "Figure 4. Measured training time and greedy decoding latency.")
    add_body(doc, "The scratch Transformer is the fastest system, requiring 0.57 hours to train and 0.236 seconds per test sentence. LLaMA is the strongest but slowest at inference. Gemma trained for 7.71 hours, showing that a smaller total parameter count does not guarantee faster fine-tuning when sequence lengths, architecture, checkpointing, and kernels differ.")

    doc.add_heading("6 Qualitative Analysis", level=1)
    add_body(doc, "Five fixed examples are saved in the repository. The three examples below illustrate a successful short translation, a partially adequate domain sentence, and a sensitive-content failure. Outputs are shown exactly as generated.")
    sample_map = {item["id"]: item for item in qualitative}
    for sample_id, note in (
        (0, "LLaMA and Gemma preserve the core meaning of a short question, while the scratch and mT5 systems collapse into repetition."),
        (147, "LLaMA captures the finance and government-treasury content but introduces awkward wording; the other systems are largely unrelated."),
        (179, "On violence-related news, LLaMA hallucinates an interface-like question, Gemma returns an empty string, and the other models repeat unrelated text. This is an unacceptable high-stakes failure."),
    ):
        sample = sample_map[sample_id]
        doc.add_heading(f"6.{1 if sample_id == 0 else 2 if sample_id == 147 else 3} Test Item {sample_id}", level=2)
        add_body(doc, sample["source"], bold_lead="English: ")
        add_body(doc, sample["reference"], bold_lead="Reference: ")
        rows = [[name, prediction or "[empty output]"] for name, prediction in sample["predictions"].items()]
        table = add_table(doc, ["System", "Generated Hindi"], rows, [1.55, 4.65])
        for row in table.rows[1:]:
            row.cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
        add_body(doc, note, bold_lead="Interpretation: ")
    add_body(doc, "Across the selected examples, recurring failure modes are degeneration into a repeated phrase, empty output, omission of named entities and numbers, mistranslation of domain terminology, and fluent but unsupported content. LLaMA is comparatively better on some short and medium examples, but its aggregate lead does not remove these risks.")

    doc.add_heading("7 Responsible and Ethical AI Analysis", level=1)
    doc.add_heading("7.1 Bias and Cultural Representation", level=2)
    add_body(doc, "The corpus inherits the topics, register, geographic focus, gender representation, political language, and religious references of its source material. Models can amplify those imbalances or attach culturally loaded language to neutral inputs. Formal Hindi references may not represent regional varieties or mixed Hindi-English usage.")
    doc.add_heading("7.2 Hallucination and Harm", level=2)
    add_body(doc, "The test outputs demonstrate hallucinated interface questions, unrelated religious-style phrases, repetition, and empty strings. These errors can distort events, identities, quantities, and intent. Harm is highest in medical, legal, financial, government, public-safety, and violence-related translation, where a fluent sentence may be trusted even when it is unsupported.")
    doc.add_heading("7.3 Mitigations", level=2)
    add_bullets(doc, [
        "Require fluent human review for consequential translations and retain the source sentence beside every output.",
        "Detect empty, highly repetitive, excessively long, or low-confidence generations and abstain instead of returning them.",
        "Evaluate named entities, numbers, negation, gender, caste, religion, region, and domain terminology with targeted test suites.",
        "Conduct stereotype, toxicity, and cultural-appropriateness review with diverse Hindi speakers.",
        "Record model, data, decoding, and reviewer metadata so problematic translations remain traceable.",
        "Do not deploy these four checkpoints for high-stakes translation on the evidence available here.",
    ])

    doc.add_heading("8 Temperature and Decoding Scope", level=1)
    add_body(doc, "Temperature experiments were intentionally deferred. Every reported prediction uses deterministic greedy decoding, so temperature is not active in the comparison. This keeps the four systems comparable and avoids presenting unrun sampling results.")
    add_body(doc, "A future study should hold the test subset and generation limit fixed, test several nonzero temperatures with multiple random seeds, and report mean and variance for BLEU, chrF, repetition, empty-output rate, hallucination rate, and human adequacy. Greedy decoding should remain the temperature-free baseline.")

    doc.add_heading("9 Limitations", level=1)
    add_bullets(doc, [
        "The 30,000-pair training subset is small relative to modern multilingual pretraining and may not cover the test domains adequately.",
        "Only one data seed and one training seed were used, so run-to-run uncertainty is not measured.",
        "Different tokenizers and objectives make cross-model loss values descriptive rather than directly comparable.",
        "BLEU and chrF do not directly assess factual consistency, toxicity, cultural appropriateness, or human preference.",
        "Greedy decoding controls the comparison but does not establish the best decoding policy for each architecture.",
        "The causal models used mirrored model repositories; the identifiers and local paths are disclosed for reproducibility.",
    ])

    doc.add_heading("10 Conclusion", level=1)
    add_body(doc, "LLaMA 3.2 1B LoRA is the best system in this controlled experiment. It leads BLEU and chrF while training only 0.454% of the base parameters. The scratch Transformer is the most computationally efficient, and mT5 offers a strong multilingual encoder-decoder baseline, but both exhibit severe degeneration in the selected outputs. Gemma is parameter-efficient yet produces the lowest chrF and frequent empty or unrelated outputs.")
    add_body(doc, "The practical conclusion is cautious: pretrained capacity and parameter-efficient adaptation improve limited-data translation, but the measured checkpoints remain unreliable. Future work should use more and cleaner parallel data, targeted domain evaluation, sequence-level controls, repeated seeds, and bilingual human assessment before deployment.")

    doc.add_heading("11 Reproducibility and Repository", level=1)
    add_body(doc, "The repository contains dataset preparation, four training implementations, durable queue and checkpoint handling, tracked comparison tables and figures, selected predictions, and this report. Raw checkpoints and full run outputs are kept locally because of their size; aggregate results and examples are tracked in Git.")
    paragraph = doc.add_paragraph()
    set_font(paragraph.add_run("GitHub repository: "), bold=True)
    add_hyperlink(paragraph, "mizar-0 MachineTranslationSystemsComparison", "https://github.com/mizar-0/MachineTranslationSystemsComparison")
    add_body(doc, "Core run identifiers are transformer_full_20261029, mt5_small_20261029_r3, llama_lora_20261030_r2, and gemma_lora_20261030. Each contains a configuration file, epoch history, summary, and exactly 2,507 test predictions.")

    doc.add_heading("References", level=1)
    references = [
        "Vaswani, A. et al. 2017. Attention Is All You Need. https://arxiv.org/abs/1706.03762",
        "Kunchukuttan, A., Mehta, P., and Bhattacharyya, P. 2018. The IIT Bombay English-Hindi Parallel Corpus. https://aclanthology.org/L18-1548/",
        "Xue, L. et al. 2021. mT5 A Massively Multilingual Pre-trained Text-to-Text Transformer. https://arxiv.org/abs/2010.11934",
        "Meta. Llama 3.2 1B Model Card. https://huggingface.co/meta-llama/Llama-3.2-1B",
        "Google. Gemma 3 Model Overview. https://ai.google.dev/gemma/docs/core/model_card_3",
        "Post, M. 2018. A Call for Clarity in Reporting BLEU Scores. https://aclanthology.org/W18-6319/",
    ]
    for i, reference in enumerate(references, start=1):
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.left_indent = Inches(0.25)
        paragraph.paragraph_format.first_line_indent = Inches(-0.25)
        set_font(paragraph.add_run(f"{i}. {reference}"), size=10)

    doc.add_page_break()
    doc.add_heading("Appendix A Run Configuration Summary", level=1)
    add_table(
        doc,
        ["Run", "Best epoch", "Parameters", "Trainable share", "Stopping"],
        [[row["run_id"], row["best_epoch"], f"{int(row['total_parameters']):,}", f"{float(row['trainable_percent']):.3f}%", "Early stopping" if row["model"] == "Scratch Transformer" else "Epoch cap"] for row in comparison],
        [2.1, 0.8, 1.35, 1.0, 1.1],
    )
    add_body(doc, "The scratch run selected epoch 11 by validation loss and stopped after epoch 16 following five stale epochs. mT5 selected epoch 5. Both causal LoRA systems selected epoch 2. Checkpoints were saved before final generation so evaluation failures would not discard completed training.")
    doc.add_heading("Appendix B Logged Artifacts", level=2)
    add_bullets(doc, [
        "run_config.json: data counts, software and GPU environment, hyperparameters, parameter counts, truncation counts, and decoding configuration.",
        "steps.jsonl and epochs.jsonl: step losses, learning rate, gradient norm, timing, validation metrics, and peak allocated GPU memory.",
        "summary.json: selected epoch, stopping reason, total timing, test BLEU, test chrF, and test-set size.",
        "test_predictions.jsonl: test identifier, English source, Hindi reference, and generated prediction for every sentence.",
        "Best and last checkpoints or adapters: recoverable model state at training boundaries.",
    ])

    REPORT_DIR.mkdir(exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build_report()
