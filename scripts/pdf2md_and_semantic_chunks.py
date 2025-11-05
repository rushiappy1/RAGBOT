import os
import json
import re
import numpy as np
from pathlib import Path
import pymupdf4llm as p4l
from sentence_transformers import SentenceTransformer

PDF_DIR = Path("/media/rishikesh/Rishi/RAGBOT/rag-banking/data/policies")
OUT_MD_DIR = Path("data/markdown")
OUT_CHUNK_DIR = Path("data/chunks")
OUT_MD_DIR.mkdir(parents=True, exist_ok=True)
OUT_CHUNK_DIR.mkdir(parents=True, exist_ok=True)

PDF_FILES = [
    "Bajaj Allianz Motor.pdf",
    "Reliance General Motor.pdf",
    "HDFC ERGO Motor.pdf",
    "SBI General Motor.pdf",
    "ICICI Lombard Motor.pdf",
    "TATA AIG Motor.pdf",
]

MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)")
MD_TABLE_RE = re.compile(r"^\|.*\|$")
MD_BULLET_RE = re.compile(r"^[ \t]*([-•*])\s+")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SECTION_SPLIT_THRESHOLD = 800
TARGET_SPLIT_SIZE = 450
MIN_CHUNK = 150
OVERLAP = 60
SIM_THRESHOLD = 0.38

def md_blocks(md_text):
    lines = md_text.splitlines()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1; continue
        m = MD_HEADING_RE.match(line)
        if m:
            yield ("heading", line.strip())
            i += 1; continue
        if MD_TABLE_RE.match(line):
            j = i; buf = []
            while j < n and MD_TABLE_RE.match(lines[j]):
                buf.append(lines[j]); j += 1
            yield ("table", "\n".join(buf)); i = j; continue
        j = i; buf = []
        while j < n and not MD_HEADING_RE.match(lines[j]) and not MD_TABLE_RE.match(lines[j]):
            if lines[j].strip():
                buf.append(lines[j].rstrip())
            j += 1
        para = "\n".join(buf)
        if para: yield ("para", para)
        i = j

class BalancedChunker:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    @staticmethod
    def est_tokens(text):
        return max(1, int(len(text) / 4))

    def overlap_tail(self, sents):
        acc, tot = [], 0
        for s in reversed(sents):
            t = self.est_tokens(s)
            if tot + t > OVERLAP: break
            acc.insert(0, s); tot += t
        return acc

    def sentences_from_block(self, kind, text):
        if kind in ("heading", "table"):
            return [text.strip()]
        lines = text.split("\n")
        block = []
        cluster = []
        for l in lines:
            if MD_BULLET_RE.match(l):
                cluster.append(l.strip())
            else:
                if cluster:
                    block.append(" ".join(cluster))
                    cluster = []
                if l.strip():
                    block.append(l.strip())
        if cluster:
            block.append(" ".join(cluster))
        out = []
        for seg in block:
            if len(seg) < 250:
                out.append(seg)
            else:
                sents = re.split(r'(?<=[.!?])\s+', seg)
                out.extend([s.strip() for s in sents if s.strip()])
        return out

    def split_large_section(self, sents, lineage):
        texts = [s for s in sents]
        embeds = self.model.encode(texts, normalize_embeddings=True) if len(texts) > 1 else None
        def anchor(lin): return " > ".join(lin) + " :: " if lin else ""
        def should_break(prev_idx, idx):
            if prev_idx is None or embeds is None: return False
            sim = float(np.dot(embeds[prev_idx], embeds[idx]))
            return sim < SIM_THRESHOLD
        chunks, current, current_tok, prev_idx = [], [], 0, None
        for idx, s in enumerate(sents):
            txt = anchor(lineage) + s
            est = self.est_tokens(txt)
            boundary = should_break(prev_idx, idx)
            need_flush = (current_tok + est) > TARGET_SPLIT_SIZE or (boundary and current_tok >= MIN_CHUNK)
            if need_flush and current:
                chunks.append(("\n".join(current).strip(), current_tok))
                current = self.overlap_tail(current)
                current_tok = sum(self.est_tokens(x) for x in current)
            current.append(txt)
            current_tok += est
            prev_idx = idx
        if current:
            chunks.append(("\n".join(current).strip(), current_tok))
        return chunks

    def chunk(self, md_text, doc_id):
        parent_headings = []
        sections = []
        current_heading, current_lineage, current_sents = None, [], []
        for kind, block in md_blocks(md_text):
            if kind == "heading":
                if current_sents:
                    sections.append((current_heading, current_lineage.copy(), current_sents))
                    current_sents = []
                m = MD_HEADING_RE.match(block)
                level = len(m.group(1))
                title = m.group(2).strip()
                while len(parent_headings) >= level:
                    if parent_headings: parent_headings.pop()
                    else: break
                parent_headings.append(title)
                current_heading = block
                current_lineage = parent_headings.copy()
            else:
                for s in self.sentences_from_block(kind, block):
                    current_sents.append(s)
        if current_sents:
            sections.append((current_heading, current_lineage, current_sents))
        
        def anchor(lin): return " > ".join(lin) + " :: " if lin else ""
        chunks = []
        for heading, lineage, sents in sections:
            section_parts = []
            if heading: section_parts.append(heading)
            section_parts.extend(sents)
            section_text = "\n".join([anchor(lineage) + p for p in section_parts])
            section_tok = self.est_tokens(section_text)
            
            if section_tok <= SECTION_SPLIT_THRESHOLD:
                chunks.append((section_text.strip(), section_tok))
            else:
                chunks.extend(self.split_large_section(section_parts, lineage))
        return chunks

def process_pdf(pdf_path: Path):
    doc_id = pdf_path.stem.replace(" ", "_")
    md_out = OUT_MD_DIR / f"{doc_id}.md"
    jsonl_out = OUT_CHUNK_DIR / f"{doc_id}.jsonl"
    md_text = p4l.to_markdown(str(pdf_path), page_chunks=False, write_images=False)
    md_out.write_text(md_text, encoding="utf-8")
    chunker = BalancedChunker()
    chunks = chunker.chunk(md_text, doc_id=doc_id)
    with open(jsonl_out, "w", encoding="utf-8") as f:
        for i, (chunk, tok) in enumerate(chunks):
            f.write(json.dumps({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}::chunk-{i+1}",
                "text": chunk,
                "approx_tokens": tok
            }, ensure_ascii=False) + "\n")
    print(f"Processed {pdf_path.name}: total_chunks={len(chunks)}")

def main():
    for name in PDF_FILES:
        pdf_path = PDF_DIR / name
        if not pdf_path.exists():
            print(f"WARNING: Missing file {pdf_path}")
            continue
        process_pdf(pdf_path)

if __name__ == "__main__":
    main()
