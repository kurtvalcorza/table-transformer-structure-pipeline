# DIMER Notebook: Table Intelligence

## Table Detection → Structure Recognition → Table Reconstruction → TAPAS QA

**Proposed filename:** `DIMER_Table_Intelligence_Workshop.ipynb`  
**DIMER Notebook Specification:** `2.1`  
**Profile:** `TASK-INFERENCE`  
**Pedagogical mode:** `WORKSHOP`  
**Workflow scope:** `COMPOSED-PIPELINE`  
**Standalone:** `true`  
**Recommended runtime:** Kaggle / Colab Tesla T4  
**Canonical dataset:** SciTSR-PD public-domain scientific tables  
**Canonical workflow:** Frozen inference only  
**Models:**
1. Table Transformer Detection
2. Table Transformer Structure Recognition v1.1-all
3. TAPAS Large WTQ

**DePlot:** explicitly outside the canonical workflow

## Guided-notebook layer

**Who it is for.** Learners who can run Python cells in Colab/Jupyter and are new to composed document-intelligence pipelines.

**How to use this notebook.**
1. Use a Tesla T4-class GPU or equivalent.
2. Choose **Run all** for the canonical path; the defaults are the reference settings.
3. Read the conceptual and interpretation cells while the notebook runs.
4. Cells marked **Infrastructure** handle setup, immutable model/data acquisition, provenance, or orchestration. Learners may run those cells without studying their implementation.

**Task at a glance.**

`document page → table detection → table crop → structure recognition → cell grid reconstruction → text assignment → structured table → TAPAS QA → answer`

**Roadmap.**
1. Distinguish the three table-intelligence tasks.
2. Inspect the controlled SciTSR-PD inputs.
3. Establish stage-level reference metrics.
4. Run detection, structure recognition, reconstruction, and QA.
5. Compare gold-crop, structure-only, and full end-to-end paths.
6. Trace one downstream error back to the stage that introduced it.
7. Change one bounded parameter, rerun the affected stage, and explain the tradeoff.
8. Write an evidence-based conclusion and limitations.

**Required learner scaffolding in the generated notebook.**
- Before each principal stage, state the question being tested and ask for a prediction where meaningful.
- After each principal stage, include **Expected result** or **What to notice** guidance.
- Include interpretation checkpoints with collapsible sample answers.
- Include at least one **Predict → Change one thing → Run → Observe → Explain** activity, preferably a threshold or crop-padding change on the validation path only.
- Label long setup/model-staging/helper cells **Infrastructure**.
- End with troubleshooting, glossary, and an evidence-based conclusion template.
- Use **notebook** for the `.ipynb` artifact. Reserve **workshop** for the actual event/training activity.

---

# 1. Purpose

This notebook demonstrates a complete table-intelligence pipeline:

```text
document page
    ↓
table detection
    ↓
table crop
    ↓
structure recognition
    ↓
rows + columns
    ↓
cell grid reconstruction
    ↓
cell text assignment
    ↓
structured table
    ↓
TAPAS
    ↓
question answer
```

The notebook focuses on the interfaces between these stages.

The objective is not simply to run three models independently. It is to show how errors propagate through a composed document-understanding system.

---

# 2. Learning objectives

By the end of the notebook activity, learners should be able to:

1. explain the difference between table detection, structure recognition and table QA;
2. locate a table within a page;
3. crop the detected table with the correct coordinate transformation;
4. recognize rows and columns from a table crop;
5. reconstruct a rectangular table grid from predicted structure boxes;
6. understand why geometry alone does not recover cell text;
7. assign externally supplied text to predicted cells;
8. convert a visual table into the string-table representation TAPAS expects;
9. ask lookup and aggregation questions over the reconstructed table;
10. distinguish geometry errors from QA-model errors;
11. measure stage-by-stage performance rather than only final answer accuracy;
12. understand how early-stage errors propagate downstream;
13. identify where OCR belongs in a production pipeline; and
14. apply the workflow to a page plus user-supplied OCR/text boxes.

---

# 3. Why three separate capabilities are required

## Table detection

Answers:

> Where is the table on this page?

Output:

```text
table bounding box
```

---

## Structure recognition

Answers:

> Where are the rows, columns and spanning cells inside this table?

Output:

```text
row boxes
column boxes
header / spanning-cell boxes
```

---

## Table QA

Answers:

> What does this structured table say?

Input:

```text
string table
+
question
```

Output:

```text
selected cells
+
aggregation
+
answer
```

None of these three models replaces the others.

---

# 4. Model 1 — Table Transformer Detection

**Model**

`microsoft/table-transformer-detection`

**Immutable revision**

`2357cbe2b5a5d1c03e54f32764f06058933b65ab`

**License**

MIT

**Architecture**

DETR with ResNet-18 backbone.

**Classes**

```text
table
table rotated
```

**Maximum queries**

```text
15
```

**Default detection threshold**

```text
0.90
```

**Model weight**

```text
model.safetensors
115,317,516 bytes
SHA-256:
8f1aa73170102c038d40155e2734b343bf07e0fe12594228a8590943b01dccf7
```

Total snapshot:

```text
115,320,191 bytes
```

Input preprocessing resizes the shortest page edge to approximately 800 px.

---

# 5. Model 2 — Table Transformer Structure Recognition

**Model**

`microsoft/table-transformer-structure-recognition-v1.1-all`

**Immutable revision**

`7587a7ef111d9dcbf8ac695f1376ab7014340a0c`

**License**

MIT

**Architecture**

DETR / ResNet family specialized for table structure.

**Classes**

```text
table
table column
table row
table column header
table projected row header
table spanning cell
```

**Maximum queries**

```text
125
```

**Default threshold**

```text
0.50
```

**Expected crop convention**

Approximately:

```text
10 px surrounding page context
```

**Model weight**

```text
model.safetensors
115,437,156 bytes
SHA-256:
9df416575a3a36ebd0129342d4f597f14d6e5170268f3d52d28584ab4466a501
```

Total snapshot:

```text
115,515,347 bytes
```

The processor scales the table crop to an 800-px maximum edge.

---

# 6. Model 3 — TAPAS Large WTQ

**Model**

`google/tapas-large-finetuned-wtq`

**Immutable revision**

`f58317ab2577d17647d9acafa790c744a0388b30`

**License**

Apache-2.0

**Parameters**

```text
336,734,214
```

**Model weight**

```text
model.safetensors
1,346,985,282 bytes
SHA-256:
149247e13732c222ba621e0c4e7b90ba260869b36adcbe115dc872c2c98bccf0
```

Total snapshot:

```text
1,347,256,837 bytes
```

---

# 7. TAPAS input contract

TAPAS receives:

```text
structured string table
+
question
```

Limits:

```text
MAX_ROWS       = 64
MAX_COLUMNS    = 32
MAX_TOKENS     = 512
MAX_QUERY_CHARS = 500
MAX_CELL_CHARS = 200
```

Supported aggregation operators:

```text
NONE
SUM
AVERAGE
COUNT
```

Cell-selection threshold:

```text
0.50
```

Decision rule:

```text
cell selected when mean token sigmoid score > 0.50

aggregation =
argmax(NONE, SUM, AVERAGE, COUNT)
```

For `SUM`, `AVERAGE` and `COUNT`, the numeric answer is calculated by the DIMER reference logic from the selected cell strings.

TAPAS itself does not directly generate the final number.

---

# 8. Why DePlot is not included

DePlot solves:

```text
chart image
→ generated table
```

That is an adjacent **chart understanding** task.

This notebook solves:

```text
document page
→ actual table
→ table structure
→ structured table
→ QA
```

Including DePlot would mix chart-to-table translation with document-table structure recognition.

Therefore:

```text
DePlot = out of canonical scope
```

A future Chart Intelligence notebook may cover it separately.

---

# 9. Canonical dataset — SciTSR-PD

Reuse the existing DIMER Table Transformer Structure sample.

**Dataset**

```text
SciTSR-PD
```

**Source**

```text
bevaya/SciTSR-pd
```

**License**

```text
CC0 / public-domain source papers
```

The current DIMER carrier pins the dataset at the immutable SciTSR-PD revision beginning:

```text
dae336ef...
```

The workshop build MUST copy the full immutable revision from the current carrier provenance rather than relying on the shortened identifier.

---

# 10. Existing DIMER corpus

The current Table Transformer Structure carrier contains:

```text
94 tables
46 source papers
```

rendered at:

```text
150 DPI
```

The images are already digest-pinned.

Structure annotations include:

- table;
- rows;
- columns;
- spanning cells.

SciTSR does not provide the two PubTables-style header labels used by the structure model:

```text
table column header
table projected row header
```

Those detections MUST NOT be penalized as false positives merely because the tutorial corpus lacks those annotations.

---

# 11. Existing split

Reuse the exact DIMER paper-disjoint split.

Seed:

```text
42
```

Paper counts:

```text
test         10 papers
validation    7 papers
train        29 papers
```

Resulting table counts:

```text
test         21 tables
validation   23 tables
train        50 tables
```

The workshop performs no training.

The canonical pipeline uses only the **21 held-out test tables**.

---

# 12. Why a text companion is necessary

The structure carrier currently exposes:

```text
image
+
structure boxes
```

but not a ready-made table string.

TAPAS requires:

```text
headers
+
cell strings
```

The workshop therefore needs a small **SciTSR text companion** derived from the same source logical cells and text chunks used to derive the current structure boxes.

This is not OCR inference.

It is dataset annotation.

---

# 13. Text-companion contract

At build time derive and pin:

```text
scitsr_text_companion.json
```

For every table:

```text
table_id
paper_id

n_rows
n_columns

cells:
  row_start
  row_end
  column_start
  column_end
  text
  box
```

For ordinary cells:

```text
row_start == row_end
column_start == column_end
```

Spanning cells retain their original row/column spans.

---

# 14. Text-companion provenance

The companion MUST record:

```text
SciTSR-PD dataset ID
immutable revision
source parquet digests

derivation script revision
companion SHA-256
```

The workshop repository may embed the resulting small JSON because the selected source material is public domain.

The notebook MUST NOT download arbitrary OCR or regenerate text with an OCR model.

---

# 15. Why use annotation text instead of OCR

This notebook is intended to isolate:

```text
table geometry
→ table reconstruction
→ table QA
```

If OCR were added here, final QA errors could originate from:

```text
table detection
structure recognition
OCR
cell assignment
TAPAS
```

making the learning workflow substantially harder to interpret.

The next notebook in the Document Intelligence queue owns OCR / Document Extraction.

---

# 16. End-to-end QA subset

The full structure test set contains 21 tables.

Select:

```text
END_TO_END_TABLES = 10
```

for the composed QA workflow.

Selection MUST occur **before model inference**.

---

# 17. Eligibility criteria

A table is eligible when its gold logical structure satisfies all of:

```text
no spanning cells

3..20 rows including header

2..12 columns

first row:
  every cell non-empty
  normalized headers unique

first body column:
  every cell non-empty
  values unique

at least one other column:
  contains >= 2 simply parseable numeric cells

all cells <= 200 characters

gold table fits TAPAS 512-token budget
```

This deliberately restricts the end-to-end QA subset to simple rectangular tables.

---

# 18. Deterministic selection

Among eligible held-out tables:

```text
ranking key =
SHA256("42:" + source_table_id)
```

Sort ascending.

Take the first:

```text
10
```

The build script MUST freeze those exact IDs into the workshop manifest.

If fewer than ten satisfy the criteria:

```text
build fails
```

Do not silently weaken the eligibility rules.

---

# 19. Complex-table probe

Separately select up to:

```text
3
```

held-out test tables containing spanning cells.

These are used only for qualitative structure-recognition analysis.

They do not flow into TAPAS.

Purpose:

> show why a simple row × column intersection grid is insufficient for complex tables.

---

# 20. Create document pages from real table crops

SciTSR-PD supplies cropped table images, while the detection model expects a page containing tables.

For each canonical table create a deterministic synthetic page.

The original table pixels MUST remain unchanged.

---

# 21. Page wrapper

Recommended composition:

```text
white page

top margin: 160 px
left/right margin: 120 px
bottom margin: 120 px
```

Place the original SciTSR table crop within the page.

Add only non-table contextual content such as:

```text
Table <source_id>
```

above the table and a generic caption below it.

Use Pillow's bundled font.

Do not place additional numeric/tabular structures on the page.

---

# 22. Ground-truth table box

Because the workshop itself places the crop, the exact page-level table box is known:

```text
[x0, y0, x1, y1]
```

Record:

```text
source table image digest
synthetic page digest
placement coordinates
```

This gives the detection stage measurable ground truth while preserving a real scientific table.

---

# 23. Detection stage

Run Table Transformer Detection on the page.

Native threshold:

```text
DETECTION_THRESHOLD = 0.90
```

Retain detections labelled:

```text
table
table rotated
```

sorted by score.

---

# 24. Selecting the target table

The canonical page contains one table.

Select:

```text
highest-scoring table/table-rotated detection
```

If no detection survives the threshold:

```text
status = detection_failed
```

No structure model is run for that page in the full end-to-end path.

The failure propagates downstream.

---

# 25. Detection metrics

Across the ten canonical pages report:

```text
AP50
AP75

detection hit rate @ IoU 0.50
detection hit rate @ IoU 0.75

mean best IoU
median best IoU

mean detections/page
```

Given the tiny sample, call these:

```text
tutorial metrics
```

not a detection benchmark.

---

# 26. Crop construction

For a detected box:

1. clip box to page bounds;
2. expand by:

```text
10 px
```

on every side;
3. clip again;
4. crop the page.

Record the page→crop transform.

No arbitrary resizing is performed before the structure model's own processor.

---

# 27. Rotation

The canonical pages contain upright tables.

If the detector unexpectedly returns:

```text
table rotated
```

record the label, but do not silently rotate the crop.

The detection model does not supply a reliable rotation direction.

No orientation correction is part of this workshop.

---

# 28. Structure stage — two paths

Run Table Transformer Structure Recognition through two separate paths.

## A. Gold-crop path

Use the exact known table box.

This isolates:

```text
structure-recognition quality
```

from page detection.

## B. Detected-crop path

Use the predicted table box.

This measures:

```text
detection
+
structure recognition
```

together.

---

# 29. Structure threshold

Use:

```text
RECOGNITION_THRESHOLD = 0.50
```

The notebook must expose it as a form parameter but keep the canonical value unchanged.

---

# 30. Structure labels

Preserve all six model labels:

```text
table
table column
table row
table column header
table projected row header
table spanning cell
```

Canonical grid construction primarily uses:

```text
table row
table column
```

Spanning cells are separately retained.

---

# 31. Same-label duplicate suppression

Before grid construction apply deterministic same-label NMS separately to:

```text
table row
table column
```

Suggested:

```text
IoU threshold = 0.50
```

Process boxes in descending score order.

This is a **workshop grid-building operation**, not part of the underlying model.

Raw detections must also be exported.

---

# 32. Grid reconstruction

After NMS:

### Rows

Sort by vertical center:

```text
top → bottom
```

### Columns

Sort by horizontal center:

```text
left → right
```

The implied grid is:

```text
n_rows × n_columns
```

Each cell rectangle is the intersection of one predicted row box and one predicted column box.

---

# 33. Invalid geometry

A reconstructed cell is invalid when:

```text
row ∩ column has zero or negative area
```

A table reconstruction is invalid when:

- zero rows;
- zero columns;
- any expected intersection is invalid;
- number of rows exceeds TAPAS limits;
- number of columns exceeds TAPAS limits.

Invalid reconstruction is a measurable pipeline failure.

Do not repair it with gold geometry.

---

# 34. Coordinate system

Structure detections originate in crop coordinates.

The notebook MUST retain transforms:
