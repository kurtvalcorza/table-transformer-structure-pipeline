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

```text
crop coordinates
→ page coordinates
→ original SciTSR table coordinates
```

All metrics must compare boxes in one explicitly named coordinate frame.

Recommended:

```text
synthetic page pixel coordinates
```

---

# 35. Structure metrics

For both:

```text
gold crop
detected crop
```

report:

```text
row AP50
row AP75

column AP50
column AP75

mean best row IoU
mean best column IoU

row-count exact rate
column-count exact rate
grid-shape exact rate
```

---

# 36. Header-label handling

SciTSR-PD does not annotate:

```text
table column header
table projected row header
```

Therefore those predictions:

- remain visible;
- are exported;
- may be shown in visualization;

but are excluded from quantitative penalties.

---

# 37. Spanning-cell handling

The ten QA tables contain no gold spanning cells.

Therefore the rectangular reconstruction contract is valid.

For the separate complex-table probe:

```text
table spanning cell
```

detections are visualized and compared to gold boxes.

No rectangular table reconstruction is claimed for those examples.

---

# 38. Cell-text assignment

The canonical notebook uses the **SciTSR annotation text provider**.

For every gold logical cell:

1. calculate the center of its gold cell box;
2. determine which predicted row contains the center;
3. determine which predicted column contains the center;
4. assign the cell text to that predicted `(row, column)`.

No gold row/column index is supplied to the reconstruction algorithm.

---

# 39. Ambiguous assignment

A source cell is considered unmapped when:

- no predicted row contains its center;
- no predicted column contains its center.

It is ambiguous when multiple predicted rows or columns contain the center after NMS.

Ambiguous/unmapped cells count as reconstruction failures.

Do not silently choose gold coordinates.

---

# 40. Reconstructed text grid

Result:

```text
[
  [header1, header2, ...],
  [cell,    cell,    ...],
  ...
]
```

For the QA subset, one annotation cell is expected per logical position.

If several text fragments map into one predicted cell, join them in source reading order with a single space.

---

# 41. Header construction

The first reconstructed row becomes TAPAS column names.

Required:

```text
all headers non-empty
all normalized headers unique
```

If this fails:

```text
reconstruction_valid_for_qa = false
```

Do not substitute the gold header.

---

# 42. Reconstruction metrics

For each table report:

```text
gold_rows
predicted_rows

gold_columns
predicted_columns

shape_exact

gold_cells
mapped_cells
unmapped_cells
ambiguous_cells

cell_assignment_accuracy
cell_text_accuracy

header_exact
reconstruction_valid_for_qa
```

---

# 43. Cell-assignment accuracy

For every gold simple cell, compare:

```text
gold logical (row, column)
```

against the predicted grid position receiving that cell's text.

Metric:

```text
correctly assigned gold cells
/
all gold cells
```

This measures geometry-to-grid correctness.

---

# 44. Cell-text accuracy

After reconstruction compare the predicted grid position-by-position with the gold logical grid.

Normalize only:

```text
case
outer whitespace
repeated whitespace
```

Do not apply fuzzy matching.

Report:

```text
matched cells / gold cells
```

Missing or extra rows/columns count as mismatches.

---

# 45. Three QA paths

This is the central evaluation design.

## Path 1 — Gold table → TAPAS

```text
gold logical table
→ TAPAS
```

Measures primarily:

```text
QA-model capability
```

---

## Path 2 — Gold crop → predicted structure → TAPAS

```text
exact table crop
→ structure model
→ reconstructed table
→ TAPAS
```

Measures:

```text
structure + QA
```

without table-detection error.

---

## Path 3 — Detected crop → predicted structure → TAPAS

```text
page
→ table detector
→ predicted crop
→ structure model
→ reconstructed table
→ TAPAS
```

Measures:

```text
full visual table pipeline
```

---

# 46. Why use a waterfall

Final QA accuracy alone cannot identify whether a wrong answer originated from:

```text
table detection
row/column detection
grid reconstruction
cell assignment
TAPAS cell selection
TAPAS aggregation
```

The three paths expose where quality is lost.

---

# 47. QA dataset generation

Generate exactly:

```text
5 questions/table
```

for each of the ten canonical tables.

Total:

```text
50 questions
```

Question generation happens entirely from the **gold logical table** before any model output is inspected.

---

# 48. Lookup questions

Generate two `NONE` questions per table.

Require the first body column to contain unique row identifiers.

Choose deterministic:

```text
(row, target-column)
```

pairs using a SHA-256 ranking of eligible cells.

Example:

```text
What is the Accuracy for Model B?
```

Gold:

```text
aggregation = NONE
coordinates = [[row, column]]
denotation = [cell_text]
```

---

# 49. Numeric column

Select one body column whose cells are all compatible with the workshop's simple numeric parser.

Accepted examples:

```text
12
1,250
42.7
-3.5
18%
```

Reject columns containing expressions such as:

```text
12 ± 0.4
<0.05
3/7
```

for the aggregation exercises.

---

# 50. SUM question

Generate:

```text
What is the total <column header>?
```

Gold:

```text
aggregation = SUM
coordinates = all body cells in numeric column
denotation = computed numeric sum
```

---

# 51. AVERAGE question

Generate:

```text
What is the average <column header>?
```

Gold:

```text
aggregation = AVERAGE
coordinates = all body cells in numeric column
denotation = arithmetic mean
```

---

# 52. COUNT question

Generate:

```text
How many <column header> entries are listed?
```

Gold:

```text
aggregation = COUNT
coordinates = all body cells in numeric column
denotation = body row count
```

---

# 53. Question-generation assertions

Per table:

```text
2 NONE
1 SUM
1 AVERAGE
1 COUNT
```

Across ten tables:

```text
20 NONE
10 SUM
10 AVERAGE
10 COUNT
```

No model output affects the generated question set.

---

# 54. TAPAS metrics — gold-table path

For the gold table report:

```text
denotation accuracy
aggregation accuracy
cell-coordinate accuracy
```

This uses the current DIMER TAPAS metric semantics:

### Denotation

Numeric results:

```text
match within 1e-6
```

Lookup answers:

```text
same multiset of lower-cased cell strings
```

### Aggregation

Predicted operator exactly equals gold.

### Cell accuracy

Selected coordinates exactly equal the gold set.

---

# 55. TAPAS metrics — reconstructed paths

For reconstructed tables report:

```text
denotation accuracy
aggregation accuracy
```

Cell-coordinate accuracy SHOULD NOT be treated as directly comparable when the reconstructed grid has a different geometry.

Instead separately report the table reconstruction metrics.

---

# 56. Invalid reconstruction handling

If a reconstructed table is invalid for TAPAS:

```text
QA result = pipeline failure
```

All five questions for that table count as:

```text
denotation incorrect
```

for the end-to-end metric.

Do not drop failed tables from the denominator.

This is essential.

---

# 57. TAPAS truncation

Record for every call:

```text
n_tokens
tokens_before_truncation
rows_kept
truncated
```

The canonical subset is selected to fit the gold table without truncation.

A reconstructed table may nevertheless differ.

Truncation must remain visible.

---

# 58. QA baselines

On the gold tables include the current TAPAS baselines:

### First-cell baseline

Always selects:

```text
row 0, column 0
```

with aggregation:

```text
NONE
```

### Keyword lookup baseline

Uses:

- header overlap;
- row-value mentions;
- question wording for aggregation.

These provide a non-neural floor.

---

# 59. Main QA comparison

Produce:

| Path | Denotation Accuracy | Aggregation Accuracy | Valid Tables |
|---|---:|---:|---:|
| First-cell baseline — gold | measured | measured | 10/10 |
| Keyword baseline — gold | measured | measured | 10/10 |
| TAPAS — gold table | measured | measured | 10/10 |
| TAPAS — predicted structure on gold crop | measured | measured | measured |
| TAPAS — full detected pipeline | measured | measured | measured |

Do not select a winner.

The rows represent different levels of upstream information, not competing models.

---

# 60. Pipeline waterfall

Produce a compact summary:

```text
10 source tables

↓ table detection
N detected @ IoU ≥ .50

↓ structure
N exact grid shapes

↓ reconstruction
N valid string tables

↓ TAPAS
N / 50 correct denotations
```

Also report the gold-crop structure path in parallel.

---

# 61. Failure taxonomy

Every table/question should receive a stage status where applicable:

```text
ok

detection_miss
poor_detection_iou

no_rows
no_columns
grid_invalid

header_missing
header_duplicate

cell_unmapped
cell_ambiguous

tapas_truncated
tapas_wrong_cells
tapas_wrong_aggregation
tapas_wrong_denotation
```

A record may carry more than one diagnostic finding.

---

# 62. Workshop prediction exercise

Before revealing results ask participants:

1. Which stage is most likely to reduce end-to-end accuracy?
2. Can perfect table detection guarantee correct structure?
3. Can perfect structure guarantee correct QA?
4. Why does TAPAS need strings rather than row/column boxes?
5. Where would OCR enter a production version of this workflow?
6. What happens if one predicted row is missing?
7. What happens if the first reconstructed row is mistaken?

Then reveal the waterfall.

---

# 63. Visualizations

For each canonical table create a four-stage panel:

### Panel 1 — Page

Synthetic page + predicted table box.

### Panel 2 — Crop

Detected crop + structure boxes:

```text
rows
columns
spanning cells
```

### Panel 3 — Reconstructed table

Render predicted grid as a compact text table.

### Panel 4 — QA

Display one sample question:

```text
question
gold answer
TAPAS answer
selected cells
aggregation
```

---

# 64. Complex-table visualization

For the three spanning-cell probes show:

```text
source crop
gold spanning cells
predicted spanning cells
predicted rows/columns
```

Include the message:

> This table is intentionally excluded from the simple rectangular QA reconstruction contract.

---

# 65. Detection-vs-structure comparison

For each canonical table record:

```text
detection_iou

structure_grid_exact_gold_crop
structure_grid_exact_detected_crop

cell_accuracy_gold_crop
cell_accuracy_detected_crop
```

This directly shows whether crop quality changes downstream structure quality.

---

# 66. Resource measurement

Measure separately for:

1. Table Transformer Detection
2. Table Transformer Structure
3. TAPAS

Record:

```text
snapshot verification time
model load time
parameter count
weight bytes

mean inference latency
median inference latency
total stage time

peak GPU memory
```

---

# 67. Sequential model loading

Do not retain all three models in accelerator memory simultaneously.

Canonical order:

```text
load detector
→ process all pages
→ unload

load structure recognizer
→ process gold and detected crops
→ unload

load TAPAS
→ process all QA paths
→ unload
```

Retain only normalized outputs between stages.

---

# 68. Runtime

Common runtime family:

```text
Python 3.12

torch==2.14.0
torchvision==0.29.0
torchaudio==2.11.0

transformers==4.57.6
safetensors==0.8.0

numpy==2.5.3
pillow==11.3.0
huggingface-hub==0.36.2
```

TAPAS may additionally require the pinned pandas compatibility layer used by its current carrier.

Use the same implementation strategy as the existing DIMER TAPAS notebook.

---

# 69. Default parameters

```python
USE_BYOD = False

DETECTION_THRESHOLD = 0.90
STRUCTURE_THRESHOLD = 0.50

CROP_PADDING = 10
GRID_NMS_IOU = 0.50

END_TO_END_TABLES = 10
QUESTIONS_PER_TABLE = 5

RUN_COMPLEX_STRUCTURE_PROBE = True
COMPLEX_STRUCTURE_TABLES = 3

OUTPUT_DIR = "outputs/table_intelligence"
```

---

# 70. Standalone requirements

The notebook MUST NOT:

```text
git clone
pip install -e .
import any DIMER pipeline repository
fetch DIMER source code
call DIMER workers
call DIMER APIs
```

Core logic must be notebook-local using:

- Transformers;
- PyTorch;
- Pillow;
- NumPy;
- general-purpose data libraries.

---

# 71. Model acquisition

Embed all three exact DIMER manifests.

For every snapshot:

```text
model ID
immutable revision
file list
byte lengths
SHA-256
```

Fetch only listed files.

Verify every asset before trusted load.

Use:

```text
local_files_only=True
trust_remote_code=False
```

No `.bin` fallback.

---

# 72. Machine-readable output directory

Use:

```text
outputs/table_intelligence/
```

---

# 73. `table_detection.csv`

```text
table_id
page_id

gt_box

predicted
label
score
predicted_box

iou

hit50
hit75

latency_seconds
```

---

# 74. `structure_objects.csv`

One row per raw structure detection:

```text
table_id
path

gold_crop | detected_crop

label
score

x0
y0
x1
y1

page_x0
page_y0
page_x1
page_y1
```

---

# 75. `structure_metrics.csv`

```text
table_id
path

gold_rows
predicted_rows

gold_columns
predicted_columns

row_count_exact
column_count_exact
grid_shape_exact

row_ap50
row_ap75
column_ap50
column_ap75

mean_row_iou
mean_column_iou
```

---

# 76. `reconstruction_metrics.csv`

```text
table_id
path

gold_rows
gold_columns

predicted_rows
predicted_columns

shape_exact

gold_cells
mapped_cells
unmapped_cells
ambiguous_cells

cell_assignment_accuracy
cell_text_accuracy

header_exact
valid_for_tapas
```

---

# 77. Reconstructed tables

Write:

```text
tables/
  <table_id>_gold.json
  <table_id>_gold_crop_reconstructed.json
  <table_id>_detected_reconstructed.json
```

Each contains:

```text
headers
rows
shape
source
findings
```

Optionally also export CSV.

---

# 78. `qa_questions.jsonl`

One canonical record/question:

```text
id
table_id

question
category

gold_aggregation
gold_coordinates
gold_denotation
```

Exactly:

```text
50 records
```

---

# 79. `qa_predictions.csv`

```text
question_id
table_id

path
baseline | gold | structure_only | end_to_end

question

gold_aggregation
predicted_aggregation

gold_denotation
predicted_denotation

denotation_correct
aggregation_correct

selected_cells
coordinates

truncated
pipeline_status

latency_seconds
```

---

# 80. `waterfall.json`

Example structure:

```json
{
  "tables": 10,
  "detection": {
    "hit50": 0
  },
  "structure_gold_crop": {
    "grid_exact": 0
  },
  "structure_detected_crop": {
    "grid_exact": 0
  },
  "reconstruction": {
    "valid_tables": 0
  },
  "qa": {
    "questions": 50,
    "gold_table_denotation_accuracy": 0,
    "structure_only_denotation_accuracy": 0,
    "end_to_end_denotation_accuracy": 0
  }
}
```

---

# 81. `resource_metrics.csv`

```text
stage
model
model_id
revision

parameter_count
weight_bytes

load_seconds
mean_latency_seconds
median_latency_seconds
total_seconds

peak_gpu_memory_bytes
```

---

# 82. `provenance.json`

Must record:

```text
notebook_spec
profile
pedagogical_mode

SciTSR-PD:
  dataset ID
  immutable revision
  source-file digests
  licence
  image/table manifest
  split seed
  paper-disjoint split

text companion:
  derivation revision
  file SHA-256

canonical table IDs
eligibility rules

synthetic page:
  rendering parameters
  font identity
  page digests
  table placement

Table Transformer Detection:
  ID
  revision
  weight digest

Table Transformer Structure:
  ID
  revision
  weight digest

TAPAS:
  ID
  revision
  weight digest

thresholds
grid-building policy
question-generation policy

runtime
device
timings
```

---

# 83. Input manifest

Export:

```text
input_manifest.json
```

including:

```text
94 source tables verified
21 held-out test tables

10 QA-eligible canonical tables
3 complex probes if available

table dimension ranges
row/column ranges
text-length ranges

validation findings
```

---

# 84. BYOD full-pipeline contract

A real deployment needs text to populate reconstructed cells.

Therefore BYOD requires:

```text
page image
+
OCR/text boxes
```

The notebook does not run OCR itself.

---

# 85. BYOD format

Recommended:

```text
dataset/
  pages/
    page001.png
    page002.png

  ocr.csv
  questions.csv       optional
```

`ocr.csv`:

```text
page_id
text
x0
y0
x1
y1
order
```

---

# 86. BYOD questions

Optional `questions.csv`:

```text
id
page_id
question
accepted_answer
```

If `accepted_answer` is absent:

```text
QA evaluation = not-measurable
```

but predictions are still exported.

---

# 87. BYOD table assumption

For workshop simplicity:

> Each page should contain one primary target table.

If several tables survive the detection threshold:

```text
highest-scoring detection
```

is processed and a warning is emitted.

The notebook must not claim robust multi-table page routing.

---

# 88. BYOD OCR assignment

After structure reconstruction:

1. take each OCR word center;
2. locate predicted row;
3. locate predicted column;
4. assign word to that cell;
5. order words by the supplied `order` column;
6. join with spaces.

This replaces the SciTSR annotation text provider.

---

# 89. BYOD limits

Recommended:

```text
1..20 pages

page sides:
16..4096 px

OCR words/page:
1..5000

detected tables:
up to model query ceiling

reconstructed table:
<=64 rows
<=32 columns

cell strings:
<=200 chars
```

---

# 90. BYOD privacy warning

Tables commonly contain:

- financial data;
- research results;
- personal records;
- employee data;
- medical information;
- customer information.

Do not upload confidential, restricted or regulated documents into a hosted notebook runtime without authorization.

---

# 91. OCR boundary

The workshop MUST explicitly state:

> Table Transformer models recover geometry, not text.

The canonical notebook uses dataset annotation text.

A real system must obtain cell text from:

- PDF text extraction;
- OCR;
- another document-extraction model.

That dependency is intentionally deferred to the next Document Intelligence notebook.

---

# 92. Required assertions

## Model assets

```text
all 3 model IDs exact
all revisions exact
all weight SHA-256 values exact
all manifests verified
```

---

## SciTSR-PD

```text
94 source tables
46 papers

50 train tables
23 validation tables
21 test tables

paper-disjoint split

all image digests valid
all text-companion records align
```

---

## QA subset

```text
10 canonical tables

no spanning cells
row count 3..20
column count 2..12

headers non-empty
headers unique

first body column unique

numeric aggregation column exists

gold TAPAS representation fits token budget
```

---

## Questions

```text
50 questions

20 NONE
10 SUM
10 AVERAGE
10 COUNT
```

---

## Detection

```text
scores finite
boxes finite
boxes inside page
<=15 detections
```

---

## Structure

```text
scores finite
boxes valid
labels in six-class vocabulary
<=125 detections
```

---

## Reconstruction

```text
every mapped cell has unique predicted row/column
headers valid before TAPAS
no gold geometry silently substituted
```

---

## TAPAS

```text
<=64 rows
<=32 columns

all headers/cells str

aggregation in:
NONE/SUM/AVERAGE/COUNT

cell-selection threshold = 0.5
```

---

# 93. What MUST NOT be asserted

The notebook must not require:

```text
table detection must be perfect

structure on gold crops must outperform detected crops

TAPAS must answer gold tables better than reconstructed tables

EVA-style larger models are irrelevant here

a particular structure threshold must be optimal

end-to-end denotation must exceed a fixed score
```

All quality values are observations.

---

# 94. Notebook cell plan

| # | Type | Section |
|---:|---|---|
| 0 | Markdown | Title and objectives |
| 1 | Markdown | What “Table Intelligence” means |
| 2 | Markdown | Three-stage architecture |
| 3 | Code | Form parameters |
| 4 | Markdown | Runtime |
| 5 | Code | Install/check pins |
| 6 | Markdown | Model provenance |
| 7 | Code | Three embedded manifests |
| 8 | Markdown | SciTSR-PD provenance |
| 9 | Code | Load/verify 94 tables + text companion |
| 10 | Markdown | Paper-disjoint test split |
| 11 | Code | Reproduce 50/23/21 table split |
| 12 | Markdown | QA-eligible subset |
| 13 | Code | Deterministically select 10 tables |
| 14 | Markdown | Create page wrappers |
| 15 | Code | Render/verify synthetic pages |
| 16 | Markdown | Table detection |
| 17 | Code | Stage/load detector |
| 18 | Code | Detect ten tables |
| 19 | Markdown | Detection interpretation |
| 20 | Code | Detection metrics |
| 21 | Code | Unload detector |
| 22 | Markdown | Structure recognition |
| 23 | Code | Stage/load structure model |
| 24 | Code | Recognize gold crops |
| 25 | Code | Recognize detected crops |
| 26 | Markdown | Grid reconstruction |
| 27 | Code | Row/column NMS + sorting |
| 28 | Code | Build predicted grids |
| 29 | Markdown | Assign annotation text |
| 30 | Code | Build reconstructed tables |
| 31 | Markdown | Structure/reconstruction metrics |
| 32 | Code | Gold-vs-detected crop comparison |
| 33 | Markdown | Complex spanning-cell probe |
| 34 | Code | Qualitative complex examples |
| 35 | Code | Unload structure model |
| 36 | Markdown | Generate QA questions |
| 37 | Code | Create 50 questions |
| 38 | Markdown | TAPAS semantics |
| 39 | Code | Stage/load TAPAS |
| 40 | Markdown | Gold-table QA |
| 41 | Code | Baselines + TAPAS gold tables |
| 42 | Markdown | Structure-only QA |
| 43 | Code | TAPAS on gold-crop reconstruction |
| 44 | Markdown | Full end-to-end QA |
| 45 | Code | TAPAS on detected reconstruction |
| 46 | Markdown | Pipeline waterfall |
| 47 | Code | Waterfall + failure taxonomy |
| 48 | Markdown | Workshop interpretation exercise |
| 49 | Code | Selected example outputs |
| 50 | Markdown | Resource comparison |
| 51 | Code | Stage timing/memory |
| 52 | Markdown | Visualization |
| 53 | Code | Four-stage panels |
| 54 | Markdown | Machine-readable exports |
| 55 | Code | CSV/JSON/provenance |
| 56 | Markdown | BYOD |
| 57 | Code | Optional BYOD branch |
| 58 | Markdown | Interpretation and limitations |
| 59 | Code | Terminal summary + assertions |

Every executable cell must be preceded by explanatory markdown.

---

# 95. Interpretation requirements

## Detection and structure are different problems

A detector can find the correct table while the structure model still misses rows or columns.

---

## Correct structure does not provide text

Rows and columns provide geometry.

A downstream text provider is still required.

---

## Cell assignment is an independent failure point

Even when:

```text
row count correct
column count correct
```

small box shifts may place a text fragment in the wrong cell.

---

## TAPAS cannot repair a malformed table reliably

Once reconstruction changes:

- headers;
- rows;
- columns;
- cell positions;

the semantic table received by TAPAS is different.

---

## Gold-table QA is an upper-stage diagnostic

Strong TAPAS accuracy on the gold table does not imply strong end-to-end performance.

Conversely, poor end-to-end QA does not necessarily mean TAPAS itself failed.

---

## OCR is intentionally absent

The annotation text provider removes OCR as a confounder.

This notebook therefore measures:

```text
visual table geometry
+
table reconstruction
+
table QA
```

not OCR quality.

---

# 96. Explicit non-goals

The notebook does not perform:

- OCR;
- text recognition;
- PDF parsing;
- multi-table routing;
- spanning-cell normalization for QA;
- nested table handling;
- table editing;
- model fine-tuning;
- DePlot chart extraction;
- chart QA;
- spreadsheet formula inference;
- semantic type inference;
- LLM table reasoning;
- DIMER API calls.

---

# 97. Relationship to DePlot

Document-table workflow:

```text
page
→ detect table
→ recognize structure
→ extract cells
→ TAPAS
```

Chart workflow:

```text
chart image
→ DePlot
→ generated data table
```

These should remain separate.

---

# 98. Relationship to the next notebook

Current Document Intelligence track:

```text
Document Intelligence
│
├── Document-Type Classification
│     └── DiT / RVL-CDIP
│
├── Document QA
│     └── LayoutLM vs Pix2Struct
│
├── Table Intelligence
│     ├── Table Transformer Detection
│     ├── Table Transformer Structure
│     └── TAPAS
│
└── OCR / Document Extraction
      └── next
```

The final OCR notebook will provide the capability that a production version of this workflow would use in place of SciTSR's annotation text.

---

# 99. Recommended repository placement

```text
ml-worker/
  integrations/
    dimer/
      workshops/
        table-intelligence/
          DIMER_Table_Intelligence_Workshop.ipynb
          README.md

          data/
            scitsr_text_companion.json
            workshop_manifest.json

          docs/
            release-verification.md

          tools/
            build_notebook.py
            build_scitsr_companion.py
            validate_notebook.py
```

The three carrier repositories can link to the central workshop.

---

# 100. Release verification

Preferred environment:

```text
Kaggle Tesla T4
Python 3.12
```

Cold-run qualification must verify:

```text
no repository checkout
no DIMER services
no credentials

all 3 models staged and verified

94 SciTSR tables verified
text companion verified

50 / 23 / 21 split reproduced

10 canonical tables selected
50 QA records generated

table detection completes
gold-crop structure completes
detected-crop structure completes
grid reconstruction completes

TAPAS gold-table evaluation completes
TAPAS structure-only evaluation completes
TAPAS end-to-end evaluation completes

complex structure probe completes
BYOD disabled by default

all exports exist
```

---

# 101. Release evidence

Record:

```text
notebook commit
notebook blob

runtime
GPU

SciTSR revision
source digests
companion digest

canonical table IDs
page digests

Detection:
  model revision
  weight digest
  AP50 / AP75
  mean IoU
  timing
  memory

Structure:
  model revision
  weight digest
  gold-crop metrics
  detected-crop metrics
  timing
  memory

Reconstruction:
  grid-shape accuracy
  cell assignment accuracy
  cell text accuracy

TAPAS:
  model revision
  weight digest
  gold-table metrics
  structure-only metrics
  end-to-end metrics
  timing
  memory

wall time
cell success count
output inventory
```

---

# 102. Terminal summary

The final cell should resemble:

```text
DIMER Table Intelligence Workshop
---------------------------------

Tables:
  source test tables: 21
  canonical QA tables: 10
  questions: 50

Stage 1 — Table detection
  AP50: ...
  hit@0.50: ...
  mean IoU: ...

Stage 2 — Structure
  grid exact — gold crop: ...
  grid exact — detected crop: ...

Stage 3 — Reconstruction
  valid tables: ... / 10
  cell assignment accuracy: ...
  cell text accuracy: ...

Stage 4 — TAPAS
  gold-table denotation: ...
  structure-only denotation: ...
  end-to-end denotation: ...

Outputs:
  outputs/table_intelligence/
```

Do not print a model winner.

---

# 103. Workshop learning arc

**Page understanding**  
↓  
*First locate the table.*

**Table localization**  
↓  
*Crop the correct region.*

**Structure understanding**  
↓  
*Recover rows and columns.*

**Grid construction**  
↓  
*Turn geometry into cells.*

**Text assignment**  
↓  
*Cells still need text.*

**Structured representation**  
↓  
*Now the image has become a machine-readable table.*

**Table reasoning**  
↓  
*TAPAS selects cells and aggregation operators.*

**Error waterfall**  
↓  
*A correct final answer depends on every upstream stage.*

**System-design lesson**  
↓  
*Table Intelligence is not one model. It is a pipeline whose interfaces—detection, structure, text extraction, reconstruction and reasoning—must each be measured independently.*
