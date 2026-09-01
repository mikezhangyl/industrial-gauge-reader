---
name: onboard-industrial-gauges
description: Onboard a new industrial gauge image or instrument type into industrial-gauge-reader by confirming the requested readout scope with the user, diagnosing the current pipeline, choosing metadata versus reusable code versus model training, and verifying the automated JSON/HTML report. Use for new gauge types, unfamiliar display regions, or failed readings; skip for merely opening an already-generated report.
---

# Onboard industrial gauges

Turn each new instrument into a reusable type capability, not a one-image patch. Human interaction owns the readout scope and business semantics; the program owns the automated value.

## Load the project contract

Locate the `industrial-gauge-reader` checkout. Read its nearest `AGENTS.md`, then read `analog-gauge-poc/docs/user-instrument-batch-runbook.md` completely before changing code or generating the final report. Preserve unrelated work in a dirty checkout.

## 1. Confirm the readout scope

Inspect the full image and enumerate every plausible display region before implementation. Keep separate physical meanings separate, including:

- analog or discrete pointers;
- mechanical counter windows;
- status or position indicators;
- digital displays;
- multiple physical instruments such as A/B/C phases.

Do not treat logos, serial numbers, nameplates, warning text, switch labels, reflections, shadows, pointer counterweights, or decorative marks as readouts unless the user explicitly needs them.

Present one concise proposal and ask the user to confirm it:

```text
读取范围确认
- 图片/实例：<visible label or position>
- 建议读取：<display region → business meaning → unit if known>
- 建议不读取：<display regions and why>
- 尚需确认：<only the ambiguity that changes implementation>

请确认“读取/不读取”的范围是否正确。
```

Ask for scope and semantics, not the numeric answer. Only ask for a human reading when the user explicitly requests labeled regression or acceptance comparison.

Resolve these ambiguities with the user when they affect the output:

- which pointer or display is operationally relevant;
- whether counters, internal status pointers, or auxiliary windows are in scope;
- the active range for a dual-scale meter; preserve all valid candidates when it is unknown;
- instance labels and expected count for multi-instrument images;
- whether a displayed number is a measurement, a position, or an accumulated count.

Do not start behavioral implementation until the scope is confirmed. Type metadata remains complete even when a channel is excluded from the current batch. Apply batch/report scope separately; if the project lacks a reusable scope filter, add one rather than deleting type knowledge or hard-coding the current image.

## 2. Establish the baseline

Run the current automated pipeline on the original image before changing anything. Preserve the baseline JSON or terminal output and classify the earliest failure:

1. instrument type was not uniquely matched;
2. dial or display region was not detected;
3. pointer, counter, or status element was not extracted;
4. geometry or scale mapping failed;
5. the visual value was found but its business meaning was wrong;
6. the value was correct but the report included out-of-scope channels.

Use the earliest failing boundary to choose the change. Do not infer the cause only from the final HTML status.

## 3. Choose the smallest reusable extension

Use this order:

### Metadata-only

Add or amend a type knowledge bundle when existing visual primitives work and the gap is the model marking, channel meaning, unit, scale, allowed value, normalization, or evidence source.

### Reusable program capability

Change code when the display geometry is genuinely new, such as a rectangular meter, discrete position dial, counter window, counterweight, or type-independent OCR/geometry recovery. Name the capability after the visual problem, not after the sample filename. Add no fixed ROI, center, pointer angle, label coordinate, or expected reading from the image.

Instrument-specific routing is acceptable when metadata uniquely identifies the type, but the underlying detector or interpreter must remain reusable for other images of that type.

### Model training

Consider training or fine-tuning only after representative images show that the pretrained detector or segmenter cannot reliably cover the visual class and metadata/program logic cannot solve the gap. Confirm dataset scope, labeling effort, evaluation split, licensing, and user authorization before changing weights. One difficult image is not evidence that training is necessary.

State explicitly which level was used. Never describe a program or metadata change as model training.

## 4. Add knowledge without mixing facts

Store reusable professional knowledge in `metadata/instrument-types/<type_id>/metadata.json`. Keep a verified manufacturer manual in the same local type directory when available and record its source, document code, filename, and digest as required by the project.

Keep photo-specific confirmation out of type metadata. New-image automated reports require no observation. If the user explicitly supplies a ground-truth value for regression, store it by image digest in local `observations/`; it must not become model input or appear in the automated HTML.

Examples of semantic separation:

- `005` in an arrester monitor counter window is five accumulated operations, not `5 mA`;
- an outer tap-position pointer, an inner mechanism-status pointer, and an operation counter are independent channels;
- a single-revolution discharge counter shows the current visible digit, not necessarily lifetime operations.

## 5. Implement and verify

Add a targeted regression test at the boundary being extended, then run the original image again. Verify all of the following:

- every confirmed in-scope instance/channel appears once;
- excluded channels are absent from the current report scope;
- ambiguous scales remain candidates rather than invented single values;
- source images are embedded in the HTML;
- HTML shows only automated results and omits confidence, human answers, and human/automatic comparison;
- JSON retains complete automated diagnostics;
- previously supported instrument tests still pass;
- code checks, type checks, and `git diff --check` pass.

Use the real model-backed command for final validation. A unit test or mocked run does not prove the image works.

## 6. Report the evidence boundary

Deliver the generated HTML and JSON paths, the confirmed readout scope, the extension level used, the automated status for each requested channel, and any remaining ambiguity.

Use these maturity labels:

- `sample-verified`: the supplied image works;
- `type-regression-covered`: multiple representative images of the type pass regression;
- `model-trained`: weights were actually trained and evaluated.

Do not promote one successful image to type-wide reliability. When the image is unreadable, the active range is unknowable, or the business meaning is unresolved, keep the result ambiguous or unavailable and ask for the minimum missing information.
