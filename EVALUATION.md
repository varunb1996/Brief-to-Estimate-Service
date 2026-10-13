# Evaluation

One eval run on 2026-10-13: 25 labeled cases in `eval/cases.json`, 3 runs each. The primary model was `dots-studio/dots-3-note-preview:free` and the fallback was `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free`, both on OpenRouter. Raw output is in `eval/results.json`. The replay with the current code is in `eval/results_rescored.json`.

The run hit the free tier's limit of 50 requests per day after 52 calls. From case t08 on, both models returned 429, so nine cases never reached a model: t08, t09, t10, and the six photo cases i01 to i06. Each came back as `fallback`, 27 runs in all. The empty-brief case t07 accounts for the other 3 `fallback` runs; it's supposed to skip the model. The "all 25" column below includes the quota failures. The "reached a model" columns drop the nine blocked cases and keep t07.

## Metrics

Field accuracy counts only fields listed in a case's `expected_spec`. Text fields are compared after the matcher's normalization, lists as sets, bead size within 0.5mm and length within 1cm. "Before" is the code that produced `results.json`. "After" replays the same saved extractions through the code with the three fixes described under the failure cases, using `--rescore`, so it made no new API calls.

| field | n, all 25 | all 25 | n, reached | before | after | same value in all 3 runs |
|---|---|---|---|---|---|---|
| product_type | 24 | 62% | 15 | 100% | 100% | 100% |
| bead_material | 21 | 59% | 14 | 88% | 100% | 79% |
| bead_size_mm | 20 | 70% | 14 | 100% | 100% | 100% |
| length_cm | 19 | 67% | 13 | 97% | 97% | 92% |
| bead_color | 19 | 58% | 11 | 100% | 100% | 100% |
| string_material | 15 | 53% | 11 | 73% | 100% | 91% |
| clasp_type | 8 | 62% | 5 | 100% | 100% | 100% |
| bead_shape | 3 | 33% | 1 | 100% | 100% | 100% |
| stated_bead_count | 1 | 100% | 1 | 100% | 100% | 100% |

| metric | all 25 | reached, before | reached, after |
|---|---|---|---|
| top-1 match | 72% | 94% | 94% |
| top-3 match | 72% | 94% | 94% |
| flag rate | 56% | 31% | 25% |
| flag precision | 71% | 80% | 100% |
| flag recall | 91% | 80% | 80% |
| mean latency per run | 12.0s | 18.1s | 18.1s |
| total latency | 898.2s | 869.0s | 869.0s |
| cost | $0 | $0 | $0 |

The all-25 latency mean is lower only because quota-blocked runs fail in under a second.

The "after" column is optimistic for two reasons. I designed the fixes after looking at these 16 cases, so this set can't show how well they generalize. And part of the material gain is scoring, not extraction. The model still writes "gold brass", and both the matcher and the scorer now reduce it to "brass".

## CV method comparison

The 13 synthetic photos each target one condition. Hough is OpenCV's `HOUGH_GRADIENT_ALT` with the parameter values from its documentation. Contour is an adaptive threshold plus an area and circularity filter. Blobs that fail the circularity check are split by area into single beads.

| image | truth | Hough | contour | condition |
|---|---|---|---|---|
| 01_clean_separated | 12 | 12 | 12 | separated |
| 02_small_separated | 30 | 30 | 30 | separated, small |
| 03_strand_spaced | 18 | 18 | 19 | strand with thread |
| 04_strand_touching | 24 | 24 | 24 | touching strand |
| 05_overlap | 16 | 16 | 16 | 30% overlap |
| 06_glare | 14 | 14 | 14 | specular highlights |
| 07_mixed_sizes | 24 | 24 | 24 | 8 large, 16 small |
| 08_low_contrast_pearls | 14 | 14 | 37 | white on light gray |
| 09_shadows | 14 | 14 | 14 | cast shadows |
| 10_noisy_blurry | 20 | 19 | 20 | heavy noise and blur |
| 11_cubes | 10 | 0 | 10 | not round |
| 12_dense_pile | 40 | 40 | 40 | dense overlapping pile |
| 13_seed_beads | 60 | 0 | 56 | 6px radius, touching |

| method | mean abs error | within 10% of truth |
|---|---|---|
| Hough | 15.8% | 11/13 |
| contour, before blob splitting | 40.5% | 7/13 |
| contour, with blob splitting | 13.6% | 12/13 |

On the six photo cases in the main eval, Hough had an MAE of 11.7 beads (33.3%) and contour 4.7 beads (29.4%). The vision model's own count was scored on 0 of 18 runs because of the quota.

Hough is close to perfect on round beads, because my synthetic beads are exactly what it searches for. It returns 0 on cubes and on beads smaller than its minimum radius. Contour handles both of those, but on pearls it counts crescents. The service only uses a photo count when the two methods agree within 20%, so in each of these failures the photo count was flagged and not used. I added blob splitting after seeing these images, so its gain here overstates what it will do on new photos. Real photos will be harder for both methods.

## Three failure cases

I picked one failure from each stage of the pipeline. Each root cause was checked against saved outputs, recomputed scores, or a debug image.

**Extraction: "stretch bracelet" lost the string.** Cases c01, c05 and t02 ("pink rose quartz stretch bracelet, 8mm...", "Amethyst stretch bracelet...", "Stretch bracelet for a 6.5 inch wrist...") should give `string_material: elastic`. All 9 runs returned null. When "stretch" was attached to the string, the model got it: "on stretch cord" (c08) and "6mm pink beads, stretch, 7 inches" (t01) were both extracted. So the model reads "stretch" in front of "bracelet" as a style, not a material. The missing string had a cost: the estimate assumed a clasp, took 1.5cm off, and came out 2 beads short each time (20, 16 and 21 instead of 22, 18 and 23). I added a rule after extraction: if the brief contains "stretch" and the string is empty, set it to elastic. On replay, string accuracy went from 73% to 100%. I didn't change the prompt, because I couldn't test a prompt change without API calls.

**Matching: an unstocked color still matched.** t04, "Yellow glass bead bracelet, 8mm, elastic, 18cm", should return no SKU and a flag. Extraction got `["yellow"]` right, but the catalog has no yellow. BRC-001 (black) and BRC-002 (blue) both scored 0.80: full marks on type, material, size, length and string, 0.0 on color and clasp. Color weighs 0.15, and the 0.6 match threshold applies to the total, so a complete brief in a color I don't stock passes. The two tie exactly, so which one ranks first depends on row order in the CSV. This is the one top-1 miss and the one recall miss in the reached-a-model subset. It isn't fixed.

**Counting: contours counted 37 pearls where there are 14.** On `08_low_contrast_pearls.png`, Hough found 14 and contour found 37. In the debug image, every contour is a crescent around the shaded side of a bead. The lit side is as bright as the background, so the threshold misses it. A crescent fails the circularity check and goes through area splitting. The widest circle that fits a crescent is small, so each one splits into about 2.6 beads (37 / 14). With an agreement of 0.38, `CV_METHODS_DISAGREE` fired and the count stayed out of the estimate. It isn't fixed.

The replay also includes two fixes outside these three cases. The material normalization fixed "gold brass" and "white freshwater pearls". The earring length range fixed c07 ("4cm drop"), which had been flagged `OUT_OF_RANGE` in 3 of 3 runs because every product shared the 5 to 120cm range. That was the only false positive, and fixing it moved flag precision from 80% to 100%.

## What I'd change next

1. **Rerun the eval with a quota that lasts.** That means paid credits, or Claude on Bedrock. Photo-case extraction and the vision model's count are unmeasured, and every other number comes from 16 cases.
2. **Get data I didn't produce.** Twenty or thirty real briefs with a sales rep's spec, and phone photos with hand counts. Until then, every number here measures the system against its own assumptions.
3. **Close the review gaps the eval exposed.** Flag a stated color with no catalog match (t04). Compare the length-based count with the photo count. Flag missing quote fields such as length and string.
4. **Move the wrist conversion out of the prompt.** The prompt line "wrist or ankle measurement plus 2cm" made the model turn a stated 25cm anklet into 27cm in 1 of 3 runs (c06). I'd extract wrist size as its own field and add the 2cm in code, where a test can pin it down.
5. **Revisit contour counting on low contrast, but only once real photos exist.** Tuning it on synthetic pearls would just fit my generator.
6. **Set the matching weights with the business,** using real quotes and `--rescore` to compare.

## How I used AI

I built this with Claude Code (Claude Opus 5.5). It wrote the code, tests, synthetic catalog and synthetic photos, and I made the calls along the way. I set the spec fields and accepted Claude's proposed additions (bead shape, quantity, pattern, extras). I picked OpenRouter, and when gpt-oss-120b turned out to have no free tier and the free Gemma models were rate-limited, I chose the two free models above. I chose a catalog of finished products over a loose-bead inventory. With no real photos, I approved synthetic ones, and after the first CV results I approved blob splitting. With a same-day deadline and the quota gone, I chose to replay the fixes instead of rerunning the LLM, and I approved installing Docker in WSL to test the image.

The labels and the system came from the same session, so this eval can't catch a wrong assumption the two share. That's why real data is second on the list above. At least one bug came from an AI-written change: the wrist prompt line behind c06.
