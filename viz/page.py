"""HTML and markdown for one run: the jlens slice page (optionally with the translation box),
its iframe, a text summary, the no-refit-lens note, and the table of a question's runs. The only viz module that produces HTML (and, through jlens,
touches the network); it also reads the box's static files (viz/static/)."""

import html
import json
from importlib.resources import files

from jlens.vis import SliceData, build_page

from evaluation.data_loading import TwoHopItem
from evaluation.runner import field_answers
from viz.cache import RunKey, Shown
from viz.run import RunResult

GLOSS_DATA_ID = "jb-gloss-data"   # id of the JSON <script> that viz/static/gloss.js reads


def slice_page(slice_data: SliceData, result: RunResult, title: str) -> str:
    """Self-contained jlens heatmap page (d3 inlined) for the run; `title` names the model."""
    # TODO(debt): build_page(mode="embed") fetches d3 from a CDN once per process; vendor it if offline use is needed
    description = (
        f"{title} under condition '{result.condition}' ({result.field}). Positions cover the "
        f"prompt and the response up to the answer token (the one scoring compares against); "
        f"rows at ablated layers show the post-edit residual."
    )
    page, _, _ = build_page(
        slice_data, "".join(slice_data.context_token_strs),
        title=title, description=description, mode="embed",
    )
    return page


def with_glosses(page: str, payload: dict) -> str:
    """`page` with the translation box (gloss.css, the payload as JSON, gloss.js) inserted
    before its last `</body>`. `<` in the JSON is escaped, so no dictionary text can close the tag."""
    end = page.rfind("</body>")
    if end < 0:
        raise ValueError("page has no </body> to insert the translation box before")
    return page[:end] + _gloss_tags(payload) + page[end:]


def _gloss_tags(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return (f"<style>{_static('gloss.css')}</style>"
            f'<script type="application/json" id="{GLOSS_DATA_ID}">{data}</script>'
            f"<script>{_static('gloss.js')}</script>")


def _static(name: str) -> str:
    return (files("viz") / "static" / name).read_text(encoding="utf-8")


def iframe(page: str, height: int) -> str:
    """`page` as a srcdoc iframe (its own document, so its scripts run and styles stay isolated)."""
    return f'<iframe srcdoc="{html.escape(page)}" style="width:100%;height:{height}px;border:0"></iframe>'


def no_refit_lens_note(label: str, command: str) -> str:
    """HTML for the refit-lens pane of a model with no refit lens: the command that fits one."""
    return (f"<div style=\"padding:1em\"><p>No refit lens for <b>{html.escape(label)}</b>. "
            f"Fit one, then restart the viewer. As written the command uses the config's default "
            f"prompt count and batch size; commands.md section 7 has the settings and timing used "
            f"for the other lenses:</p><pre>{html.escape(command)}</pre></div>")


def summary(result: RunResult, mismatches: list[int]) -> str:
    """Short markdown: prompt, bridge, answers, full response, score, shown positions, replay check."""
    n_prompt = len(result.prompt_ids)
    verdict = "correct" if result.correct else "incorrect"
    return "\n".join([
        f"**Prompt** ({result.field}): {result.prompt}",
        f"**Bridge:** {bridge_label(result.item)} | **Expected:** {', '.join(result.answers)}",
        f"**Response** ({result.condition}): `{one_line(result.response_text)}` | **{verdict}**",
        _shown_line(result),
        _replay_line(mismatches, n_prompt),
    ])


def responses_table(rows: list[tuple[RunKey, Shown]]) -> str:
    """Markdown table of one question's cached runs (`ResultCache.entries`), in run order:
    model, prompt field, ablation, response, correct. "" when there are none."""
    if not rows:
        return ""
    header = ["| Model | Prompt field | Ablation | Response | Correct |", "|---|---|---|---|---|"]
    return "\n".join(header + [_table_row(key, shown) for key, shown in rows])


def _table_row(key: RunKey, shown: Shown) -> str:
    cells = [key.model, key.field, key.condition, shown.response, "yes" if shown.correct else "no"]
    return "| " + " | ".join(_table_cell(c) for c in cells) + " |"


def _table_cell(text: str) -> str:
    """`text` on one line, HTML-escaped, with pipes escaped, so it cannot break the table."""
    return html.escape(one_line(text).replace("\r", "")).replace("|", "\\|")


def one_line(text: str) -> str:
    """`text` with newlines shown as ⏎ (as the jlens page shows them), so it stays one markdown line."""
    return text.replace("\n", "⏎")


def item_card(item: TwoHopItem, field: str) -> str:
    """Short markdown for a chosen example before it is run: the field's prompt, bridge, answers."""
    swap = "" if item.swap_to is None else f" | **Swap to:** {item.swap_to}"
    return "\n".join([
        f"**{item.name}** ({item.category})",
        f"**Prompt** ({field}): {getattr(item, field)}",
        f"**Bridge:** {bridge_label(item)}{swap} | **Expected:** {', '.join(field_answers(item, field))}",
    ])


def bridge_label(item: TwoHopItem) -> str:
    """The bridge with its Chinese spellings, e.g. "France (法国, 法兰西)"; just "France" when it has none."""
    if not item.bridge_zh:
        return item.bridge
    return f"{item.bridge} ({', '.join(item.bridge_zh)})"


def _shown_line(result: RunResult) -> str:
    """Which heatmap positions hold the response, and which of them is the answer token."""
    n_prompt, i = len(result.prompt_ids), result.answer_index
    return (f"Shown response tokens are positions {n_prompt}-{n_prompt + i}; position "
            f"{n_prompt + i} is the answer token (response token {i + 1} of {len(result.response_ids)}).")


def _replay_line(mismatches: list[int], n_prompt: int) -> str:
    if not mismatches:
        return "Readout reproduces generation at every position."
    positions = ", ".join(str(n_prompt + i) for i in mismatches)
    return f"Readout top-1 differs from the generated token at response positions {positions}."
