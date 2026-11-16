"""HTML and markdown for one run: the jlens slice page, its iframe, and a text summary.
The only viz module that produces HTML (and, through jlens, touches the network)."""

import html

from jlens.vis import SliceData, build_page

from evaluation.data_loading import TwoHopItem
from evaluation.runner import field_answers
from viz.run import RunResult


def slice_page(slice_data: SliceData, result: RunResult, title: str) -> str:
    """Self-contained jlens heatmap page (d3 inlined) for the run; `title` names the model."""
    # TODO(debt): build_page(mode="embed") fetches d3 from a CDN once per process; vendor it if offline use is needed
    description = (
        f"{title} under condition '{result.condition}' ({result.field}). Positions cover the "
        f"prompt and the generated response; rows at ablated layers show the post-edit residual."
    )
    page, _, _ = build_page(
        slice_data, result.prompt + result.response_text,
        title=title, description=description, mode="embed",
    )
    return page


def iframe(page: str, height: int) -> str:
    """`page` as a srcdoc iframe (its own document, so its scripts run and styles stay isolated)."""
    return f'<iframe srcdoc="{html.escape(page)}" style="width:100%;height:{height}px;border:0"></iframe>'


def summary(result: RunResult, mismatches: list[int]) -> str:
    """Short markdown: prompt, bridge, answers, response, score, response positions, replay check."""
    n_prompt = len(result.prompt_ids)
    last = n_prompt + len(result.response_ids) - 1
    verdict = "correct" if result.correct else "incorrect"
    return "\n".join([
        f"**Prompt** ({result.field}): {result.prompt}",
        f"**Bridge:** {result.item.bridge} | **Expected:** {', '.join(result.answers)}",
        f"**Response** ({result.condition}): `{one_line(result.response_text)}` | **{verdict}**",
        f"Response tokens are positions {n_prompt}-{last}.",
        _replay_line(mismatches, n_prompt),
    ])


def one_line(text: str) -> str:
    """`text` with newlines shown as ⏎ (as the jlens page shows them), so it stays one markdown line."""
    return text.replace("\n", "⏎")


def item_card(item: TwoHopItem, field: str) -> str:
    """Short markdown for a chosen example before it is run: the field's prompt, bridge, answers."""
    swap = "" if item.swap_to is None else f" | **Swap to:** {item.swap_to}"
    return "\n".join([
        f"**{item.name}** ({item.category})",
        f"**Prompt** ({field}): {getattr(item, field)}",
        f"**Bridge:** {item.bridge}{swap} | **Expected:** {', '.join(field_answers(item, field))}",
    ])


def _replay_line(mismatches: list[int], n_prompt: int) -> str:
    if not mismatches:
        return "Readout reproduces generation at every position."
    positions = ", ".join(str(n_prompt + i) for i in mismatches)
    return f"Readout top-1 differs from the generated token at response positions {positions}."
