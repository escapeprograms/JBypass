"""Gradio wiring for the J-space viewer: controls, handlers and the event chains between them.
No model or tensor logic here; handlers call viz.catalog / run / slice / gloss / page, viz.session
and viz.cache."""

import random
from dataclasses import dataclass
from functools import partial

import gradio as gr
from jlens.vis import SliceData
from transformers import PreTrainedTokenizerBase

from evaluation.config import EvalConfig
from evaluation.data_loading import TwoHopItem
from model.config import RunConfig
from viz.cache import ResultCache, RunKey, Shown
from viz.catalog import (ALL_SPLITS, ModelChoice, buildable_conditions, examples, fit_command, pick_random,
                         refit_lens_file, split_names, valid_fields)
from viz.cedict import Cedict
from viz.config import VizConfig
from viz.gloss import gloss_payload
from viz.page import (iframe, item_card, no_refit_lens_note, responses_table, slice_page, summary,
                      with_glosses)
from viz.run import RunResult, run_example
from viz.session import Session
from viz.slice import replay_mismatches, run_slices

LENS_NAMES = ("original", "refit")   # page-title names of Loaded.readers, in order

# Run (the only event that touches ctx.session) and show share one queue slot, so two tabs never
# load or run at once, and a model flip during a Run is shown after it.
MODEL_QUEUE = dict(concurrency_id="model", concurrency_limit=1)


@dataclass
class AppContext:
    run: RunConfig
    viz: VizConfig
    eval_cfg: EvalConfig
    session: Session                        # the only holder of the loaded model
    tokenizer: PreTrainedTokenizerBase      # the session profile's (question list without a model)
    has_stats: bool                         # readout statistics exist for the session profile
    choices: dict[str, ModelChoice]         # dropdown label -> choice
    default_model: str                      # a key of `choices`
    rng: random.Random                      # for the Random example button
    dictionary: Cedict | None               # CC-CEDICT for the translation box; None = off
    cache: ResultCache                      # rendered results of the selected question, per model


@dataclass
class _ResultArea:
    """The widgets on_show and on_run fill, in `_outputs` order."""
    table: gr.Markdown                      # every cached run of the selected question
    summary: gr.Markdown
    original: gr.HTML                       # heatmap under the original (prefit) lens
    own: gr.HTML                            # heatmap under the model's refit lens, or how to fit one

    def components(self) -> list:
        return [self.table, self.summary, self.original, self.own]


@dataclass
class _Controls:
    dataset: gr.Dropdown
    split: gr.Dropdown
    example: gr.Dropdown
    random: gr.Button
    field: gr.Dropdown
    condition: gr.Dropdown
    card: gr.Markdown
    model: gr.Dropdown
    run: gr.Button
    result: _ResultArea
    items: gr.State                         # {item name: TwoHopItem} of the current dataset/split


# ---- handlers -------------------------------------------------------------------------------

def on_dataset(ctx: AppContext, dataset: str) -> gr.Dropdown:
    """Splits that contain at least one readable item of the dataset; resets to ALL_SPLITS."""
    items = _items(ctx, dataset, ALL_SPLITS)
    return gr.Dropdown(choices=split_names(items, ctx.viz.split_file), value=ALL_SPLITS)


def on_split(ctx: AppContext, dataset: str, split: str) -> tuple[dict, gr.Dropdown]:
    """The examples of the split, kept in session state, and the example dropdown (first selected)."""
    by_name = {it.name: it for it in _items(ctx, dataset, split)}
    names = list(by_name)
    return by_name, gr.Dropdown(choices=names, value=names[0])


def on_example(ctx: AppContext, items: dict, name: str, field: str,
               condition: str) -> tuple[gr.Dropdown, gr.Dropdown]:
    """Only the prompt fields and conditions the item (and the profile's readout stats) allow;
    keeps the current ones if still allowed."""
    item = items[name]
    fields = valid_fields(item, ctx.eval_cfg.prompt_fields)
    conditions = buildable_conditions(ctx.has_stats, item, ctx.eval_cfg.conditions)
    return (gr.Dropdown(choices=fields, value=_keep(field, fields)),
            gr.Dropdown(choices=conditions, value=_keep(condition, conditions)))


def on_card(items: dict, name: str, field: str) -> str:
    return item_card(items[name], field)


def on_random(ctx: AppContext, items: dict) -> gr.Dropdown:
    return gr.Dropdown(value=pick_random(list(items.values()), ctx.rng).name)


def on_show(ctx: AppContext, label: str, dataset: str, name: str | None, field: str | None,
            condition: str | None) -> tuple[str, ...]:
    """The result widgets' values stored for this model and question, or a placeholder; a cache
    lookup only, so it never loads a model."""
    key = RunKey(dataset, name, field, condition, label)
    return _outputs(ctx, key, ctx.cache.get(key))


def on_run(ctx: AppContext, label: str, dataset: str, items: dict, name: str | None, field: str | None,
           condition: str | None) -> tuple[str, ...]:
    """The result widgets' values for this model and question: from the cache, else computed
    and stored."""
    if not (name and field and condition):
        raise gr.Error("Pick an example, a prompt field and an ablation first.")
    key = RunKey(dataset, name, field, condition, label)
    shown = ctx.cache.get(key)
    if shown is None:
        shown = _compute(ctx, key, items[name])
        ctx.cache.put(key, shown)
    return _outputs(ctx, key, shown)


def _compute(ctx: AppContext, key: RunKey, item: TwoHopItem) -> Shown:
    """Load the model (freeing the previous one), generate under the condition with the original
    lens, read out every layer through each of the model's lenses and render one page per lens."""
    choice = ctx.choices[key.model]
    loaded = ctx.session.get(choice)
    result = run_example(loaded.js, item, key.field, key.condition, ctx.eval_cfg.seed,
                         ctx.run.generation.max_new_tokens)
    slices = run_slices(loaded.js, loaded.readers, result, ctx.viz.top_n, ctx.viz.max_tracked)
    panes = [_pane(ctx, s, result, _title(choice, key, lens_name, reader))
             for lens_name, reader, s in zip(LENS_NAMES, loaded.readers, slices)]
    own = panes[1] if len(panes) > 1 else _no_lens_pane(ctx, choice)
    mismatches = replay_mismatches(slices[0], result)   # checked on the original lens's slice
    return Shown(summary=f"**Model:** {choice.label}\n{summary(result, mismatches)}",
                 original=panes[0], own=own, response=result.response_text, correct=result.correct)


def _title(choice: ModelChoice, key: RunKey, lens_name: str, reader) -> str:
    return f"{choice.label}: {key.name} | {lens_name} lens ({reader.lens.n_prompts} prompts)"


def _no_lens_pane(ctx: AppContext, choice: ModelChoice) -> str:
    """The refit pane of a model with no refit lens: the command that fits one."""
    return no_refit_lens_note(choice.label, fit_command(choice, refit_lens_file(choice, ctx.viz.lenses_dir)))


def _outputs(ctx: AppContext, key: RunKey, shown: Shown | None) -> tuple[str, ...]:
    """The result widgets' values, in `_ResultArea.components` order: the table of the question's
    cached runs, then `shown`, or a placeholder and empty panes when it is None."""
    table = responses_table(ctx.cache.entries(key.question))
    if shown is None:
        return table, f"Not run yet for **{key.model}** on this question. Press Run.", "", ""
    return table, shown.summary, shown.original, shown.own


def _pane(ctx: AppContext, slice_data: SliceData, result: RunResult, title: str) -> str:
    """One lens's jlens page (with its own translation payload) as an iframe."""
    return iframe(_page(ctx, slice_data, result, title), ctx.viz.page_height)


def _page(ctx: AppContext, slice_data: SliceData, result: RunResult, title: str) -> str:
    """The jlens page, with the click-to-translate box when a dictionary is loaded."""
    page = slice_page(slice_data, result, title=title)
    if ctx.dictionary is None:
        return page
    return with_glosses(page, gloss_payload(slice_data, ctx.dictionary))


def _items(ctx: AppContext, dataset: str, split: str) -> list[TwoHopItem]:
    return examples(ctx.tokenizer, dataset, split, ctx.eval_cfg, ctx.viz.split_file)


def _keep(current: str | None, options: list[str]) -> str:
    return current if current in options else options[0]


# ---- layout and wiring ----------------------------------------------------------------------

def build_app(ctx: AppContext) -> gr.Blocks:
    """The viewer. Opens without loading a model: the question list needs only the tokenizer."""
    with gr.Blocks(title="J-space viewer") as demo:
        ui = _controls(ctx)
        _wire(ctx, demo, ui)
    return demo


def _controls(ctx: AppContext) -> _Controls:
    """Create the components top to bottom: the question, its card, the model, then the result.
    Choice lists are filled by the handlers on page open."""
    dataset, split = _source_row(ctx)
    example, random_button = _example_row()
    field, condition = _prompt_row()
    card = gr.Markdown(line_breaks=True)
    model, run_button = _model_row(ctx)
    return _Controls(dataset, split, example, random_button, field, condition, card, model, run_button,
                     _result_area(), gr.State({}))


def _source_row(ctx: AppContext) -> tuple[gr.Dropdown, gr.Dropdown]:
    datasets = list(ctx.eval_cfg.datasets)
    with gr.Row():
        dataset = gr.Dropdown(datasets, value=datasets[0], label="Dataset")
        split = gr.Dropdown([ALL_SPLITS], value=ALL_SPLITS, label="Split")
    return dataset, split


def _example_row() -> tuple[gr.Dropdown, gr.Button]:
    with gr.Row():
        example = gr.Dropdown([], label="Example", scale=3)
        random_button = gr.Button("Random example", scale=1)
    return example, random_button


def _prompt_row() -> tuple[gr.Dropdown, gr.Dropdown]:
    with gr.Row():
        field = gr.Dropdown([], label="Prompt field")
        condition = gr.Dropdown([], label="Ablation")
    return field, condition


def _model_row(ctx: AppContext) -> tuple[gr.Dropdown, gr.Button]:
    with gr.Row():
        model = gr.Dropdown(list(ctx.choices), value=ctx.default_model, label="Model", scale=3)
        run_button = gr.Button("Run", variant="primary", scale=1)
    return model, run_button


def _result_area() -> _ResultArea:
    """The result widgets: the table of the question's runs, the run summary, then one tab per
    lens. The one place that knows how the result is laid out."""
    table = gr.Markdown()
    run_summary = gr.Markdown(line_breaks=True)
    with gr.Tabs():
        with gr.Tab("Original lens"):
            original = _page_html()
        with gr.Tab("Refit lens (this model)"):
            own = _page_html()
    return _ResultArea(table, run_summary, original, own)


def _page_html() -> gr.HTML:
    """A pane for `page.iframe` output, rendered raw (see README: Embedding)."""
    return gr.HTML(js_on_load=None, apply_default_css=False)


def _steps(ctx: AppContext, ui: _Controls) -> list[dict]:
    """The refresh chain in order, as event kwargs; a change re-runs its own step and every later
    one. `api_name` names the step (Gradio suffixes repeats: split, split_1, ...)."""
    return [
        dict(api_name="dataset", fn=partial(on_dataset, ctx), inputs=[ui.dataset], outputs=[ui.split]),
        dict(api_name="split", fn=partial(on_split, ctx), inputs=[ui.dataset, ui.split],
             outputs=[ui.items, ui.example]),
        dict(api_name="example", fn=partial(on_example, ctx),
             inputs=[ui.items, ui.example, ui.field, ui.condition], outputs=[ui.field, ui.condition]),
        dict(api_name="card", fn=on_card, inputs=[ui.items, ui.example, ui.field], outputs=[ui.card]),
        dict(api_name="show", fn=partial(on_show, ctx),
             inputs=[ui.model, ui.dataset, ui.example, ui.field, ui.condition],
             outputs=ui.result.components(), **MODEL_QUEUE),
    ]


def _wire(ctx: AppContext, demo: gr.Blocks, ui: _Controls) -> None:
    """User edits (`.input`, never `.change`) start the chain at their step; later steps follow
    with `.success` (explicit chaining that stops at the first error). Every chain ends in `show`,
    so a new question never leaves an old result on screen."""
    steps = _steps(ctx, ui)
    _cascade([demo.load, ui.dataset.input], steps, "dataset")
    _cascade([ui.split.input], steps, "split")
    _cascade([ui.example.input], steps, "example")
    _cascade([ui.field.input], steps, "card")
    _cascade([ui.condition.input, ui.model.input], steps, "show")
    picked = ui.random.click(partial(on_random, ctx), [ui.items], [ui.example], api_name="random")
    _chain(picked, _tail(steps, "example"))
    ui.run.click(partial(on_run, ctx), [ui.model, ui.dataset, ui.items, ui.example, ui.field, ui.condition],
                 ui.result.components(), api_name="run", **MODEL_QUEUE)


def _cascade(triggers: list, steps: list[dict], start: str) -> None:
    first, *rest = _tail(steps, start)
    _chain(gr.on(triggers=triggers, **first), rest)


def _tail(steps: list[dict], start: str) -> list[dict]:
    names = [step["api_name"] for step in steps]
    return steps[names.index(start):]


def _chain(event, steps: list[dict]) -> None:
    for kwargs in steps:
        event = event.success(**kwargs)
