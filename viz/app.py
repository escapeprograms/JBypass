"""Gradio wiring for the J-space viewer: controls, handlers and the event chains between them.
No model or tensor logic here; handlers call viz.catalog / run / slice / gloss / page and viz.session."""

import random
from dataclasses import dataclass
from functools import partial

import gradio as gr
from jlens.vis import SliceData

from evaluation.config import EvalConfig
from evaluation.data_loading import TwoHopItem
from model.config import RunConfig
from viz.catalog import (ALL_SPLITS, ModelChoice, examples, pick_random, split_names, valid_conditions,
                         valid_fields)
from viz.cedict import Cedict
from viz.config import VizConfig
from viz.gloss import gloss_payload
from viz.page import iframe, item_card, slice_page, summary, with_glosses
from viz.run import RunResult, run_example
from viz.session import Session
from viz.slice import replay_mismatches, run_slice

# Every event that touches ctx.session shares one queue slot, so two tabs never load or run at once.
MODEL_QUEUE = dict(concurrency_id="model", concurrency_limit=1)


@dataclass
class AppContext:
    run: RunConfig
    viz: VizConfig
    eval_cfg: EvalConfig
    session: Session                        # the only holder of the loaded model
    choices: dict[str, ModelChoice]         # dropdown label -> choice
    default_model: str                      # a key of `choices`
    rng: random.Random                      # for the Random example button
    dictionary: Cedict | None               # CC-CEDICT for the translation box; None = off


@dataclass
class _Controls:
    model: gr.Dropdown
    dataset: gr.Dropdown
    split: gr.Dropdown
    example: gr.Dropdown
    random: gr.Button
    field: gr.Dropdown
    condition: gr.Dropdown
    run: gr.Button
    card: gr.Markdown
    summary: gr.Markdown
    page: gr.HTML
    items: gr.State                         # {item name: TwoHopItem} of the current model/dataset/split


# ---- handlers -------------------------------------------------------------------------------

def on_model(ctx: AppContext, label: str) -> tuple[str, str]:
    """Load the chosen model (dropping the old one); clear the previous run."""
    ctx.session.get(ctx.choices[label])
    return f"Loaded **{label}**.", ""


def on_dataset(ctx: AppContext, label: str, dataset: str) -> gr.Dropdown:
    """Splits that contain at least one readable item of the dataset; resets to ALL_SPLITS."""
    items = _items(ctx, label, dataset, ALL_SPLITS)
    return gr.Dropdown(choices=split_names(items, ctx.viz.split_file), value=ALL_SPLITS)


def on_split(ctx: AppContext, label: str, dataset: str, split: str) -> tuple[dict, gr.Dropdown]:
    """The examples of the split, kept in session state, and the example dropdown (first selected)."""
    by_name = {it.name: it for it in _items(ctx, label, dataset, split)}
    names = list(by_name)
    return by_name, gr.Dropdown(choices=names, value=names[0])


def on_example(ctx: AppContext, items: dict, name: str, field: str,
               condition: str) -> tuple[gr.Dropdown, gr.Dropdown]:
    """Only the prompt fields and conditions valid for the item; keeps the current ones if valid."""
    item = items[name]
    fields = valid_fields(item, ctx.eval_cfg.prompt_fields)
    conditions = valid_conditions(item, ctx.eval_cfg.conditions)
    return (gr.Dropdown(choices=fields, value=_keep(field, fields)),
            gr.Dropdown(choices=conditions, value=_keep(condition, conditions)))


def on_card(items: dict, name: str, field: str) -> str:
    return item_card(items[name], field)


def on_random(ctx: AppContext, items: dict) -> gr.Dropdown:
    return gr.Dropdown(value=pick_random(list(items.values()), ctx.rng).name)


def on_run(ctx: AppContext, label: str, items: dict, name: str | None, field: str | None,
           condition: str | None) -> tuple[str, str]:
    """Generate under the condition, read out every layer, return (summary, jlens page iframe)."""
    if not (name and field and condition):
        raise gr.Error("Pick an example, a prompt field and an ablation first.")
    choice = ctx.choices[label]
    js = ctx.session.get(choice)
    result = run_example(js, items[name], field, condition, ctx.eval_cfg.seed,
                         ctx.run.generation.max_new_tokens)
    slice_data = run_slice(js, result, ctx.viz.top_n, ctx.viz.max_tracked)
    mismatches = replay_mismatches(slice_data, result)
    page = _page(ctx, slice_data, result, title=f"{choice.label}: {name}")
    return summary(result, mismatches), iframe(page, ctx.viz.page_height)


def _page(ctx: AppContext, slice_data: SliceData, result: RunResult, title: str) -> str:
    """The jlens page, with the click-to-translate box when a dictionary is loaded."""
    page = slice_page(slice_data, result, title=title)
    if ctx.dictionary is None:
        return page
    return with_glosses(page, gloss_payload(slice_data, ctx.dictionary))


def _items(ctx: AppContext, label: str, dataset: str, split: str) -> list[TwoHopItem]:
    js = ctx.session.get(ctx.choices[label])
    return examples(js, dataset, split, ctx.eval_cfg, ctx.viz.split_file)


def _keep(current: str | None, options: list[str]) -> str:
    return current if current in options else options[0]


# ---- layout and wiring ----------------------------------------------------------------------

def build_app(ctx: AppContext) -> gr.Blocks:
    """The viewer. The model loads on page open (the example list needs its tokenizer)."""
    with gr.Blocks(title="J-space viewer") as demo:
        ui = _controls(ctx)
        _wire(ctx, demo, ui)
    return demo


def _controls(ctx: AppContext) -> _Controls:
    """Create the components; choice lists are filled by the handlers once the model is loaded."""
    datasets = list(ctx.eval_cfg.datasets)
    with gr.Row():
        model = gr.Dropdown(list(ctx.choices), value=ctx.default_model, label="Model")
        dataset = gr.Dropdown(datasets, value=datasets[0], label="Dataset")
        split = gr.Dropdown([ALL_SPLITS], value=ALL_SPLITS, label="Split")
    with gr.Row():
        example = gr.Dropdown([], label="Example", scale=3)
        random_button = gr.Button("Random example", scale=1)
    with gr.Row():
        field = gr.Dropdown([], label="Prompt field")
        condition = gr.Dropdown([], label="Ablation")
        run_button = gr.Button("Run", variant="primary")
    card, run_summary = gr.Markdown(line_breaks=True), gr.Markdown(line_breaks=True)
    page = gr.HTML(js_on_load=None, apply_default_css=False)
    return _Controls(model, dataset, split, example, random_button, field, condition, run_button,
                     card, run_summary, page, gr.State({}))


def _steps(ctx: AppContext, ui: _Controls) -> list[dict]:
    """The refresh chain in order, as event kwargs; a change re-runs its own step and every later
    one. `api_name` names the step (Gradio suffixes repeats: split, split_1, ...)."""
    return [
        dict(api_name="model", fn=partial(on_model, ctx), inputs=[ui.model],
             outputs=[ui.summary, ui.page], **MODEL_QUEUE),
        dict(api_name="dataset", fn=partial(on_dataset, ctx), inputs=[ui.model, ui.dataset],
             outputs=[ui.split], **MODEL_QUEUE),
        dict(api_name="split", fn=partial(on_split, ctx), inputs=[ui.model, ui.dataset, ui.split],
             outputs=[ui.items, ui.example], **MODEL_QUEUE),
        dict(api_name="example", fn=partial(on_example, ctx),
             inputs=[ui.items, ui.example, ui.field, ui.condition], outputs=[ui.field, ui.condition]),
        dict(api_name="card", fn=on_card, inputs=[ui.items, ui.example, ui.field], outputs=[ui.card]),
    ]


def _wire(ctx: AppContext, demo: gr.Blocks, ui: _Controls) -> None:
    """User edits (`.input`, never `.change`) start the chain at their step; later steps follow
    with `.success` (explicit chaining that stops at the first error)."""
    steps = _steps(ctx, ui)
    _cascade([demo.load, ui.model.input], steps, "model")
    _cascade([ui.dataset.input], steps, "dataset")
    _cascade([ui.split.input], steps, "split")
    _cascade([ui.example.input], steps, "example")
    _cascade([ui.field.input], steps, "card")
    picked = ui.random.click(partial(on_random, ctx), [ui.items], [ui.example], api_name="random")
    _chain(picked, _tail(steps, "example"))
    ui.run.click(partial(on_run, ctx), [ui.model, ui.items, ui.example, ui.field, ui.condition],
                 [ui.summary, ui.page], api_name="run", **MODEL_QUEUE)


def _cascade(triggers: list, steps: list[dict], start: str) -> None:
    first, *rest = _tail(steps, start)
    _chain(gr.on(triggers=triggers, **first), rest)


def _tail(steps: list[dict], start: str) -> list[dict]:
    names = [step["api_name"] for step in steps]
    return steps[names.index(start):]


def _chain(event, steps: list[dict]) -> None:
    for kwargs in steps:
        event = event.success(**kwargs)
