"""
utils.py — Penguins Dataset Concept Check
Contains all quiz data and Gradio demo-building logic.
Import this module from the notebook and call build_question(n) or launch_all().
All Gradio console output (startup banner, local URL, API links, etc.) is suppressed.
Compatible with both Gradio 5.x (show_api) and Gradio 6.x (footer_links).
"""

import io
import inspect
import contextlib
import gradio as gr

QUESTIONS = [
    {
        "question": "Q1. `penguins.info()` shows `bill_length_mm` has 342 non-null values "
                     "while `species` has 344. What does calling `dropna()` on the WHOLE "
                     "DataFrame do here?",
        "choices": [
            "A) Removes exactly the 2 rows missing bill_length_mm",
            "B) Removes any row missing a value in ANY column, possibly more than 2 rows",
            "C) Fills the 2 missing values with the column mean",
            "D) Does nothing unless a subset is specified",
        ],
        "answer": "B) Removes any row missing a value in ANY column, possibly more than 2 rows",
        "explanation": "dropna() with no arguments drops a row if it has a NaN in ANY "
                        "column. Since sex often has more missing values than the bill "
                        "columns, a blanket dropna() usually removes more rows than students "
                        "expect.",
    },
    {
        "question": "Q2. You want average body_mass_g per island. Which is the best approach?",
        "choices": [
            "A) Filter the DataFrame 3 times (once per island) and average manually",
            "B) penguins.groupby('island')['body_mass_g'].mean()",
            "C) penguins['body_mass_g'].mean() run 3 times",
            "D) Sort by island, then eyeball the values",
        ],
        "answer": "B) penguins.groupby('island')['body_mass_g'].mean()",
        "explanation": "groupby() is a single vectorized operation: concise, fast, and it "
                        "scales automatically if new islands appear in the data. Manual "
                        "filtering is repetitive and error-prone.",
    },
    {
        "question": "Q3. The DataFrame's index has gaps after filtering. What does "
                     "`penguins.loc[0:5]` do differently from `penguins.iloc[0:5]`?",
        "choices": [
            "A) They always return identical rows",
            "B) loc uses label-based lookup (may skip rows or error); iloc uses position "
               "and always returns the first 6 rows",
            "C) iloc uses label-based lookup; loc uses position",
            "D) loc only works on columns, not rows",
        ],
        "answer": "B) loc uses label-based lookup (may skip rows or error); iloc uses position "
                   "and always returns the first 6 rows",
        "explanation": "loc looks for index LABELS 0 through 5 inclusive, so gaps or "
                        "shuffled indices change the result. iloc is purely positional "
                        "and ignores the index labels entirely.",
    },
    {
        "question": "Q4. Only the `sex` column has missing values. Why might "
                     "`dropna(subset=['sex'])` be better than plain `dropna()`?",
        "choices": [
            "A) It's not better, they do the same thing",
            "B) It only removes rows missing sex, preserving rows complete in "
               "every other column",
            "C) It fills missing sex values automatically",
            "D) It deletes the sex column entirely",
        ],
        "answer": "B) It only removes rows missing sex, preserving rows complete in "
                   "every other column",
        "explanation": "subset=['sex'] narrows the check to just that column, so you "
                        "don't lose otherwise-complete rows because of unrelated missing "
                        "values elsewhere.",
    },
    {
        "question": "Q5. You set bins=200 on a histogram of 342 penguins. What happens?",
        "choices": [
            "A) A smooth, clean distribution appears",
            "B) A noisy, spiky plot with mostly empty or single-point bins",
            "C) Seaborn automatically caps bins at 30",
            "D) An error is raised",
        ],
        "answer": "B) A noisy, spiky plot with mostly empty or single-point bins",
        "explanation": "With 200 bins for 342 points, most bins get 0 or 1 observations, "
                        "obscuring the real distribution shape instead of revealing it.",
    },
    {
        "question": "Q6. In a flipper-length vs. body-mass scatterplot colored by species, "
                     "one species clusters clearly apart. In the real dataset, which species "
                     "is it, and what's the best way to confirm the cause?",
        "choices": [
            "A) Gentoo; confirm with groupby('species') means or facet by island/sex",
            "B) Adelie; confirm by deleting the other species",
            "C) Chinstrap; no further check is needed, it's always sex-based",
            "D) All three overlap completely, there's no separation",
        ],
        "answer": "A) Gentoo; confirm with groupby('species') means or facet by island/sex",
        "explanation": "Gentoo penguins are notably larger. Checking groupby('species') "
                        "means, or faceting by island/sex, helps rule out confounding "
                        "variables before attributing the gap purely to species.",
    },
    {
        "question": "Q7. A student says: 'I will just use body_mass_g.mean() for the whole "
                     "dataset to describe a typical penguin.' Why is this potentially misleading?",
        "choices": [
            "A) It isn't misleading, the overall mean is always representative",
            "B) The dataset mixes 3 species with different typical sizes, so the overall "
               "mean blends populations that aren't comparable",
            "C) mean() only works on integer columns, not floats",
            "D) body_mass_g cannot be averaged because it has missing values",
        ],
        "answer": "B) The dataset mixes 3 species with different typical sizes, so the overall "
                   "mean blends populations that aren't comparable",
        "explanation": "Gentoo penguins are notably heavier than Adelie and Chinstrap, so a "
                        "single overall mean doesn't represent any real species well. A more "
                        "honest summary is penguins.groupby('species')['body_mass_g'].mean().",
    },
    {
        "question": "Q8. To show a general audience that the three penguin species are "
                     "physically different, which plot is generally clearest, and why?",
        "choices": [
            "A) A box plot of a measurement grouped by species, since it shows medians "
               "and spread at a glance without explaining density",
            "B) A raw data table, since numbers are always clearer than charts",
            "C) A pie chart of species counts, since it shows physical differences directly",
            "D) A single histogram of all penguins combined, since more data is always better",
        ],
        "answer": "A) A box plot of a measurement grouped by species, since it shows medians "
                   "and spread at a glance without explaining density",
        "explanation": "A box plot grouped by species is usually clearest for a general, "
                        "non-technical audience because it shows medians and spread directly, "
                        "without requiring the viewer to understand density or bin width. "
                        "Scatterplots are better for showing relationships between two "
                        "measurements instead.",
    },
]


def _build_mc_block(q):
    """Builds one multiple-choice question inside an existing gr.Blocks context."""
    gr.Markdown(f"### {q['question']}")
    radio = gr.Radio(choices=q["choices"], label="Choose one answer")
    submit_btn = gr.Button("Submit Answer")
    result_box = gr.Markdown()

    def check(selected, q=q):
        if selected is None:
            return "Please select an answer first."
        if selected == q["answer"]:
            verdict = "✅ **Correct!**"
        else:
            verdict = f"❌ **Not quite.** Correct answer: {q['answer']}"
        return f"{verdict}\n\n**Explanation:** {q['explanation']}"

    submit_btn.click(fn=check, inputs=radio, outputs=result_box)


def _launch_kwargs(inline, share, height):
    """
    Builds a launch() kwargs dict compatible with the installed Gradio version.
    Gradio 5.x uses show_api=False; Gradio 6.x replaced it with footer_links.
    """
    kwargs = dict(inline=inline, share=share, height=height,
                   quiet=True, prevent_thread_lock=True)

    valid_params = set(inspect.signature(gr.Blocks.launch).parameters)

    if "footer_links" in valid_params:
        kwargs["footer_links"] = []
    elif "show_api" in valid_params:
        kwargs["show_api"] = False

    return kwargs


def _silent_launch(demo, inline, share, height):
    """Launches a Gradio Blocks app while suppressing all console output."""
    kwargs = _launch_kwargs(inline, share, height)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        demo.launch(**kwargs)


def build_question(n, height=300, inline=True, share=False):
    """
    Launches a single question's Gradio demo by number (1-8), fully silent.
    All questions are closed-form multiple choice.
    """
    if not (1 <= n <= len(QUESTIONS)):
        raise ValueError(f"Question number must be between 1 and {len(QUESTIONS)}")

    q = QUESTIONS[n - 1]
    with gr.Blocks() as demo:
        _build_mc_block(q)

    _silent_launch(demo, inline, share, height)


def launch_all(height=300, inline=True, share=False):
    """
    Launches all questions in a single combined Gradio app, fully silent.
    """
    with gr.Blocks(title="Penguins Dataset Concept Check") as demo:
        gr.Markdown(
            "# 🐧 Penguins Dataset Concept Check\n"
            "Answer each question, then click Submit to check yourself."
        )
        for q in QUESTIONS:
            with gr.Column(variant="panel"):
                _build_mc_block(q)

    _silent_launch(demo, inline, share, height)
