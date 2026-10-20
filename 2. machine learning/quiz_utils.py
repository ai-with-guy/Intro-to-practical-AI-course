"""Interactive concept checks for lessons 2.1–2.4."""

import inspect

import gradio as gr


QUESTIONS = [
    {
        "question": "You predict body_mass_g from flipper_length_mm. What are X and y?",
        "choices": [
            "A) X is body mass; y is flipper length",
            "B) X is flipper length; y is body mass",
            "C) X contains both columns; y is the row index",
            "D) X is the predicted mass; y is the model",
        ],
        "answer": "B",
        "explanation": "X contains the input features; y is the target we want to predict. "
        "In scikit-learn, penguins[['flipper_length_mm']] keeps X two-dimensional.",
    },
    {
        "question": "A body-mass regression model has a test MAE of 180 g. What does this mean?",
        "choices": [
            "A) Exactly 180 penguins were classified incorrectly",
            "B) Every prediction is exactly 180 g too high",
            "C) The model is correct 180% of the time",
            "D) The average error is 180 g in either direction",
        ],
        "answer": "D",
        "explanation": "MAE is just Mean Absolute Error meaning the avrage of the distance between target and prediction.",
    },
    {
        "question": "For binary logistic regression, how do predict(X) and predict_proba(X) differ?",
        "choices": [
            "A) predict returns class labels; predict_proba returns probabilities for each class",
            "B) predict returns probabilities; predict_proba returns class labels",
            "C) Both always return the same array",
            "D) predict_proba trains the model again",
        ],
        "answer": "A",
        "explanation": "predict returns a class for each row. predict_proba returns a column "
        "for each class, ordered by model.classes_. For classes [0, 1], [:, 1] selects "
        "the probability of class 1.",
    },
    {
        "question": "Why does logistic regression on the original x1 and x2 features struggle with XOR?",
        "choices": [
            "A) Logistic regression cannot accept two input features",
            "B) XOR is a regression problem rather than classification",
            "C) Opposite corners share a class, so one straight boundary cannot separate them",
            "D) Increasing the training iterations always solves XOR perfectly",
        ],
        "answer": "C",
        "explanation": "With the original features, logistic regression has a linear decision "
        "boundary. But XOR is very clearly nonlinear",
    },
    {
        "question": "A dataset has 950 class-0 and 50 class-1 examples. A model always predicts 0. "
        "What are its accuracy and class-1 recall?",
        "choices": [
            "A) 50% accuracy and 50% recall",
            "B) 95% accuracy and 0% recall",
            "C) 95% accuracy and 95% recall",
            "D) 100% accuracy and 0% recall",
        ],
        "answer": "B",
        "explanation": "The model gets all 950 majority-class examples right but misses all "
        "50 positives. High accuracy can hide a useless minority-class detector. Its "
        "balanced accuracy is (100% + 0%) / 2 = 50%.",
    },
    {
        "question": "What does LogisticRegression(class_weight='balanced') do?",
        "choices": [
            "A) Deletes majority-class examples until the classes are equal",
            "B) Guarantees perfect recall and accuracy on unseen data",
            "C) Changes the test set to contain equal class counts",
            "D) Gives rarer classes more weight in the training objective",
        ],
        "answer": "D",
        "explanation": "Balanced weights are inversely proportional to training class "
        "frequencies. They change the cost of mistakes, not the dataset's rows. This "
        "can improve minority recall, but may increase false positives; evaluate the tradeoff.",
    },
    {
        "question": "A confusion matrix has actual classes as rows and predicted classes as columns: "
        "[[90, 10], [30, 70]]. What is class-1 recall?",
        "choices": [
            "A) 70 / (70 + 10) = 87.5%",
            "B) (90 + 70) / 200 = 80%",
            "C) 70 / (70 + 30) = 70%",
            "D) 90 / (90 + 10) = 90%",
        ],
        "answer": "C",
        "explanation": "Recall asks how many actual positives were found: TP / (TP + FN). "
        "Here TP = 70 and FN = 30. The other calculations are precision, overall accuracy, "
        "and class-0 recall, respectively.",
    },
    {
        "question": "A decision tree scores 100% on training data and 72% on held-out test data. "
        "What is the best interpretation?",
        "choices": [
            "A) The gap suggests overfitting: training performance overstates generalization",
            "B) The model must also get 100% on future data",
            "C) The test labels should be used to fit the tree until it reaches 100%",
            "D) Real accuracy is probabably the avrage of the 2",
        ],
        "answer": "A",
        "explanation": "A deep tree can memorize training examples, including noise. "
        "Try limiting depth or increasing min_samples_leaf, using validation data to "
        "choose settings. Keep a final test set untouched for the final evaluation.",
    },
    {
        "question": "A model was trained on imbalanced data using a normal random train-test "
        "split without stratification. It scored 90% accuracy on our test set. "
        "Should we expect good results in the field?",
        "choices": [
            "A) Yes! 90% test accuracy is strong evidence that the model will perform well in the field",
            "B) No! An entire class might be almost completely missing from our test set, "
            "so this accuracy alone is not helpful for judging performance on that class",
            "C) Yes! A random split always gives every class enough test examples",
            "D) No! Models trained on imbalanced data can never make useful predictions",
        ],
        "answer": "B",
        "explanation": "A random split can leave very few or no minority-class examples "
        "in the test set, especially when that class is very rare or the test set is small. "
        "High overall accuracy then provides little or no evidence about that class, "
        "and field accuracy may differ if class proportions change. Check "
        "the class counts, confusion matrix, and per-class recall. Using "
        "stratify=data['class'] helps preserve class proportions, but you still need "
        "enough test examples from each class for a reliable evaluation.",
    },
    {
        "question": "A model was trained on data sampled from the field where it will be "
        "deployed. The data has substantial class imbalance, only four classes, and over "
        "one million examples. A normal random train-test split produced 90% test "
        "accuracy. Assuming field conditions and class proportions stay similar, "
        "should we expect similar overall accuracy in the field?",
        "choices": [
            "A) No! Class imbalance makes overall test accuracy meaningless, regardless of sample size",
            "B) Yes! With over a million examples, the model must perform well on every class",
            "C) Yes! Even if some classes are severely underrepresented in the test set, "
            "these rare classes contribute little to overall accuracy, so similar overall "
            "accuracy is reasonable to expect",
            "D) No! A random split is only useful when all classes have equal numbers of examples",
        ],
        "answer": "C",
        "explanation": "With a sufficiently large random test set from the deployment "
        "population, overall accuracy can be estimated reliably even with class imbalance. "
        "A class rare enough to be almost absent from that test set also has little weight "
        "in overall field accuracy, provided class proportions stay similar. This does "
        "not imply good performance on every class: inspect per-class recall and the "
        "confusion matrix too. The evaluation also assumes independent examples and no "
        "train-test leakage.",
    },
    {
        "question": "What is the main idea behind a random forest classifier?",
        "choices": [
            "A) Train one tree and randomly change its predictions",
            "B) Train trees sequentially so each corrects the previous tree's errors",
            "C) Guarantee that test accuracy is higher than any individual tree",
            "D) Combine randomized trees to reduce the instability of a single tree",
        ],
        "answer": "D",
        "explanation": "Random forests train trees using bootstrap samples by default and "
        "random feature subsets at splits, then average their class probabilities. "
        "Combining diverse trees often reduces variance, but improvement is not guaranteed. "
        "Sequential error correction describes boosting, not a random forest.",
    },
]


def check_answer(number, selected):
    """Return feedback for a one-based question number."""
    question = _get_question(number)
    if selected is None:
        return "Please select an answer first."
    answer = next(choice for choice in question["choices"] if choice.startswith(question["answer"] + ")"))
    verdict = "✅ **Correct!**" if selected == answer else f"❌ **Not quite.** Correct answer: {answer}"
    return f"{verdict}\n\n**Explanation:** {question['explanation']}"


def _get_question(number):
    if not isinstance(number, int) or not 1 <= number <= len(QUESTIONS):
        raise ValueError(f"Question number must be between 1 and {len(QUESTIONS)}")
    return QUESTIONS[number - 1]


def build_question(number, height=420, inline=True, share=False):
    """Launch one question inline, matching the Chapter 1 quiz format."""
    question = _get_question(number)
    with gr.Blocks(title=f"Machine Learning — Question {number}") as demo:
        gr.Markdown(f"### {question['question']}")
        choices = gr.Radio(choices=question["choices"], label="Choose one answer")
        submit = gr.Button("Submit Answer")
        feedback = gr.Markdown()
        submit.click(lambda selected: check_answer(number, selected), inputs=choices, outputs=feedback)

    kwargs = dict(inline=inline, share=share, height=height, quiet=True, prevent_thread_lock=True)
    parameters = inspect.signature(gr.Blocks.launch).parameters
    if "footer_links" in parameters:
        kwargs["footer_links"] = []
    elif "show_api" in parameters:
        kwargs["show_api"] = False
    demo.launch(**kwargs)
    return demo
