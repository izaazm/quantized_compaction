from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_BIOGRAPHIES_PATH = DATA_DIR / "tofu_author_biographies.json"
DEFAULT_PASSAGES_PATH = DATA_DIR / "tofu_qa_passages.json"


@dataclass(frozen=True)
class QARecord:
    author_idx: int
    qa_idx: int
    question: str
    answer: str
    passage: str
    record_id: str | None = None
    answer_aliases: tuple[str, ...] = ()
    dataset: str = "tofu"
    task: str = "qa"
    answer_prefix: str = ""
    max_new_tokens: int | None = None
    metadata: dict[str, Any] | None = None
    scoring_references: tuple[str, ...] = ()
    teacher_forced_answer: str | None = None

    @property
    def question_id(self) -> str:
        return self.record_id or f"author{self.author_idx}_q{self.qa_idx}"

    @property
    def references(self) -> tuple[str, ...]:
        """All accepted answers, with the canonical answer first."""

        if self.scoring_references:
            return self.scoring_references
        return tuple(dict.fromkeys((self.answer, *self.answer_aliases)))

    @property
    def nll_target(self) -> str:
        """Reference continuation used for teacher-forced likelihood."""

        return self.teacher_forced_answer or self.answer


@dataclass(frozen=True)
class TofuCorpus:
    text: str
    questions: tuple[QARecord, ...]
    author_ids: tuple[int, ...]
    source_format: str
    dataset_name: str = "tofu"
    context_id: str = "tofu"
    metadata: dict[str, Any] | None = None


# The runner also uses this structure for RULER, QuALITY, and HotPotQA. Keep the old
# name as a compatibility alias because it appears in the public API and tests.
EvaluationCorpus = TofuCorpus


def _read_json(path: Path) -> object:
    if not path.exists():
        raise FileNotFoundError(f"Dataset artifact not found: {path}")
    with path.open() as handle:
        return json.load(handle)


def _normalize_author_ids(author_ids: Iterable[int]) -> tuple[int, ...]:
    result = tuple(dict.fromkeys(int(author_id) for author_id in author_ids))
    if not result:
        raise ValueError("At least one author ID is required.")
    if any(author_id < 0 for author_id in result):
        raise ValueError(f"Author IDs must be non-negative: {result}")
    return result


def load_tofu_corpus(
    biographies_path: Path = DEFAULT_BIOGRAPHIES_PATH,
    passages_path: Path = DEFAULT_PASSAGES_PATH,
    author_ids: Sequence[int] = (0, 1, 2, 3, 4),
    corpus_format: str = "biography",
    max_questions: int | None = None,
    seed: int = 42,
) -> TofuCorpus:
    """Load the local synthesized TOFU corpus and its source QA pairs.

    ``biography`` uses the rewritten author biographies. ``passages`` joins the
    generated 1--3 sentence passages without another rewrite.
    """

    normalized_ids = _normalize_author_ids(author_ids)
    selected_ids = set(normalized_ids)

    raw_passages = _read_json(Path(passages_path))
    if not isinstance(raw_passages, list):
        raise ValueError(f"Expected a JSON list in {passages_path}")

    questions: list[QARecord] = []
    passages_by_author: dict[int, list[str]] = {
        author_id: [] for author_id in normalized_ids
    }
    for row in raw_passages:
        author_idx = int(row["author_idx"])
        if author_idx not in selected_ids:
            continue
        questions.append(
            QARecord(
                author_idx=author_idx,
                qa_idx=int(row["qa_idx"]),
                question=str(row["question"]),
                answer=str(row["answer"]),
                passage=str(row["passage"]),
            )
        )
        passages_by_author[author_idx].append(str(row["passage"]))

    available_question_authors = {record.author_idx for record in questions}
    missing_questions = selected_ids - available_question_authors
    if missing_questions:
        raise ValueError(
            f"No local QA passages for author IDs {sorted(missing_questions)}. "
            f"The current artifact covers {sorted(available_question_authors)}."
        )

    if corpus_format == "biography":
        raw_biographies = _read_json(Path(biographies_path))
        if not isinstance(raw_biographies, list):
            raise ValueError(f"Expected a JSON list in {biographies_path}")
        biographies = {
            int(row["author_idx"]): str(row["biography"])
            for row in raw_biographies
            if int(row["author_idx"]) in selected_ids
        }
        missing_biographies = selected_ids - set(biographies)
        if missing_biographies:
            raise ValueError(
                f"No synthesized biography for author IDs {sorted(missing_biographies)}"
            )
        sections = [
            f"--- Fictional author profile {author_id} ---\n{biographies[author_id]}"
            for author_id in normalized_ids
        ]
    elif corpus_format == "passages":
        sections = [
            f"--- Fictional author profile {author_id} ---\n"
            + "\n".join(passages_by_author[author_id])
            for author_id in normalized_ids
        ]
    else:
        raise ValueError(
            f"Unknown corpus format {corpus_format!r}; choose 'biography' or 'passages'."
        )

    questions.sort(key=lambda record: (record.author_idx, record.qa_idx))
    if max_questions is not None:
        if max_questions <= 0:
            raise ValueError("max_questions must be positive when provided.")
        shuffled = questions.copy()
        random.Random(seed).shuffle(shuffled)
        questions = shuffled[:max_questions]

    return TofuCorpus(
        text="\n\n".join(sections),
        questions=tuple(questions),
        author_ids=normalized_ids,
        source_format=corpus_format,
    )


def _sample_rows(
    rows: Sequence[dict[str, Any]],
    max_questions: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    sampled = list(rows)
    if max_questions is None:
        return sampled
    if max_questions <= 0:
        raise ValueError("max_questions must be positive when provided.")
    random.Random(seed).shuffle(sampled)
    return sampled[:max_questions]


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            rows.append(value)
    return rows


def load_ruler_corpora(
    dataset_path: Path,
    max_questions: int | None = None,
    seed: int = 42,
) -> tuple[EvaluationCorpus, ...]:
    """Load a prepared RULER JSONL subset.

    The preparation script preserves the common RULER fields: ``context``,
    ``question``, ``answer``, ``task``, ``answer_prefix`` and
    ``max_new_tokens``. Questions sharing the same RULER context are grouped so
    cache extraction happens once while question-level scoring remains intact.
    """

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Prepared RULER data not found: {path}. Run "
            "`python -m compaction_limit.prepare_benchmarks`."
        )
    rows = _sample_rows(_jsonl_rows(path), max_questions, seed)
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(str(row["context"]), []).append((index, row))

    corpora: list[EvaluationCorpus] = []
    for context_number, (context, context_rows) in enumerate(grouped.items()):
        records: list[QARecord] = []
        tasks: set[str] = set()
        for index, row in context_rows:
            references = row.get("answer", row.get("answers", []))
            if isinstance(references, str):
                references = [references]
            references = tuple(str(value) for value in references if str(value))
            if not references:
                raise ValueError(f"RULER row {index} has no reference answers.")
            record_id = str(
                row.get("id", row.get("question_id", f"ruler_{index}"))
            )
            task = str(row.get("task", "unknown"))
            tasks.add(task)
            teacher_forced_answer = str(
                row.get(
                    "teacher_forced_answer",
                    ", ".join(references) if task == "fwe" else references[0],
                )
            )
            answer_prefix = str(row.get("answer_prefix", ""))
            if (
                answer_prefix
                and teacher_forced_answer
                and not answer_prefix[-1].isspace()
                and not teacher_forced_answer[0].isspace()
            ):
                teacher_forced_answer = " " + teacher_forced_answer
            records.append(
                QARecord(
                    author_idx=-1,
                    qa_idx=index,
                    question=str(row["question"]),
                    answer=references[0],
                    passage=context,
                    record_id=record_id,
                    answer_aliases=references[1:],
                    dataset="ruler",
                    task=task,
                    answer_prefix=answer_prefix,
                    max_new_tokens=int(row.get("max_new_tokens", 64)),
                    metadata={
                        key: row[key]
                        for key in ("task", "source_index", "context_length")
                        if key in row
                    },
                    scoring_references=references,
                    teacher_forced_answer=teacher_forced_answer,
                )
            )
        context_id = f"ruler_context_{context_number}"
        if len(records) == 1:
            context_id = records[0].question_id
        corpora.append(
            EvaluationCorpus(
                text=context,
                questions=tuple(records),
                author_ids=(),
                source_format="ruler_jsonl",
                dataset_name="ruler",
                context_id=context_id,
                metadata={"tasks": sorted(tasks)},
            )
        )
    return tuple(corpora)


def load_hotpotqa_corpora(
    dataset_path: Path,
    max_questions: int | None = None,
    seed: int = 42,
) -> tuple[EvaluationCorpus, ...]:
    """Load HotPotQA distractor examples in their official JSON format."""

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Prepared HotPotQA data not found: {path}. Run "
            "`python -m compaction_limit.prepare_benchmarks`."
        )
    raw = _read_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list in {path}.")
    rows = _sample_rows(raw, max_questions, seed)
    corpora: list[EvaluationCorpus] = []
    for index, row in enumerate(rows):
        context_id = str(row.get("_id", f"hotpotqa_{index}"))
        sections = []
        for title, sentences in row.get("context", []):
            paragraph = " ".join(str(value).strip() for value in sentences)
            sections.append(f"## {title}\n{paragraph}")
        context = "\n\n".join(sections)
        answer = str(row["answer"])
        record = QARecord(
            author_idx=-1,
            qa_idx=index,
            question=str(row["question"]),
            answer=answer,
            passage=context,
            record_id=context_id,
            dataset="hotpotqa",
            task=str(row.get("type", "qa")),
            metadata={
                "level": row.get("level"),
                "type": row.get("type"),
                "supporting_facts": row.get("supporting_facts", []),
            },
        )
        corpora.append(
            EvaluationCorpus(
                text=context,
                questions=(record,),
                author_ids=(),
                source_format="hotpotqa_distractor",
                dataset_name="hotpotqa",
                context_id=context_id,
                metadata={"level": row.get("level"), "type": row.get("type")},
            )
        )
    return tuple(corpora)


def load_quality_corpora(
    dataset_path: Path,
    max_questions: int | None = None,
    max_contexts: int | None = None,
    seed: int = 42,
) -> tuple[EvaluationCorpus, ...]:
    """Load QuALITY and group every article's questions under one KV context."""

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Prepared QuALITY data not found: {path}. Run "
            "`python -m compaction_limit.prepare_benchmarks --skip-ruler "
            "--skip-hotpotqa`."
        )

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    question_index = 0
    for row in _jsonl_rows(path):
        article_id = str(row["article_id"])
        article = str(row.get("article", "")).strip()
        if not article:
            raise ValueError(f"QuALITY article {article_id!r} is empty.")
        group = grouped.setdefault(
            (article_id, article),
            {
                "article_id": article_id,
                "article": article,
                "title": str(row.get("title", "")),
                "questions": [],
            },
        )
        set_id = str(row.get("set_unique_id", article_id))
        for local_index, question in enumerate(row.get("questions", [])):
            options = tuple(str(value) for value in question.get("options", []))
            if len(options) != 4:
                raise ValueError(
                    f"QuALITY question {set_id}:{local_index} has {len(options)} "
                    "options; expected four."
                )
            gold_label = int(question["gold_label"])
            if gold_label not in (1, 2, 3, 4):
                raise ValueError(
                    f"QuALITY question {set_id}:{local_index} has invalid gold "
                    f"label {gold_label}."
                )
            letters = "ABCD"
            prompt = "\n".join(
                [
                    str(question["question"]),
                    "",
                    *(f"{letter}. {option}" for letter, option in zip(letters, options)),
                ]
            )
            group["questions"].append(
                QARecord(
                    author_idx=-1,
                    qa_idx=question_index,
                    question=prompt,
                    answer=letters[gold_label - 1],
                    passage=article,
                    record_id=f"quality_{set_id}_q{local_index}",
                    dataset="quality",
                    task="multiple_choice",
                    max_new_tokens=4,
                    metadata={
                        "article_id": article_id,
                        "set_unique_id": set_id,
                        "options": list(options),
                        "gold_label": gold_label,
                        "difficult": bool(question.get("difficult", False)),
                    },
                )
            )
            question_index += 1

    if max_questions is not None and max_questions <= 0:
        raise ValueError("max_questions must be positive when provided.")
    if max_contexts is not None and max_contexts <= 0:
        raise ValueError("max_contexts must be positive when provided.")

    articles = list(grouped.values())
    if max_questions is not None or max_contexts is not None:
        random.Random(seed).shuffle(articles)
    if max_contexts is not None:
        articles = articles[:max_contexts]

    corpora: list[EvaluationCorpus] = []
    remaining = max_questions
    for article in articles:
        questions = article["questions"]
        if remaining is not None:
            if remaining <= 0:
                break
            questions = questions[:remaining]
            remaining -= len(questions)
        if not questions:
            continue
        corpora.append(
            EvaluationCorpus(
                text=article["article"],
                questions=tuple(questions),
                author_ids=(),
                source_format="quality_v1.0.1_htmlstripped_dev",
                dataset_name="quality",
                context_id=f"quality_{article['article_id']}",
                metadata={"title": article["title"]},
            )
        )
    return tuple(corpora)


def load_evaluation_corpora(
    dataset_name: str,
    *,
    biographies_path: Path = DEFAULT_BIOGRAPHIES_PATH,
    passages_path: Path = DEFAULT_PASSAGES_PATH,
    dataset_path: Path | None = None,
    author_ids: Sequence[int] = (0, 1, 2, 3, 4),
    corpus_format: str = "biography",
    max_questions: int | None = None,
    max_contexts: int | None = None,
    seed: int = 42,
) -> tuple[EvaluationCorpus, ...]:
    normalized = dataset_name.lower()
    if normalized == "tofu":
        return (
            load_tofu_corpus(
                biographies_path=biographies_path,
                passages_path=passages_path,
                author_ids=author_ids,
                corpus_format=corpus_format,
                max_questions=max_questions,
                seed=seed,
            ),
        )
    if dataset_path is None:
        raise ValueError(f"dataset_path is required for dataset {dataset_name!r}.")
    if normalized == "ruler":
        return load_ruler_corpora(dataset_path, max_questions=max_questions, seed=seed)
    if normalized == "quality":
        return load_quality_corpora(
            dataset_path,
            max_questions=max_questions,
            max_contexts=max_contexts,
            seed=seed,
        )
    if normalized == "hotpotqa":
        return load_hotpotqa_corpora(
            dataset_path, max_questions=max_questions, seed=seed
        )
    raise ValueError(
        f"Unknown dataset {dataset_name!r}; choose 'tofu', 'ruler', 'quality', "
        "or 'hotpotqa'."
    )
