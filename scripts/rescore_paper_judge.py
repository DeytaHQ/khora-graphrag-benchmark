"""Diagnostic: re-score an existing run's answers with a paper-style 0/2/4
correctness judge (NVIDIA AnswerAccuracy-style) that grades factual correctness
vs gold and explicitly ignores phrasing/verbosity. Compares to the repo's F-beta
accuracy to quantify how much of the gap is a scorer artifact.

No graph build / no retrieval - reuses generated answers from report.json.
"""

import asyncio
import json
import sys

import litellm

REPORT = sys.argv[1] if len(sys.argv) > 1 else "results/af8486c67091/report.json"
MODEL = "gpt-4o-mini"
CONCURRENCY = 8

JUDGE = """You are grading whether a generated answer is factually correct relative to a reference (gold) answer for a question about a novel.

Grade ONLY factual correctness. IGNORE phrasing, length, word order, and extra true detail. The answer is correct if it conveys the key fact(s) of the gold answer, even if reworded or more verbose.

Rate on this scale:
- 4 = fully correct (conveys all key gold facts)
- 2 = partially correct (conveys some but not all key gold facts, or adds a wrong fact alongside a right one)
- 0 = incorrect (misses the key gold fact or contradicts it)

Question: {q}
Gold answer: {gold}
Generated answer: {gen}

Respond with JSON: {{"rating": 0|2|4}}"""


async def grade(sem, q):
    async with sem:
        prompt = JUDGE.format(q=q["question"], gold=q["gold_answer"], gen=q["generated_answer"])
        try:
            r = await litellm.acompletion(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=20,
                response_format={"type": "json_object"},
            )
            rating = int(json.loads(r.choices[0].message.content).get("rating", 0))
        except Exception as e:
            print("err:", e, file=sys.stderr)
            rating = -1
        return rating


async def main():
    data = json.load(open(REPORT))["result"]
    pq = data["per_question"]
    sem = asyncio.Semaphore(CONCURRENCY)
    ratings = await asyncio.gather(*[grade(sem, q) for q in pq])

    n = len(pq)
    fbeta_correct = [q["answer_correct"] for q in pq]
    # paper-style accuracy under two thresholds
    lenient = sum(1 for r in ratings if r >= 2) / n  # >= partially correct
    strict = sum(1 for r in ratings if r == 4) / n  # fully correct
    fbeta_acc = sum(fbeta_correct) / n

    # cross-tab: of F-beta FAILS, how many does the paper judge pass (artifact magnitude)
    fb_fail = [i for i in range(n) if not fbeta_correct[i]]
    rescued_full = sum(1 for i in fb_fail if ratings[i] == 4)
    rescued_partial = sum(1 for i in fb_fail if ratings[i] >= 2)
    # and the reverse: F-beta passes the paper judge calls incorrect
    fb_pass = [i for i in range(n) if fbeta_correct[i]]
    overscored = sum(1 for i in fb_pass if ratings[i] == 0)

    print(f"n = {n}")
    print(f"F-beta accuracy (current headline):        {fbeta_acc:.3f}")
    print(f"paper-style accuracy, fully-correct (==4): {strict:.3f}")
    print(f"paper-style accuracy, >=partial (>=2):     {lenient:.3f}")
    print()
    print(f"F-beta FAILS that paper judge rates FULLY correct (==4): {rescued_full}/{len(fb_fail)}")
    print(f"F-beta FAILS that paper judge rates >= partial (>=2):    {rescued_partial}/{len(fb_fail)}")
    print(f"F-beta PASSES that paper judge rates incorrect (==0):    {overscored}/{len(fb_pass)}")
    print(f"unparseable judge calls: {sum(1 for r in ratings if r < 0)}")


asyncio.run(main())
