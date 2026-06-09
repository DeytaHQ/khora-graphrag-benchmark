# Third-Party Licenses

`khora-graphrag-benchmark` is licensed under Apache-2.0 (see [`LICENSE`](LICENSE)).
It includes material adapted from the third-party project below, whose license
is reproduced in full as that license requires.

## GraphRAG-Bench

Portions of this repository are adapted from the GraphRAG-Bench reference
implementation (<https://github.com/GraphRAG-Bench/GraphRAG-Benchmark>),
specifically:

- the LLM-judge prompts and few-shot examples in
  `src/khora_graphrag_bench/harness/evaluation.py` (adapted from the upstream
  `Evaluation/metrics/*.py`), and
- the novel-domain entity / relationship type allowlists in
  `src/khora_graphrag_bench/datasets/loader.py`.

The GraphRAG-Bench dataset (questions + source corpus) is likewise MIT-licensed
and is downloaded at runtime from Hugging Face
(`GraphRAG-Bench/GraphRAG-Bench`); no dataset content is redistributed in this
repository.

GraphRAG-Bench is distributed under the MIT License:

```
MIT License

Copyright (c) 2025 XMU-DeepLIT

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
