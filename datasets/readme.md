## Datasets Candidate

* [Open RAG Bench][1]
  * need to run the code to get the dataset
  * Code base is very valuable 🌟
    * Download papers from arxiv and get metadata: https://github.com/vectara/open-rag-bench/blob/main/openragbench/pipeline/data_processing/get_arxiv.py
      * Metadata includes title, author, category, etc.
    * It's using Mistral OCR to extract content from PDF: http://github.com/vectara/open-rag-bench/blob/main/openragbench/models/processors.py
      * It can collect both text and images in the paper
      * The code in the file can also extract tables from the paper
    * It auto generates QA pairs
      * Extract spans for numbers, definitions, comparison, results, concepts
      * Auto filter low quality spans (such as vague spans)
      * LLM refine span content
      * For each span, LLM to generate the question
* [WixQA][2]
  * enterprise RAG benchmark
* [mmRAG Benchmark][4]

### Relevant Resources
* [Vidore Leaderboard for Embedding Models][3]

[1]:https://github.com/vectara/open-rag-bench/tree/main
[2]:https://huggingface.co/datasets/Wix/WixQA
[3]:https://huggingface.co/spaces/vidore/vidore-leaderboard
[4]:https://huggingface.co/datasets/Askio/mmrag_benchmark