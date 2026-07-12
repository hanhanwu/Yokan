# Yokan - My Experiment Space

## About
Before any implementation goes to [RAGDoctor][1], [GroundLens][2] or more, I do experiments here.

## Experiments
* [Golden Dataset Generation][4]
* [Root Cause Analysis for Information Retrieval][3]
* [Multimodal][4]


## Human-AI Interaction UI
* `Yokan/frontend/` shows side by side chunking strategies comparison applied on the same document
  * Commands to refresh chunks:
    * `cd experiments_goldenset`
    * `source ../.venv13/bin/activate`
    * `python generate_chunks.py`
  * Commands to launch the UI:
    * `cd frontend`
    * `npx expo start --web`
* `Yokan/frontend2/` shows automatic retrieved text --> human review --> submit review & AI diagnose potential reasons
  * Commands to launch the UI:
    * `source ../.venv13/bin/activate`
    * `cd frontend2`
    * `npx expo start --web`
* Other UIs, check each sub-folder's readme


[1]:https://github.com/hanhanwu/RagDoctor
[2]:https://github.com/hanhanwu/GroundLens
[3]:https://github.com/hanhanwu/Yokan/tree/main/RCA4IR
[4]:https://github.com/hanhanwu/Yokan/tree/main/experiments_goldenset
[5]:https://github.com/hanhanwu/Yokan/tree/main/experiments_multimodal