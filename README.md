# IITM LLM + DLCV Study Hub

Open the study hub at <https://michaeldatascience.github.io/IITM-STUDY/>. This repository contains the reviewed study artifacts for both courses and can also be served locally from `index.html`.

## Status classes

- **Canonical:** generated or reviewed against official material and approved for the main study sequence.
- **Reviewed archive:** useful prior material that has been inspected and cataloged, but has not yet been rebuilt into the canonical sequence.
- **Planned:** syllabus slot exists; artifact is not yet generated.

## Structure

- `index.html` - single study entry point.
- `shared/` - concepts taught once and reused by both courses.
- `llm/` - canonical LLM modules.
- `dlcv/` - canonical DLCV modules.
- `archive/` - reviewed prior notes and interactive aids.
- `docs/` - resource catalog and syllabus/progress tracker snapshots.

Source PDFs, raw assignments, and PYQ collections remain outside this folder. Only reviewed study outputs belong here.

## Update rule

After a new volume is discussed and reviewed:

1. Place its final Markdown/HTML inside the appropriate course/shared folder.
2. Add it to `index.html`.
3. Update the status/progress documents.
4. Test all links and interactive controls through the local server.
5. Publish the complete `study-hub` folder only when requested.
