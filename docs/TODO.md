Things that need to be done:
- by default, pdf ingest should not ingest figures. so if a user decided it needs the figures, then it can ask to reingest the paper with figures.
- Check whether digest that score >= 9 are added to db as summary or full text? I think >= 9 should be added as full text. easiest thing to do is just drop a full copy into the pdf watch dir, and let daemon automatically ingest it as full text but with the rating set at whatever it was. Anything less should just be summary.
- add ability to create new obsidian notes based on conversation.
- add a button to copy llm response into clipboard. it should copy the response out in md format.