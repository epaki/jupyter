StudyBuddy CM2015 Midterm Project
=================================

Files included:
- StudyBuddy_CM2015_midterm.ipynb: main Jupyter notebook with chatbot code, demo, and tests.
- intents.json: external regex intent dataset loaded by the notebook.
- StudyBuddy_CM2015_report.pdf: PDF report describing use case, design, tests, advanced features, and reflection.
- StudyBuddy_CM2015_midterm.html: exported HTML version of the notebook.

How to run:
1. Put StudyBuddy_CM2015_midterm.ipynb and intents.json in the same folder.
2. Open the notebook in Jupyter.
3. Run all cells from top to bottom.
4. To use the interactive chatbot, run run_chatbot() in a new notebook cell.
5. Type exit or quit to end the chat.

Notes:
- The notebook uses relative paths, so intents.json must remain beside the notebook.
- NLTK is used for Porter stemming when available; no NLTK downloads are required.
