# AI-based Question Answer Pair Generation with Llama Index
This project leverages Llama Index to generate question-answer pairs from various file formats such as PDF, TXT, PPT, or CSV. Additionally, it provides justification for each answer's correctness along with the source from the uploaded file.

## How to Use:
First create a virtual environment and install the requirements

### Using qa.py (Terminal Version):

1. Place the qa.py file in the data folder.
2. Run the qa.py file in your terminal.
3. The output will be saved in a CSV file named "eval_results.csv".

### Using Api.py (API Version):

1. Run Api.py.
2. Send a POST request to http://127.0.0.1:8000/evaluate with the file attached.
3. Receive the questions and answers generated as output in a CSV file.
