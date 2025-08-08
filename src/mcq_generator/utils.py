import json
import pandas as pd
from PyPDF2 import PdfReader
from langchain_core.output_parsers import JsonOutputParser

def read_file(file):
    """
    Reads the content of a file (PDF or text).
    Parameters:
        file: A file-like object with a `.name` attribute (e.g., an uploaded file from Streamlit).
    Returns:
        str: Extracted text content from the file.
    Supported formats:
        - PDF (.pdf): Extracts text from all pages using PyPDF2.
        - Text (.txt): Reads and decodes UTF-8 text files.
    Raises:
        Exception: If the file format is unsupported or if reading fails.
    """
    if file.name.endswith(".pdf"):
        try:
            reader = PdfReader(file)
            text = ""
            pages = getattr(reader, "pages", None)
            if pages is None:
                pages = reader.pages
            for page in pages:
                try:
                    text += page.extract_text() or ""
                except Exception:
                    text += page.get_text() or ""
            return text
        except Exception as e:
            raise Exception("error reading the PDF file: " + str(e))
    elif file.name.endswith(".txt"):
        try:
            return file.read().decode("utf-8")
        except Exception:
            return file.read()
    else:
        raise Exception("unsupported file format only pdf and text file supported")


def extract_json_from_markdown(text):
    """
    Try to find and return the first valid JSON object within the text,
    by scanning progressively larger substrings and trying to json.loads them.
    """
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace == -1 or last_brace == -1:
        return text  # no braces, return as is

    for end in range(last_brace, first_brace - 1, -1):
        try:
            candidate = text[first_brace:end + 1]
            obj = json.loads(candidate)
            return candidate  # return substring that is valid JSON
        except json.JSONDecodeError:
            continue
    return text


def get_table_data(quiz_input):
    """
    Convert a quiz response (JSON-like) into a structured pandas DataFrame.
    This function:
      1. Accepts either a dictionary or string containing quiz MCQ data.
      2. If the input is a dictionary, it is converted to a JSON string.
      3. If the input is a string, attempts to extract the valid JSON portion
         using `extract_json_from_markdown`.
      4. Parses the JSON into Python objects using `JsonOutputParser`.
      5. Iterates over the parsed quiz data to create a tabular structure with
         MCQ number, question text, options, and the correct answer.
    Parameters:
        quiz_input (dict | str):
            - A dictionary already containing MCQ data in structured form. OR
            - A string containing either raw JSON or JSON embedded in other text.
    Returns:
        pandas.DataFrame:
            A DataFrame with the following columns:
                - "MCQ Number": Question number (int or str)
                - "MCQ": The question text
                - "Option A" to "Option D": Answer choices
                - "Correct": Correct option in "key: value" format
    Notes:
        - Keys in the parsed JSON are sorted numerically if possible.
        - If no valid JSON is found or parsing fails, returns None.
    """

    # Clean/convert input to valid JSON string for parsing
    if isinstance(quiz_input, dict):
        quiz_input = json.dumps(quiz_input)
    elif isinstance(quiz_input, str):
        # If string is not a pure JSON, try to extract clean JSON part
        quiz_input = extract_json_from_markdown(quiz_input)

    parser = JsonOutputParser()
    try:
        parsed_output = parser.invoke(quiz_input)
    except Exception as e:
        print("Error parsing result content:", e)
        parsed_output = {}
    if not parsed_output:
        print("[Error] Parsed Output is empty. Please check your input.")
        return

    quiz_table_data = []
    # Sort keys numerically if possible
    keys_sorted = sorted(parsed_output.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
    for key in keys_sorted:
        value = parsed_output[key]
        mcq = value.get("mcq", "")
        options = value.get("options", {})
        option_a = options.get("a", "")
        option_b = options.get("b", "")
        option_c = options.get("c", "")
        option_d = options.get("d", "")
        correct_key = value.get("correct", "")
        correct_text = options.get(correct_key, "")
        correct = f"{correct_key}: {correct_text}" if correct_key else ""
        try:
            mcq_number = int(key)
        except Exception:
            mcq_number = key
        quiz_table_data.append({
            "MCQ Number": mcq_number,
            "MCQ": mcq,
            "Option A": option_a,
            "Option B": option_b,
            "Option C": option_c,
            "Option D": option_d,
            "Correct": correct
        })
    quiz_df = pd.DataFrame(quiz_table_data)

    # Save to Excel
    # excel_filename = r"D:\Learning\generative ai ineuron\MCQ Generator\data\_mcq1.xlsx"
    # quiz_df.to_excel(excel_filename, index=False)
    # print(f"\n Quiz saved to {excel_filename}")
    return quiz_df
