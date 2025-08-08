import json
import traceback
import importlib.resources as pkg_resources
import pandas as pd
import streamlit as st
from langchain_core.callbacks import UsageMetadataCallbackHandler
import config
from src.mcq_generator.utils import read_file, get_table_data
from src.mcq_generator.mcq_generator import generate_evaluate_chain

# create a title
st.title("MCQs Creator")

# Setup callback for token/cost tracking
callback = UsageMetadataCallbackHandler()

with pkg_resources.open_text(config, "response.json") as f:
    RESPONSE_JSON = json.load(f)

# create a form using st.form
with st.form("user_inputs"):
    # file_upload
    uploaded_file=st.file_uploader("Upload a PDF or Text file.")

    # Input Fields
    mcq_count = st.number_input("No. of MCQs", min_value=3, max_value=50)

    # Subject
    subject=st.text_input("Insert Subject", max_chars=20)

    # Quiz Tone
    tone=st.text_input("Complexity Level of Questions", max_chars=20, placeholder="Simple")

    # Add Button
    button=st.form_submit_button("Create MCQs")

if button and uploaded_file is not None and mcq_count and subject and tone:
    with st.spinner("loading.."):
        try:
            text=read_file(uploaded_file)
            # count tokens and the cost of API call using config callbacks
            response = generate_evaluate_chain.invoke(
                {
                    "text": text,
                    "number": mcq_count,
                    "subject": subject,
                    "tone": tone,
                    "response_json": json.dumps(RESPONSE_JSON)
                },
                config={"callbacks": [callback]}
            )
            st.write(response)
            if isinstance(response, str):
                # Extract the quiz data from the response
                if response is not None:
                    table_data = get_table_data(response)
                    if table_data is not None:
                        df = pd.DataFrame(table_data)
                        df.index = df.index + 1
                        st.table(df)
                        # Create download button (no file saved)
                        st.download_button(
                            label="Download Quiz Output",
                            data=df.to_csv(index=False).encode("utf-8"),
                            file_name=f"{subject}_quiz_output.csv",
                            mime="text/csv"
                        )
                        # Display the review in a text box as well
                        # st.text_area(label="Review", value=response.get("review", ""))
                    else:
                        st.error("Error in the table data")
                else:
                    st.error("No quiz data found in the response")
        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
            st.error("Error")
        else:
            usage = callback.usage_metadata
            gemini_usage = usage.get("gemini-2.5-pro", {})
            print("-------------------------------------------------------------------------")
            print(f"Total Tokens: {gemini_usage.get('total_tokens')}")
            print(f"Prompt Tokens (input): {gemini_usage.get('input_tokens')}")
            print(f"Completion Tokens (output): {gemini_usage.get('output_tokens')}")
            print(f"Output Token Details: {gemini_usage.get('output_token_details')}")
            print("-------------------------------------------------------------------------")








