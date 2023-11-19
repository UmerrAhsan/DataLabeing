import pandas as pd
from config import llm_generated_output_path_in_dict_format , llm_generated_output_path_in_entity_type_format,enitity_type_map
import ast 


def convert_llm_answers_to_dict(llm_answers_path):

    df = pd.read_csv(llm_answers_path)

    # Initialize an empty list to store rows for the new DataFrame
    new_rows = []

    for i in range(len(df)):   
        input_string = df['LLM_Answer'][i]

        # Initialize an empty dictionary
        result_dict = {}

        # Mapping of custom keys
        custom_keys = {
            'Document name': 'Document Name',
            'Parties': 'Party',
            'Governing law': 'Governing Law',
            'Agreement date': 'Agreement Date',
            'Effective date': 'Effective Date',
            'Expiration date': 'Expiration Date'
        }

        # Split the input string into lines
        lines = input_string.split('\n')

        # Iterate through each line and extract relevant information
        for line in lines:
            # Split the line into key and value using ':'
            key, value = map(str.strip, line.split(':'))
            
            # Check if a custom key mapping exists
            custom_key = custom_keys.get(key, key)
            
            # Check for the 'Parties' key and split the value by '#'
            if key == 'Parties':
                value_list = [item.strip() for item in value.split('#')]
            else:
                value_list = [value.strip()]
            
            # Add custom key and value to the dictionary
            result_dict[custom_key] = value_list

        # Append the 'uuid' and the dictionary-converted answer to the new_rows list
        new_rows.append({'uuid': df['uuid'][i], 'text' : df['text'][i],'converted_answer': result_dict})

    # Create a new DataFrame from the list of rows
    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(llm_generated_output_path_in_dict_format, index=False)

    return new_df




def convert_dict_format_into_entity_type_format(datapath):
    
    # Read CSV file
    df = pd.read_csv(datapath)

    # Initialize an empty list to store rows for the new DataFrame
    new_rows = []

    # Iterate through each row
    for i in range(len(df)):
        uuid = df['uuid'][i]
        
        # Convert the string back to a dictionary using ast.literal_eval
        converted_dict = ast.literal_eval(df['converted_answer'][i])

        # Iterate through the dictionary in the 'converted_answer' column
        for entity_type, entity_text_list in converted_dict.items():
            # Iterate through the list of entity texts
            for entity_text in entity_text_list:
                # Append a new row to the list
                new_rows.append({'uuid': uuid, 'entity_type': entity_type, 'entity_text': entity_text})

    # Create a new DataFrame from the list of rows
    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(llm_generated_output_path_in_entity_type_format, index=False)

    return new_df







def encode_entity(datapath):
    df =pd.read_csv(datapath)
    df['entity_type'] = df['entity_type'].map(enitity_type_map)
    df.to_csv(datapath,index=False)

