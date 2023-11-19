#This file contains the configuration variables 


entities = ['Document Name','Parties','Agreement Date','Effective Date','Expiration Date','Governing Law']
data_path = 'train_input_data.csv'
llm_answers_path_n_samples = 'llm_answers_for_n_samples.csv'
llm_answers_path = 'llm_answers.csv'
llm_generated_output_path_in_dict_format = 'llm_generated_output_in_dict_format.csv'
llm_generated_output_path_in_entity_type_format = 'llm_generated_output_in_entity_type_format.csv'
gold_train_data_path = 'gold_train_data_entitywise.csv'


enitity_type_map = {
    'Document Name': 1,
    'Party': 2, 
    'Agreement Date': 4,
    'Effective Date': 5,
    'Expiration Date': 6,
    'Governing Law': 3
}                        