import llm_generation as llm
import outputParser as op
from config import llm_answers_path
from config import llm_answers_path_n_samples
from config import data_path
from config import llm_generated_output_path_in_dict_format
from config import llm_generated_output_path_in_entity_type_format


#llm.generate_labels_from_llm_for_n_samples(data_path,10)
llm.generate_labels_from_llm(data_path)
op.convert_llm_answers_to_dict(llm_answers_path)
op.convert_dict_format_into_entity_type_format(llm_generated_output_path_in_dict_format)
op.encode_entity(llm_generated_output_path_in_entity_type_format)
