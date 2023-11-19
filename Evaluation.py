from config import gold_train_data_path,llm_generated_output_path_in_entity_type_format
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import spacy

def get_gold_train_data():
    df = pd.read_csv(gold_train_data_path)
    return df


def get_llm_generated_data():
    df = pd.read_csv(llm_generated_output_path_in_entity_type_format)
    df = df.dropna(subset=['entity_text'])
    return df


def count_unique_samples():
    total_data_to_label = get_gold_train_data().uuid.nunique()
    print(f'unique uuids in gold_data is {total_data_to_label}')

    with open('EvaluationResult.txt', 'w') as file:
        # Write content to the file
        file.write(f'unique uuids in gold_data is {total_data_to_label}\n\n')


def count_unique_samples_2():
    # find the no of unique sentences in df where ws_entity_type is not equal to 0. It means that sentence has atleast one entity
    data_labeled_by_ws = get_llm_generated_data()['uuid'].nunique()
    print(f'Total data labeled means any non-zero ws_entity_type assigned: {data_labeled_by_ws}')
    # Open a file for appending
    with open('EvaluationResult.txt', 'a') as file:
        # Append content to the file
        file.write(f'Total data labeled means any non-zero ws_entity_type assigned: {data_labeled_by_ws}\n\n')


def count_unique_entity_wise_samples():
    
    #gold_dataset

    gold_data = get_gold_train_data()
    print(f'entity wise count in gold_data is \n{gold_data.gt_entity_type.value_counts()}')
    # Open a file for appending
    with open('EvaluationResult.txt', 'a') as file:
        # Append content to the file
        file.write(f'entity wise count in gold_data is \n{gold_data.gt_entity_type.value_counts()}\n\n')

    #llm_generated_dataset
    # show now the value counts of ws_entity_type in labeled_data
    ws_data = get_llm_generated_data()
    print(f'Labeled data ws_entity_type value counts: \n{ws_data.entity_type.value_counts()}')
    # Open a file for appending
    with open('EvaluationResult.txt', 'a') as file:
        # Append content to the file
        file.write(f'Labeled data ws_entity_type value counts: \n{ws_data.entity_type.value_counts()}\n\n')


def agregate_data():

    ws_data = get_llm_generated_data()
    gold_data = get_gold_train_data()
        # aggregate the data on the basis of uuid and ws_entity_type 
    ws_train_data_agg = ws_data.groupby(['uuid','entity_type']).agg({
        'entity_text': lambda x: ' '.join(x), 
    }).reset_index()
    gold_data_agg = gold_data.groupby(['uuid','gt_entity_type']).agg({
        'gt_entity_text': lambda x: ' '.join(x), 
    }).reset_index()

    with open('EvaluationResult.txt', 'a') as file:
        # Append content to the file
        file.write(f'ws_train_data_agg unique samples values counts: \n{ws_train_data_agg.entity_type.value_counts()}\n\n')
        file.write(f'gold_data_agg unique samples values counts: \n{gold_data_agg.gt_entity_type.value_counts()}\n\n')

    
    # Filter the ws_data_agg for class 1
    correct_doc_name_entity = ws_train_data_agg[
        (ws_train_data_agg['entity_type'] == 1) & (ws_train_data_agg['uuid'].isin(gold_data_agg['uuid']))]

    # Merge correct_doc_name_entity with gold_data_agg on uuid and select desired columns
    correct_doc_name_entity = correct_doc_name_entity.merge(gold_data_agg[['uuid','gt_entity_type', 'gt_entity_text']], on='uuid', how='left')

    # Filter for class 1 again to ensure only relevant data is included
    correct_doc_name_entity = correct_doc_name_entity[(correct_doc_name_entity['entity_type'] == 1)
                                                    & (correct_doc_name_entity['gt_entity_type'] == 1)]

    # Select the desired columns
    correct_doc_name_entity = correct_doc_name_entity[['uuid', 'entity_type', 'entity_text', 'gt_entity_type', 'gt_entity_text']]
    print(f'No of correct document name entities extracted by ws: {correct_doc_name_entity.shape[0]}')


        # Open a file for appending
    with open('EvaluationResult.txt', 'a') as file:
        # Append content to the file
        file.write(f'No of correct document name entities extracted by ws: {correct_doc_name_entity.shape[0]}\n\n')

    # Filter the ws_data_agg for class 2
    correct_party_name_entity = ws_train_data_agg[
        (ws_train_data_agg['entity_type'] == 2) & (ws_train_data_agg['uuid'].isin(gold_data_agg['uuid']))]

    # Merge correct_party_name_entity with gold_data_agg on uuid and select desired columns
    correct_party_name_entity = correct_party_name_entity.merge(gold_data_agg[['uuid','gt_entity_type', 'gt_entity_text']], on='uuid', how='left')

    # Filter for class 2 again to ensure only relevant data is included
    correct_party_name_entity = correct_party_name_entity[(correct_party_name_entity['entity_type'] == 2)
                                                    & (correct_party_name_entity['gt_entity_type'] == 2)]

    # Select the desired columns
    correct_party_name_entity = correct_party_name_entity[['uuid', 'entity_type', 'entity_text', 'gt_entity_type', 'gt_entity_text']]
    print(f'No of correct party name entities extracted by ws: {correct_party_name_entity.shape[0]}')


        # Open a file for appending
    with open('EvaluationResult.txt', 'a') as file:
        # Append content to the file
        file.write(f'No of correct party name entities extracted by ws: {correct_party_name_entity.shape[0]}\n\n')

        # Filter the ws_data_agg for class 3
    correct_gov_law_entity = ws_train_data_agg[
        (ws_train_data_agg['entity_type'] == 3) & (ws_train_data_agg['uuid'].isin(gold_data_agg['uuid']))]

    # Merge correct_gov_law_entity with gold_data_agg on uuid and select desired columns
    correct_gov_law_entity = correct_gov_law_entity.merge(gold_data_agg[['uuid','gt_entity_type', 'gt_entity_text']], on='uuid', how='left')

    # Filter for class 3 again to ensure only relevant data is included
    correct_gov_law_entity = correct_gov_law_entity[(correct_gov_law_entity['entity_type'] == 3)
                                                    & (correct_gov_law_entity['gt_entity_type'] == 3)]

    # Select the desired columns
    correct_gov_law_entity = correct_gov_law_entity[['uuid', 'entity_type', 'entity_text', 'gt_entity_type', 'gt_entity_text']]
    print(f'No of correct government law entities extracted by ws: {correct_gov_law_entity.shape[0]}')


        # Open a file for appending
    with open('EvaluationResult.txt', 'a') as file:
        # Append content to the file
        file.write(f'No of correct government law entities extracted by ws: {correct_gov_law_entity.shape[0]}\n\n')



        # Filter the ws_data_agg for class 1
    unique_correct_doc_name_entity = ws_train_data_agg[(ws_train_data_agg['entity_type'] == 1) 
                                        & (ws_train_data_agg['uuid'].isin(gold_data_agg['uuid']))
                                        & (gold_data_agg['gt_entity_type'] == 1)]
    print(f'No of correct labeled doc name entity: {correct_doc_name_entity.shape[0]}')
    unique_correct_party_name_entity = ws_train_data_agg[(ws_train_data_agg['entity_type'] == 2) 
                                        & (ws_train_data_agg['uuid'].isin(gold_data_agg['uuid']))
                                        & (gold_data_agg['gt_entity_type'] == 2)]
    print(f'No of correct labeled party name entity: {correct_party_name_entity.shape[0]}')
    unique_correct_gov_law_entity = ws_train_data_agg[(ws_train_data_agg['entity_type'] == 3) 
                                        & (ws_train_data_agg['uuid'].isin(gold_data_agg['uuid']))
                                        & (gold_data_agg['gt_entity_type'] == 3)]
    print(f'No of correct labeled gov law entity: {correct_gov_law_entity.shape[0]}')

        # Open a file for appending
    with open('EvaluationResult.txt', 'a') as file:
        # Append content to the file
        file.write(f'No of correct labeled doc name entity: {unique_correct_doc_name_entity.shape[0]}\n\n')
        file.write(f'No of correct labeled party name entity: {unique_correct_party_name_entity.shape[0]}\n\n')
        file.write(f'No of correct labeled gov law entity: {unique_correct_gov_law_entity.shape[0]}\n\n')  


    return correct_doc_name_entity,correct_party_name_entity,correct_gov_law_entity



def aggregate():
   
    ws_data = get_llm_generated_data()
    gold_data = get_gold_train_data()
        # aggregate the data on the basis of uuid and ws_entity_type 
    ws_train_data_agg = ws_data.groupby(['uuid','entity_type']).agg({
        'entity_text': lambda x: ' '.join(x), 
    }).reset_index()
    gold_data_agg = gold_data.groupby(['uuid','gt_entity_type']).agg({
        'gt_entity_text': lambda x: ' '.join(x), 
    }).reset_index()


        # Filter the ws_data_agg for class 1
    correct_doc_name_entity = ws_train_data_agg[
        (ws_train_data_agg['entity_type'] == 1) & (ws_train_data_agg['uuid'].isin(gold_data_agg['uuid']))]

    # Merge correct_doc_name_entity with gold_data_agg on uuid and select desired columns
    correct_doc_name_entity = correct_doc_name_entity.merge(gold_data_agg[['uuid','gt_entity_type', 'gt_entity_text']], on='uuid', how='left')

    # Filter for class 1 again to ensure only relevant data is included
    correct_doc_name_entity = correct_doc_name_entity[(correct_doc_name_entity['entity_type'] == 1)
                                                    & (correct_doc_name_entity['gt_entity_type'] == 1)]

    # Select the desired columns
    correct_doc_name_entity = correct_doc_name_entity[['uuid', 'entity_type', 'entity_text', 'gt_entity_type', 'gt_entity_text']]



    correct_party_name_entity = ws_train_data_agg[
    (ws_train_data_agg['entity_type'] == 2) & (ws_train_data_agg['uuid'].isin(gold_data_agg['uuid']))]

    # Merge correct_party_name_entity with gold_data_agg on uuid and select desired columns
    correct_party_name_entity = correct_party_name_entity.merge(gold_data_agg[['uuid','gt_entity_type', 'gt_entity_text']], on='uuid', how='left')

    # Filter for class 2 again to ensure only relevant data is included
    correct_party_name_entity = correct_party_name_entity[(correct_party_name_entity['entity_type'] == 2)
                                                    & (correct_party_name_entity['gt_entity_type'] == 2)]

    # Select the desired columns
    correct_party_name_entity = correct_party_name_entity[['uuid', 'entity_type', 'entity_text', 'gt_entity_type', 'gt_entity_text']]

    # Filter the ws_data_agg for class 3
    correct_gov_law_entity = ws_train_data_agg[
        (ws_train_data_agg['entity_type'] == 3) & (ws_train_data_agg['uuid'].isin(gold_data_agg['uuid']))]

    # Merge correct_gov_law_entity with gold_data_agg on uuid and select desired columns
    correct_gov_law_entity = correct_gov_law_entity.merge(gold_data_agg[['uuid','gt_entity_type', 'gt_entity_text']], on='uuid', how='left')

    # Filter for class 3 again to ensure only relevant data is included
    correct_gov_law_entity = correct_gov_law_entity[(correct_gov_law_entity['entity_type'] == 3)
                                                    & (correct_gov_law_entity['gt_entity_type'] == 3)]

    # Select the desired columns
    correct_gov_law_entity = correct_gov_law_entity[['uuid', 'entity_type', 'entity_text', 'gt_entity_type', 'gt_entity_text']]

    return correct_doc_name_entity,correct_party_name_entity,correct_gov_law_entity

   

# create a class for similarity score calculation using tfidf and using embedding
class similarity_evaluation:
  def __init__(self):
    self.tfidf_vectorizer = TfidfVectorizer()
    # Load the spaCy model with word vectors (e.g., 'en_core_web_sm')
    self.nlp = spacy.load("en_core_web_lg")

  def calculate_TfIdf(self, string1,string2):
    # Fit and transform the vectorizer on the two strings
    tfidf_matrix = self.tfidf_vectorizer.fit_transform([string1, string2])
    # Calculate the cosine similarity between the two vectors
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    # The similarity score is in cosine_sim[0][1]
    similarity = cosine_sim[0][1]
    return similarity

  def similarity_evaluation_using_tfidf(self, df):
    # receive the dataframe and find similarity between ws_entity_text and gt_entity_text and create a new column of similarity score and append it to the dataframe
    # and store the similarity score of each sample in that column and return the updated dataframe
    tfidf_similarity_score = []
    for index, row in df.iterrows():
      tfidf_similarity_score.append(self.calculate_TfIdf(row['entity_text'], row['gt_entity_text']))
    df['tfidf_similarity_score'] = tfidf_similarity_score
    return df
  
  def similarity_evaluation_using_spacy_embedding(self, df):
    # receive the dataframe and find similarity between ws_entity_text and gt_entity_text and create a new column of similarity score and append it to the dataframe
    # and store the similarity score of each sample in that column and return the updated dataframe
    spacy_similarity_score = []
    for index, row in df.iterrows():
      spacy_similarity_score.append(self.nlp(row['entity_text']).similarity(self.nlp(row['gt_entity_text'])))
    df['embedding_similarity_score'] = spacy_similarity_score
    return df
  


def calculate_TfIdf_similarity_score():

    # get the aggregated data
    correct_doc_name_entity,correct_party_name_entity,correct_gov_law_entity = aggregate()

    evaluate_similarity = similarity_evaluation()
    # find similarity of document name entity text extracted and actual
    document_name_df_tfidf = evaluate_similarity.similarity_evaluation_using_tfidf(correct_doc_name_entity)
    # now find the mean of similarity score column 
    print("Similarity percentage of document_name entity text extracted and actual:", document_name_df_tfidf['tfidf_similarity_score'].mean())
    # save this to a csv file document_name_df
    # document_name_df.to_csv('../output/evaluation results/document_name_results.csv', index=False)
    party_name_df_tfidf = evaluate_similarity.similarity_evaluation_using_tfidf(correct_party_name_entity)
    print("Similarity percentage of parties entity text extracted and actual:", party_name_df_tfidf['tfidf_similarity_score'].mean())
    # save this to a csv file party_name_df
    # party_name_df.to_csv('../output/evaluation results/party_name_results.csv', index=False)
    governing_law_df_tfidf = evaluate_similarity.similarity_evaluation_using_tfidf(correct_gov_law_entity)
    print("Similarity percentage of governing law entity text extracted and actual:", governing_law_df_tfidf['tfidf_similarity_score'].mean())

    # Open a file for appending
    with open('EvaluationResult.txt', 'a') as file:
        # Append content to the file
        file.write(f'Similarity percentage of document_name entity text extracted and actual: {document_name_df_tfidf["tfidf_similarity_score"].mean()}\n\n')
        file.write(f'Similarity percentage of parties entity text extracted and actual: {party_name_df_tfidf["tfidf_similarity_score"].mean()}\n\n')
        file.write(f'Similarity percentage of governing law entity text extracted and actual: {governing_law_df_tfidf["tfidf_similarity_score"].mean()}\n\n')


def calcualte_embedding_similarity_score():
   
    correct_doc_name_entity,correct_party_name_entity,correct_gov_law_entity = aggregate()

    evaluate_similarity = similarity_evaluation()
    # find similarity of document name entity text extracted and actual using spacy embeddings
    document_name_df = evaluate_similarity.similarity_evaluation_using_spacy_embedding(correct_doc_name_entity)
    # now find the mean of similarity score column
    print("Similarity percentage of document_name entity text extracted and actual:", document_name_df['embedding_similarity_score'].mean())
    # save this to a csv file document_name_df
    # document_name_df.to_csv('../output/evaluation results/document_name_results.csv', index=False)
    party_name_df = evaluate_similarity.similarity_evaluation_using_spacy_embedding(correct_party_name_entity)
    print("Similarity percentage of parties entity text extracted and actual:", party_name_df['embedding_similarity_score'].mean())
    # save this to a csv file party_name_df
    # party_name_df.to_csv('../output/evaluation results/party_name_results.csv', index=False)
    governing_law_df = evaluate_similarity.similarity_evaluation_using_spacy_embedding(correct_gov_law_entity)
    print("Similarity percentage of governing law entity text extracted and actual:", governing_law_df['embedding_similarity_score'].mean())
    # save this to a csv file governing_law_df
    # governing_law_df.to_csv('../output/evaluation results/governing_law_results.csv', index=False)
   
    with open('EvaluationResult.txt', 'a') as file:
        # Append content to the file
        file.write(f'Similarity percentage of document_name entity text extracted and actual: {document_name_df["embedding_similarity_score"].mean()}\n\n')
        file.write(f'Similarity percentage of parties entity text extracted and actual: {party_name_df["embedding_similarity_score"].mean()}\n\n')
        file.write(f'Similarity percentage of governing law entity text extracted and actual: {governing_law_df["embedding_similarity_score"].mean()}\n\n')