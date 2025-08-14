import pandas as pd
import re
import os


# Your WVS questions list
wvs_questions = [
    {
        "qid": "Q6",
        "question": "Q6 Important in life: Religion For each of the following aspects, indicate how important it is in your life. Would you say it is very important, rather important, not very important or not important at all? – Religion",
        "scale": "1.- Very important 2.- Rather important 3.- Not very important 4.- Not at all important"
    },
    {
        "qid": "Q8",
        "question": "Q8 Important child qualities: Independence  Here is a list of qualities that children can be encouraged to learn at home. Which, if any, do you consider to be especially important? Please choose up to five. – Independence",
        "scale": "1.- Important 2.- Not mentioned "
    },
    {
        "qid": "Q11",
        "question": "Q11 Important child qualities: Imagination Here is a list of qualities that children can be encouraged to learn at home. Which, if any, do you consider to be especially important? Please choose up to five. – Imagination",
        "scale": "1.- Important 2.- Not mentioned"
    },
    {
        "qid": "Q17",
        "question": "Q17 Important child qualities: Obedience Here is a list of qualities that children can be encouraged to learn at home. Which, if any, do you consider to be especially important? Please choose up to five. – Obedience",
        "scale": "1.- Important 2.- Not mentioned"
    },
    {
        "qid": "Q27",
        "question": "Q27 One of main goals in life has been to make my parents proud For each of the following statements I read out, can you tell me how much you agree with each. Do you agree strongly, agree, disagree, or disagree strongly? - One of my main goals in life has been to make my parents proud",
        "scale": "1.- Agree strongly 2.- Agree 3.- Disagree 4.- Strongly disagree"
    },
    {
        "qid": "Q29",
        "question": "Q29 Men make better political leaders than women do For each of the following statements I read out, can you tell me how much you agree with each. Do you agree strongly, agree, disagree, or disagree strongly? - On the whole, men make better political leaders than women do",
        "scale": "1.- Agree strongly 2.- Agree 3.- Disagree 4.- Strongly disagree"
    },
    {
        "qid": "Q30",
        "question": "Q30 University is more important for a boy than for a girl For each of the following statements I read out, can you tell me how much you agree with each. Do you agree strongly, agree, disagree, or disagree strongly? - A university education is more important for a boy than for a girl",
        "scale": "1.- Agree strongly 2.- Agree 3.- Disagree 4.- Strongly disagree"
    },
    {
        "qid": "Q33_3",
        "question": "Q33_3 Jobs scarce: Men should have more right to a job than women Do you agree, disagree or neither agree nor disagree with the following statements? - When jobs are scarce, men should have more right to a job than women",
        "scale": "1.- Agree 2.- Disagree 3.- Neither agree nor disagree"
    },
    {
        "qid": "Q45",
        "question": "Q45 Future changes: Greater respect for authority I'm going to read out a list of various changes in our way of life that might take place in the near future. Please tell me for each one, if it were to happen, whether you think it would be a good thing, a bad thing, or don't you mind? – Greater respect for authority",
        "scale": "1.- Good thing 2.- Don't mind 3.- Bad thing"
    },
    {
        "qid": "Q65",
        "question": "Q65 Confidence: Armed Forces I am going to name a number of organizations. For each one, could you tell me how much confidence you have in them: is it a great deal of confidence, quite a lot of confidence, not very much confidence or none at all? The armed forces",
        "scale": "1.- A great deal 2.- Quite a lot 3.- Not very much 4.- None at all"
    },
    {
        "qid": "Q69",
        "question": "Q69 Confidence: The Police I am going to name a number of organizations. For each one, could you tell me how much confidence you have in them: is it a great deal of confidence, quite a lot of confidence, not very much confidence or none at all? The police",
        "scale": "1.- A great deal 2.- Quite a lot 3.- Not very much 4.- None at all"
    },
    {
        "qid": "Q70",
        "question": "Q70 Confidence: Justice System/Courts I am going to name a number of organizations. For each one, could you tell me how much confidence you have in them: is it a great deal of confidence, quite a lot of confidence, not very much confidence or none at all? The courts",
        "scale": "1.- A great deal 2.- Quite a lot 3.- Not very much 4.- None at all"
    },
    {
        "qid": "Q152",
        "question": "Q152 Aims of country: first choice People sometimes talk about what the aims of this country should be for the next ten years. On this card are listed some of the goals which different people would give top priority. Would you please say which one of these you consider the most important?",
        "scale": "1.- A high level of economic growth 2.- Strong defence forces 3.- People have more say about how things are done 4.- Trying to make our cities and countryside more beautiful"
    },
    {
        "qid": "Q153",
        "question": "Q153 Aims of country: second choice People sometimes talk about what the aims of this country should be for the next ten years. On this card are listed some of the goals which different people would give top priority. Would you please say which one of these you consider the second most important?",
        "scale": "1.- A high level of economic growth 2.- Making sure this country has strong defence forces 3.- Seeing that people have more say about how are done at their jobs and in their communities 4.- Trying to make our cities and countryside more beautiful"
    },
    {
        "qid": "Q154",
        "question": "Q154 Aims of respondent: first choice If you had to choose, which one of the things on this card would you say is most important?",
        "scale": "1.- Maintaining order in the nation 2.- Giving people more say in important government decisions 3.- Fighting rising prices 4.- Protecting freedom of speech"
    },
    {
        "qid": "Q155",
        "question": "Q155 Aims of respondent: second choice And which would be the next most important?",
        "scale": "1.- Maintaining order in the nation 2.- Giving people more say in important government decisions 3.- Fighting rising prices 4.- Protecting freedom of speech"
    },
    {
        "qid": "Q171",
        "question": "Q171 How often do you attend religious services Apart from weddings, funerals and christenings, about how often do you attend religious services these days?",
        "scale": "1.- More than once a week 2.- Once a week 3.- Once a month 4.- Only on special holy days 5.- Once a year 6.- Less often 7.- Never, practically never"
    },
    {
        "qid": "Q173",
        "question": "Q173 Religious person Independently of whether you go to church or not, would you say you are…",
        "scale": "1.- A religious person 2.- Not a religious person 3.- An atheist"
    },
    {
        "qid": "Q178",
        "question": "Q178 Justifiable: Avoiding a fare on public transport Please tell me for each of the following statements whether you think it can always be justified, never be justified, or something in between, using this card. - Avoiding a fare on public transport",
        "scale": "1.- Never justifiable 2.- 2 3.- 3 4.- 4 5.- 5 6.- 6 7.- 7 8.- 8 9.- 9 10.- Always justifiable"
    },
    {
        "qid": "Q180",
        "question": "Q180 Justifiable: Cheating on taxes Please tell me for each of the following statements whether you think it can always be justified, never be justified, or something in between, using this card. Cheating on taxes if you have a chance",
        "scale": "1.- Never justifiable 2.- 2 3.- 3 4.- 4 5.- 5 6.- 6 7.- 7 8.- 8 9.- 9 10.- Always justifiable"
    },
    {
        "qid": "Q181",
        "question": "Q181 Justifiable: Someone accepting a bribe in the course of their duties Please tell me for each of the following statements whether you think it can always be justified, never be justified, or something in between, using this card. Someone accepting a bribe in the course of their duties",
        "scale": "1.- Never justifiable 2.- 2 3.- 3 4.- 4 5.- 5 6.- 6 7.- 7 8.- 8 9.- 9 10.- Always justifiable"
    },
    {
        "qid": "Q182",
        "question": "Q182 Justifiable: Homosexuality Please tell me for each of the following statements whether you think it can always be justified, never be justified, or something in between, using this card. Homosexuality",
        "scale": "1.- Never justifiable 2.- 2 3.- 3 4.- 4 5.- 5 6.- 6 7.- 7 8.- 8 9.- 9 10.- Always justifiable"
    },
    {
        "qid": "Q184",
        "question": "Q184 Justifiable: Abortion Please tell me for each of the following statements whether you think it can always be justified, never be justified, or something in between, using this card. Abortion",
        "scale": "1.- Never justifiable 2.- 2 3.- 3 4.- 4 5.- 5 6.- 6 7.- 7 8.- 8 9.- 9 10.- Always justifiable"
    },
    {
        "qid": "Q185",
        "question": "Q185 Justifiable: Divorce Please tell me for each of the following statements whether you think it can always be justified, never be justified, or something in between, using this card. Divorce",
        "scale": "1.- Never justifiable 2.- 2 3.- 3 4.- 4 5.- 5 6.- 6 7.- 7 8.- 8 9.- 9 10.- Always justifiable"
    },
    {
        "qid": "Q254",
        "question": "Q254 National pride How proud are you to be of nationality of this country?",
        "scale": "1.- Very proud 2.- Quite proud 3.- Not very proud 4.- Not at all proud"
    },
]

# Map broad QID → question text (without leading QID prefix)
def clean_question_text(qtext):
    return re.sub(r"^Q\d+(_\d+)?\s+", "", qtext)

qid_to_question = {q['qid']: clean_question_text(q['question']) for q in wvs_questions}
qid_to_scale = {q['qid']: q['scale'] for q in wvs_questions}

# Input/output paths
input_path = r"C:/Users/secki/OneDrive/Desktop/MY498 Capstone Under Supervision/WV Corpus/2024.lrec-main.1195.OptionalSupplementaryMaterial/RVR_SUPPLEMENT_COLING_2024/data/WVC/WVC_stem_encoding.csv"
output_path = r"C:/Users/secki/OneDrive/Desktop/MY498 Capstone Under Supervision/UNGA Speech Corpus/Jankin Full Dataset/capstone_pipeline/group_WVC_stem_encoding.csv"

# Exact QIDs to keep (includes subquestions and variants)
wvc_qids_to_keep = [
    "Q6","Q8","Q11","Q17","Q27","Q29","Q30","Q33","Q33_3","Q45","Q65","Q69",
    "Q70","Q152-3_aesthetic","Q152-3_defense","Q152-3_growth","Q152-3_individual_voice",
    "Q154-5_order","Q154-5_price","Q154-5_speech","Q154-5_voice",
    "Q171","Q173","Q178","Q180","Q181","Q182","Q184","Q185","Q254"
]

# Mapping for special subquestions to broad_qid + response_code
subquestion_map = {
    "Q33": ("Q33_3", None),  # We'll handle responses from scale
    "Q33_3": ("Q33_3", None),
    "Q152-3_aesthetic": ("Q152", "4"),
    "Q152-3_defense": ("Q152", "2"),
    "Q152-3_growth": ("Q152", "1"),
    "Q152-3_individual_voice": ("Q152", "3"),
    "Q154-5_order": ("Q154", "1"),
    "Q154-5_price": ("Q154", "3"),
    "Q154-5_speech": ("Q154", "4"),
    "Q154-5_voice": ("Q154", "2"),
}

# Read CSV
df = pd.read_csv(input_path, dtype=str)

# Filter rows by exact wvc_qid match
df = df[df['wvc_qid'].isin(wvc_qids_to_keep)].copy()

# Map wvc_qid to broad_qid and fixed response_code if applicable
def map_broad_qid_and_response_code(wvc_qid):
    if wvc_qid in subquestion_map:
        return subquestion_map[wvc_qid]
    # Else broad_qid is prefix before any dash or underscore (e.g., Q45)
    broad = re.match(r"Q\d+", wvc_qid)
    return (broad.group(0) if broad else wvc_qid, None)

df[['broad_qid', 'fixed_response_code']] = df['wvc_qid'].apply(lambda x: pd.Series(map_broad_qid_and_response_code(x)))

# Map broad_qid → question_text from manual list
df['question_text'] = df['broad_qid'].map(qid_to_question)

# Parse scales to get possible responses per broad_qid
def parse_scale(scale_str):
    if pd.isna(scale_str) or not scale_str.strip():
        return []
    pattern = re.compile(r"(\d+)\.-\s*([^0-9]+)")
    return [(m.group(1), m.group(2).strip()) for m in pattern.finditer(scale_str)]

# Explode rows for each likert scale response
rows = []
for _, row in df.iterrows():
    broad_qid = row['broad_qid']
    scale_options = parse_scale(qid_to_scale.get(broad_qid, ""))
    fixed_resp_code = row['fixed_response_code']
    
    # If fixed_response_code is set (subquestion), only keep that response_code option:
    if fixed_resp_code is not None:
        # Find matching scale text
        response_text = next((text for code, text in scale_options if code == fixed_resp_code), "")
        rows.append({
            "broad_qid": broad_qid,
            "question_text": row['question_text'],
            "original_wvc_hypothesis": row['original_wvc_hypothesis'],
            "hypothesis_recoding_action": row.get('hypothesis_recoding_action', ""),
            "recoded_wvc_hypothesis_stem": row['recoded_wvc_hypothesis_stem'],
            "likert_scale": fixed_resp_code,
            "response_text": response_text,
            "response_hypothesis": ""
        })
    else:
        # Explode all scale options (for usual questions)
        if scale_options:
            for code, response_text in scale_options:
                rows.append({
                    "broad_qid": broad_qid,
                    "question_text": row['question_text'],
                    "original_wvc_hypothesis": row['original_wvc_hypothesis'],
                    "hypothesis_recoding_action": row.get('hypothesis_recoding_action', ""),
                    "recoded_wvc_hypothesis_stem": row['recoded_wvc_hypothesis_stem'],
                    "likert_scale": code,
                    "response_text": response_text,
                    "response_hypothesis": ""
                })
        else:
            # No scale options, create one empty row
            rows.append({
                "broad_qid": broad_qid,
                "question_text": row['question_text'],
                "original_wvc_hypothesis": row['original_wvc_hypothesis'],
                "hypothesis_recoding_action": row.get('hypothesis_recoding_action', ""),
                "recoded_wvc_hypothesis_stem": row['recoded_wvc_hypothesis_stem'],
                "likert_scale": "",
                "response_text": "",
                "response_hypothesis": ""
            })

expanded_df = pd.DataFrame(rows)

final_columns = [
    "broad_qid",
    "question_text",
    "original_wvc_hypothesis",
    "hypothesis_recoding_action",
    "recoded_wvc_hypothesis_stem",
    "likert_scale",
    "response_text",
    "response_hypothesis"
]

expanded_df = expanded_df[final_columns]

# For rows where broad_qid is Q152 or Q154, copy recoded_wvc_hypothesis_stem into response_hypothesis
mask = expanded_df['broad_qid'].isin(['Q152', 'Q154'])
expanded_df.loc[mask, 'response_hypothesis'] = expanded_df.loc[mask, 'recoded_wvc_hypothesis_stem']

# --- Create and append Q153 and Q155 rows from wvs_questions ---

def create_rows_for_qid(qid):
    question_text = qid_to_question.get(qid, "")
    scale_options = parse_scale(qid_to_scale.get(qid, ""))
    rows = []
    for code, response_text in scale_options:
        rows.append({
            "broad_qid": qid,
            "question_text": question_text,
            "original_wvc_hypothesis": "",
            "hypothesis_recoding_action": "",
            "recoded_wvc_hypothesis_stem": "",
            "likert_scale": code,
            "response_text": response_text,
            "response_hypothesis": ""
        })
    return pd.DataFrame(rows)

# Create DataFrames for Q153 and Q155
df_q153 = create_rows_for_qid("Q153")
df_q155 = create_rows_for_qid("Q155")

# Append these to expanded_df
expanded_df = pd.concat([expanded_df, df_q153, df_q155], ignore_index=True)

# --- End append Q153/Q155 ---

#Save as csv and json
# Get JSON output path by replacing '.csv' with '.json'
json_output_path = os.path.splitext(output_path)[0] + ".json"

# Save JSON with records, one per line
expanded_df.to_json(json_output_path, orient="records", lines=True)

expanded_df.to_csv(output_path, index=False)

print(f"Filtered, mapped, and expanded dataset saved to:\n{output_path}")