### Load JSON ###

import pandas as pd
from IPython.display import display  # Import display function correctly

# Load JSON 
df = pd.read_json("group_WVC_stem_encoding.json", lines=True)

### Annotate Response Hypotheses ###

# Step 1: Consolidate all mappings in a dictionary keyed by broad_qid
all_mappings = {
    'Q8': {
        1: "it is important for children to be independent.",
        2: "it is not important for children to be independent."
    },
    'Q11': {
        1: "it is important for children to be imaginative.",
        2: "it is not important for children to be imaginative."
    },
    'Q17': {
        1: "it is important for children to be obedient.",
        2: "it is not important for children to be obedient."
    },
    'Q171': {
        1: "frequent attendance at religious services.",
        2: "regular attendance at religious services.",
        3: "occasional attendance at religious services.",
        4: "only attendance at religious services on special holy days.",
        5: "rare attendance at religious services.",
        6: "very rare attendance at religious services.",
        7: "no attendance or negligible attendance at religious services."
    },
    'Q173': {
        1: "religiosity is an inseparable part of one's identity.",
        2: "religiosity is not a central part of one's identity.",
        3: "religiosity cannot be considered a part of one's identity."
    },
    'Q178': {
        1: "avoiding a fare on public transportation is never justifiable.",
        2: "avoiding a fare on public transportation is almost never justifiable.",
        3: "avoiding a fare on public transportation is rarely justifiable.",
        4: "avoiding a fare on public transportation is seldom justifiable.",
        5: "avoiding a fare on public transportation is sometimes justifiable.",
        6: "avoiding a fare on public transportation is occasionally justifiable.",
        7: "avoiding a fare on public transportation is somewhat justifiable.",
        8: "avoiding a fare on public transportation is mostly justifiable.",
        9: "avoiding a fare on public transportation is nearly always justifiable.",
        10: "avoiding a fare on public transportation is eminently justifiable."
    },
    'Q180': {
        10: "cheating on one's taxes is always justifiable.",
        9: "cheating on one's taxes is almost always justifiable.",
        8: "cheating on one's taxes is often justifiable.",
        7: "cheating on one's taxes is sometimes justifiable.",
        6: "cheating on one's taxes is occasionally justifiable.",
        5: "cheating on one's taxes is rarely justifiable.",
        4: "cheating on one's taxes is seldom justifiable.",
        3: "cheating on one's taxes is hardly ever justifiable.",
        2: "cheating on one's taxes is almost never justifiable.",
        1: "cheating on one's taxes is never justifiable."
    },
    'Q181': {
        10: "accepting a bribe in the course of one's duties is perfectly justifiable.",
        9: "accepting a bribe in the course of one's duties is almost always justifiable.",
        8: "accepting a bribe in the course of one's duties is often justifiable.",
        7: "accepting a bribe in the course of one's duties is sometimes justifiable.",
        6: "accepting a bribe in the course of one's duties is occasionally justifiable.",
        5: "accepting a bribe in the course of one's duties is rarely justifiable.",
        4: "accepting a bribe in the course of one's duties is seldom justifiable.",
        3: "accepting a bribe in the course of one's duties is hardly ever justifiable.",
        2: "accepting a bribe in the course of one's duties is almost never justifiable.",
        1: "accepting a bribe in the course of one's duties is never justifiable."
    },
    'Q182': {
        10: "homosexuality is completely justifiable.",
        9: "homosexuality is almost always justifiable.",
        8: "homosexuality is often justifiable.",
        7: "homosexuality is sometimes justifiable.",
        6: "homosexuality is occasionally justifiable.",
        5: "homosexuality is rarely justifiable.",
        4: "homosexuality is seldom justifiable.",
        3: "homosexuality is hardly ever justifiable.",
        2: "homosexuality is almost never justifiable.",
        1: "homosexuality is never justifiable."
    },
    'Q184': {
        10: "abortion is easily justifiable.",
        9: "abortion is almost always justifiable.",
        8: "abortion is often justifiable.",
        7: "abortion is sometimes justifiable.",
        6: "abortion is occasionally justifiable.",
        5: "abortion is rarely justifiable.",
        4: "abortion is seldom justifiable.",
        3: "abortion is hardly ever justifiable.",
        2: "abortion is almost never justifiable.",
        1: "abortion is never justifiable."
    },
    'Q185': {
        10: "divorce is completely justifiable.",
        9: "divorce is almost always justifiable.",
        8: "divorce is often justifiable.",
        7: "divorce is sometimes justifiable.",
        6: "divorce is occasionally justifiable.",
        5: "divorce is rarely justifiable.",
        4: "divorce is seldom justifiable.",
        3: "divorce is hardly ever justifiable.",
        2: "divorce is almost never justifiable.",
        1: "divorce is never justifiable."
    },
    'Q254': {
        1: "one possesses very strong national pride.",
        2: "one possesses quite strong national pride.",
        3: "one possesses minimal national pride.",
        4: "one possesses no national pride."
    },
    'Q27': {
        1: "making one's parents proud is of central importance in life.",
        2: "making one's parents proud is important in life.",
        3: "making one's parents proud is somewhat unimportant in life.",
        4: "making one's parents proud is not important in life."
    },
    'Q29': {
        1: "men always make better political leaders than women do.",
        2: "men generally make better political leaders than women do.",
        3: "men do not necessarily make better political leaders than women do.",
        4: "men do not make better political leaders than women do."
    },
    'Q30': {
        1: "university or higher education is more important for a boy than for a girl.",
        2: "university or higher education is generally more important for a boy than for a girl.",
        3: "university or higher education is not necessarily more important for a boy than for a girl.",
        4: "university or higher education is not more important for a boy than for a girl."
    },
    'Q33_3': {
        1: "under employment scarcity, men should have more right to a job than women.",
        2: "under employment scarcity, men should not have more right to a job than women.",
        3: "neither agree nor disagree about men having more right to a job than women under employment scarcity."
    },
    'Q45': {
        1: "in the future, people should show greater respect for authority.",
        2: "in the future, people showing greater respect for authority would be neither a good nor a bad thing.",
        3: "in the future, people showing greater respect for authority would be a bad thing."
    },
    'Q6': {
        1: "religion is very important in life.",
        2: "religion is rather important in life.",
        3: "religion is not very important in life.",
        4: "religion is not at all important in life."
    },
    'Q65': {
        1: "one has a great deal of confidence in the armed forces.",
        2: "one has quite a lot of confidence in the armed forces.",
        3: "one has some confidence in the armed forces.",
        4: "one has negligible or no confidence in the armed forces."
    },
    'Q69': {
        1: "one has a great deal of confidence in the police.",
        2: "one has quite a lot of confidence in the police.",
        3: "one has some confidence in the police.",
        4: "one has little or no confidence in the police."
    },
    'Q70': {
        1: "one has a great deal of confidence in the justice system.",
        2: "one has quite a lot of confidence in the justice system.",
        3: "one has some confidence in the justice system.",
        4: "one has little or no confidence in the justice system."
    },
    'Q153': {
        1: "Over the coming years, the government should also emphasize a high level of economic growth.",
        2: "Over the coming years, the government should also prioritize ensuring the country has strong defense forces.",
        3: "Over the coming years, the government should also focus on ensuring that people have more say about how things are done at their jobs and in their communities.",
        4: "Over the coming years, the government should also prioritize work to make the nation's cities and countryside more beautiful."
    },
    'Q155': {
        1: "Maintaining order in the nation is an additional priority.",
        2: "Giving people more say in important government decisions is an additional priority.",
        3: "Fighting rising prices is an additional priority.",
        4: "Protecting freedom of speech is an additional priority."
    }
}

# Normalize 'likert_scale' to int for consistent mapping
df['likert_scale'] = df['likert_scale'].astype(int)

# Apply mappings for each question in a loop
for qid, mapping in all_mappings.items():
    mask = df['broad_qid'] == qid
    df.loc[mask, 'response_hypothesis'] = df.loc[mask, 'likert_scale'].map(mapping)

# List all question IDs for verification
all_qids = list(all_mappings.keys())

### Save changes to JSON and CSV ###

import pandas as pd
from IPython.display import display

# Add 5 empty adapted_hypothesis columns
for i in range(1, 6):
    col = f"adapted_hypothesis_{i}"
    if col not in df.columns:
        df[col] = ""

# Save to JSON and CSV
df.to_json("group_WVC_stem_encoding.json", orient='records', lines=True)
df.to_csv("group_WVC_stem_encoding_complete.csv", index=False)

# Optional: Display updated DataFrame
#
display(df.head(5))