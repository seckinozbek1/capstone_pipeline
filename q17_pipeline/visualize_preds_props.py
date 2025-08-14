import pandas as pd
import plotly.express as px

df = pd.read_csv("predictions/q17_predictions_filtered.csv")

q17_labels = ['Q17_1', 'Q17_2']
df_q17 = df[df['predicted_combined_label'].isin(q17_labels)]

# Count rows per country and label (sentence counts)
agg = df_q17.groupby(['B_COUNTRY_ALPHA', 'predicted_combined_label']).size().reset_index(name='sentence_count')

# Total sentences per country (all Q17)
total_counts = agg.groupby('B_COUNTRY_ALPHA')['sentence_count'].sum().reset_index(name='total_q17_count')

agg = agg.merge(total_counts, on='B_COUNTRY_ALPHA')

agg['proportion'] = agg['sentence_count'] / agg['total_q17_count']

# Create explicit copy
agg_q17_1 = agg[agg['predicted_combined_label'] == 'Q17_1'].copy()

agg_q17_1['B_COUNTRY_ALPHA'] = pd.Categorical(
    agg_q17_1['B_COUNTRY_ALPHA'],
    categories=sorted(agg_q17_1['B_COUNTRY_ALPHA'].unique()),
    ordered=True
)

fig = px.scatter(
    agg_q17_1,
    x='B_COUNTRY_ALPHA',
    y='proportion',
    size='sentence_count',
    color='B_COUNTRY_ALPHA',
    hover_name='B_COUNTRY_ALPHA',
    labels={'B_COUNTRY_ALPHA': 'Country', 'proportion': 'Proportion of Q17_1', 'sentence_count': 'Sentence Count'},
    title='Proportion of Q17_1 Sentences Relative to Total Q17 Sentences by Country',
    size_max=40,
    height=600
)

fig.update_layout(
    xaxis={'categoryorder': 'array', 'categoryarray': sorted(agg_q17_1['B_COUNTRY_ALPHA'].unique())},
    legend_title_text='Country',
    margin=dict(l=40, r=40, t=80, b=40)
)

fig.show()
