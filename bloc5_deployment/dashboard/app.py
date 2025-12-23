import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np
import math

### Config
st.set_page_config(
    page_title="Get Around Dashboard",
    layout="wide"
)

# Style
st.markdown("""
<style>

    /* Main title */
    .main-title {
        color: #B322AA;
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0.25rem 0 0.5rem 0;
    }

    /* Headers */
    .section-header {
        color: #B322AA;
        font-size: 2.5rem;
        font-weight: 600;
        margin: 0.6rem 0;
    }
    
    /* Comments */
    .comment-box {
    background-color: #F7F7F7;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 12px;
    padding: 16px;
    margin-top: 0.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    max-width: 100%;
    }

    # Customed CSS for st.metric
    /* Target the main container of st.metric */

    /* Customize the title (label) */
    div[data-testid="stMetric"] label {
        font-size: 18px;
        color: #6b6b6b;
        font-weight: bold;
    }

    /* Customize the main value*/
    div[data-testid="stMetric"] > div > div {
        font-size: 28px;
        color: #3C3732;
        font-weight: bold;
    }

    div[data-testid="stMetricDelta"] {
        font-size: 14px;
        color: #666666;
        font-weight: normal;
        margin-top: -10px;
    }
    
    /* Customize the help text */
    div[data-testid="stMetric"] p {
        font-size: 14px;
        color: #777777;
    }
        
    [data-testid="stMetricDelta"] svg {
    display: none;
    }
    
    div[data-testid="stMetric"] {
        border: 1px solid rgba(179,34,170,0.12) !important;
        border-radius: 10px !important;
        padding: 15px !important;
        background-color: #F6E6F6 !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }
}
</style>
""", unsafe_allow_html=True)


DATA_URL = 'https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/get_around_delay_analysis.xlsx'


### App

@st.cache_data

def load_data():
    data = pd.read_excel(DATA_URL)
    
    data = data.rename(columns={'delay_at_checkout_in_minutes':'delay_at_checkout',
                                'time_delta_with_previous_rental_in_minutes':'time_delta_with_previous_rental'})
    
    # Add delay at checkout for previous ended rental_id
    data_previous = data.copy()
    col_to_keep = ['rental_id', 'car_id','delay_at_checkout']
    data_previous = data_previous[col_to_keep]
    data_previous = data_previous.rename(columns={'rental_id':'previous_ended_rental_id', 
                                            'delay_at_checkout':'previous_delay_at_checkout'})
    data_previous = data_previous.set_index(['previous_ended_rental_id','car_id'])
    data = data.set_index(['previous_ended_rental_id','car_id'])
    data_merged = data.join(data_previous, how='left')
    data_merged = data_merged.reset_index()
    
    # Add indicator of presence of a previous rental
    data_merged['has_previous_rental'] = 0
    data_merged.loc[data_merged['previous_ended_rental_id'].notnull(), 'has_previous_rental'] = 1
    
    # Add checkout status: early or on-time vs late
    data_merged['checkout_status'] = "undefined"
    data_merged.loc[data_merged['delay_at_checkout']<=0, 'checkout_status'] = "early or on-time"
    data_merged.loc[data_merged['delay_at_checkout']>0, 'checkout_status'] = "late"
    
    data_merged['previous_checkout_status'] = "no previous rental"
    data_merged.loc[data_merged['previous_delay_at_checkout']<=0, 'previous_checkout_status'] = "previous rental early or on-time"
    data_merged.loc[data_merged['previous_delay_at_checkout']>0, 'previous_checkout_status'] = "previous rental late"
    
    data_merged['previous_checkout_status_regroup'] = "no previous rental or previous rental early/on-time"
    data_merged.loc[data_merged['previous_delay_at_checkout']>0, 'previous_checkout_status_regroup'] = "previous rental late"
    
    # Add conflict indicator
    data_merged['conflict'] = "undefined"
    mask_late_conflict = (data_merged['previous_checkout_status']=='previous rental late') & (data_merged['previous_delay_at_checkout']>data_merged['time_delta_with_previous_rental'])
    mask_late_no_conflict = (data_merged['previous_checkout_status']=='previous rental late') & (data_merged['previous_delay_at_checkout']<=data_merged['time_delta_with_previous_rental'])
    mask_ontime = (data_merged['previous_checkout_status']=='previous rental early or on-time')
    mask_no_previous_rental = (data_merged['previous_checkout_status']=='no previous rental')
    mask_late_with_undefined_impact = (data_merged['previous_checkout_status']=='previous rental late') & ((data_merged['previous_delay_at_checkout'].isnull()| data_merged['time_delta_with_previous_rental'].isnull()))
    data_merged.loc[mask_late_conflict, 'conflict'] = "conflict with the next rental"
    data_merged.loc[mask_late_no_conflict, 'conflict'] = "no conflict"
    data_merged.loc[mask_ontime, 'conflict'] = "no conflict"
    data_merged.loc[mask_no_previous_rental, 'conflict'] = "no conflict"
    data_merged.loc[mask_late_with_undefined_impact, 'conflict'] = "missing values"

    # Add waiting time: previous delay at checkout - time delta with previous rental
    data_merged['waiting_time'] = np.nan
    data_merged.loc[(data_merged['previous_delay_at_checkout'].notnull()) & (data_merged['time_delta_with_previous_rental'].notnull()), 'waiting_time'] = data_merged['previous_delay_at_checkout'] - data_merged['time_delta_with_previous_rental']
    
    return data_merged

data_load_state = st.text('Loading data...')
df = load_data()
data_load_state.text("")


options = ["All", "Mobile", "Connect"]

### --------- INTRODUCTION ------------- ###

st.markdown('<div class="main-title">Get Around Dashboard</div>', unsafe_allow_html=True)
st.markdown("""
    <p>Late returns at checkout may cause significant friction for the following rental, leading to customer dissatisfaction and/or rental cancellation. </p>
    <p>This dashboard provides insights into:
    <li><b>delay characteristics</b>: frequency and magnitude of delays at checkout;</li>
    <li><b>impact of a late checkout on the following rental and analysis of conflicts</b>;</li>
    <li><b>scenario simulation</b>: effects of implementing of a <b>minimum delay</b> between consecutive car rentals.</li> </p>
""", unsafe_allow_html=True)

st.divider()

### --------- OVERVIEW ------------- ###

st.markdown('<div class="section-header">Overview</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="comment-box">
    The dataset covers a fleet of {df['car_id'].nunique()} cars.
    {len(df)} rentals are registered, with a completion rate of {round(len(df.loc[df['state']=='ended'])/len(df)*100, 0)}%.<br>
    The proportion of Mobile check-ins (80%) is notably greater than Connect check-ins (20%).<br>
</div>
""", unsafe_allow_html=True)


col1, col2 = st.columns([1, 4], gap="small")
with col1:
    checkin_filter = st.selectbox("Select the check-in type", options, key="checkin_filter1") 

if checkin_filter != "All":
    df_filter = df.loc[df['checkin_type'] == checkin_filter.lower()].copy()
else:
    df_filter = df.copy()
    
    
col1, col2, col3 = st.columns([1, 2, 2], gap="small")

with col1:
    nb_rentals = len(df_filter)
    nb_cars = df_filter['car_id'].nunique()
        
    st.metric(
        "Total rentals", 
        f"{nb_rentals:,}",
        help="Number of rentals matching the applied filter"
        )

    st.metric(
        "Total cars", 
        f"{nb_cars:,}",
        help="Number of unique cars matching the applied filter"
    )
        

with col2:
    
    color_map = {
    'canceled': '#C5C5C5', 
    'ended': '#9A1C93',
    }
    fig = px.pie(df_filter, names='state', height=350, color='state', color_discrete_map=color_map)
    fig.update_traces(
        textinfo='percent+label',
        texttemplate='%{label}<br>%{percent:.0%}'
    )

    fig.update_layout(
        title={
            'text': 'Rental state repartition',
            'x': 0.5, 
            'xanchor': 'center',
            'y': 0.95, 
            'yanchor': 'top'
        },
        showlegend=False,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
    
with col3:
    
    color_map = {
    'connect': '#D89ACF', 
    'mobile': '#9A1C93',
    }
    fig = px.pie(df_filter, names='checkin_type', height=350, color='checkin_type', color_discrete_map=color_map)
    fig.update_traces(
        textinfo='percent+label',
        texttemplate='%{label}<br>%{percent:.0%}'
    )

    fig.update_layout(
        title={
            'text': 'Check-in type repartition',
            'x': 0.5, 
            'xanchor': 'center',
            'y': 0.95,
            'yanchor': 'top'
        },
        showlegend=False,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.divider()

### --------- DELAY AT CHECKOUT ------------- ###

df_ended_delay = df.loc[(df['state']=="ended") & (df['delay_at_checkout'].notnull())]
df_ended_delayna = df.loc[(df['state']=="ended") & (df['delay_at_checkout'].isnull())]

# Total completed rentals (which are generating revenue)
df_ended = df.loc[(df['state']=='ended')]
nb_ended = len(df_ended)

nb_rentals_late = len(df_ended_delay.loc[df_ended_delay['checkout_status']=="late"])
proportion_late_vs_global = round(nb_rentals_late/len(df) * 100, 0)
proportion_late_vs_ended = round(nb_rentals_late/len(df_ended_delay)*100, 0)

st.markdown('<div class="section-header">Delay at checkout</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="comment-box">
<p>Overall, 58% of rentals experience delays. Across all rentals, half are delayed by 9 minutes or more.</p>
<p>Mobile users tend to be more late are checkout than Connect ones:</p>
<ul>
<li>A higher proportion of mobile users are late (61%) compared to Connect users (43%).</li>
<li>Half of all mobile users are delayed by 14 minutes or more, whereas half of all Connect users are at least 9 minutes early for checkout.</li>
</ul>
<p style="color:gray"><small>NB: these results apply only to rentals with available checkout delay data. 
Data is missing is missing for {len(df_ended_delayna)} rentals, representing {round(len(df_ended_delayna)/nb_ended*100, 1)}% of completed rentals.</small></p>
</div>
""", unsafe_allow_html=True)


col1, col2 = st.columns([1, 4], gap="small")

with col1:
    checkin_filter = st.selectbox("Select the check-in type", options, key="checkin_filter2") 

if checkin_filter != "All":
    df_filter = df_ended_delay.loc[df_ended_delay['checkin_type'] == checkin_filter.lower()].copy()
else:
    df_filter = df_ended_delay.copy()

delay_median = df_filter['delay_at_checkout'].median()
delay_mean = df_filter['delay_at_checkout'].mean()

col1, col2, col3 = st.columns([1, 0.3, 3.7], gap="small")


with col1:
    
    df_filter_checkout = df_filter.groupby(['checkout_status']).size().reset_index(name='count')
    
    
    color_map = {
        'early or on-time': '#D89ACF', 
        'late': '#9A1C93',
    }
    fig = px.pie(df_filter_checkout, names='checkout_status', values='count', height=350, color='checkout_status', color_discrete_map=color_map)

    fig.update_traces(
        textinfo='percent+label',
        texttemplate='%{label}<br>%{percent:.0%}',
        hovertemplate='%{label}<br>Count: %{value}'  # ← AJOUTEZ CETTE LIGNE
    )
    fig.update_layout(
        title={
            'text': 'Checkout status repartition',
            'x': 0.5, 
            'xanchor': 'center',
            'y': 0.95,
            'yanchor': 'top'
        },
        showlegend=False,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.metric(
        "Median delay at checkout", 
        f"{delay_median:,} min",
    )

with col3:
    fig = px.histogram(
        df_filter, 
        x="delay_at_checkout", 
        marginal="box",
        title=f"Distribution of delay at checkout for ended rentals", 
        color_discrete_sequence=["#9A1C93"],
    )
    fig.add_vline(
        x=delay_median,
        line_dash="dash",
        line_color="#3390FF",
    )
    fig.add_annotation(
        x=delay_median,
        y=1.08, 
        xref="x", 
        yref="paper",
        text="Median delay",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(color="#3390FF")
    )
    fig.update_traces(xbins=dict(start=-360, end=360, size=30), selector=dict(type="histogram"))
    fig.update_xaxes(range=[-360, 360])
    fig.update_layout(
        xaxis_title='Delay at checkout (min)',
        yaxis_title='Number of rentals',
        title={'x': 0.5,'xanchor': 'center',
            'y': 0.95,
            'yanchor': 'top'},
        margin=dict(b=70),
        xaxis=dict(tick0=0,dtick=60, range=[-360, 360])
    )
    fig.add_annotation(
        text="Bin size: 30 minutes",
        x=0.5, xref="paper",
        y=-0.26, yref="paper",
        showarrow=False,
        font=dict(size=12, color="gray"),
        align="center"
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()


### --------- CONSECUTIVE RENTALS ------------- ###

st.markdown('<div class="section-header">Impact of a late checkout on the following rental</div>', unsafe_allow_html=True)

st.markdown("""
<div class="comment-box">
    <p>This section analyzes consecutive rentals which represents a small proportion (8.1%) of all rentals.
    In half of these cases, the previous driver returns the car late, with a median delay of 42 minutes.</p>
    <p>However, a late return does not necessarily have a negative impact on the next rental.
    Conflicts arise only when the previous driver’s return delay exceeds the scheduled interval between rentals. 
    Overall, 218 such conflicts have been recorded, accounting for 1% of all rentals, which is relatively small.</p>
    <p>These conflicts are problematic, as they force the next driver to wait if they choose not to cancel their reservation. 
    In such cases, the median waiting time is 27 minutes.</p>
    <p>The cancellation rate remains comparable (~15% - 17%), regardless of a conflict occurrence.
    Indeed, rentals may be cancelled for various reasons beyond time conflicts with the previous rental.</p>
    <p>See below how the findings vary by check-in type (select an option)</p>
    <p style="color:gray"><small>NB: these results apply only to rentals with available checkout delay data.</small></p>

</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 4], gap="small")

with col1:
    checkin_filter = st.selectbox("Select the check-in type", options, key="checkin_filter4") 

if checkin_filter != "All":
    df_filter = df.loc[df['checkin_type'] == checkin_filter.lower()].copy()
else:
    df_filter = df.copy()

df_consecutive = df_filter.loc[(df_filter['has_previous_rental']==1) & (df_filter['previous_delay_at_checkout'].notnull())].copy()
nb_consecutive = len(df_consecutive)
prct_consecutive = nb_consecutive/len(df)*100

col1, col2, col3 = st.columns([1, 2, 2], gap="small")
with col1:
    st.metric(
        "Number of consecutive rentals", 
        f"{nb_consecutive} ",
        help="Number of consecutive rentals matching the applied filter"
    )

    st.metric(
        "Share of consecutive rentals", 
        f"{round(prct_consecutive, 1):,} %",
        help="Percentage of consecutive rentals matching the applied filter out of the total number of rentals"
    )
    

with col2:

    df_consecutive_agg = df_consecutive.groupby('previous_checkout_status', dropna=False).size().reset_index(name='count')
    
    color_map = {'previous rental early or on-time': '#D89ACF', 'previous rental late': '#9A1C93'}
    fig = px.pie(df_consecutive_agg, 
                names='previous_checkout_status',
                values='count',
                color='previous_checkout_status', 
                color_discrete_map=color_map, 
                height=420)
    fig.update_traces(
    textinfo='percent+label',
    texttemplate='%{label}<br>%{percent:.0%}',  
    insidetextorientation='horizontal',
    textposition='inside'
    )
    fig.update_layout(
        title={
            'text': 'Repartition of previous checkout delay status',
            'x': 0.5, 
            'xanchor': 'center',
            'y': 0.95,
            'yanchor': 'top'
        },
        showlegend=False,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)


with col3:
    df_consecutive_late = df_consecutive.loc[df_consecutive['previous_checkout_status']=="previous rental late"].copy()
    delay_median_consecutive_late = df_consecutive_late['previous_delay_at_checkout'].median()
    fig = px.histogram(
        df_consecutive_late, 
        x="previous_delay_at_checkout", 
        marginal="box",
        title=f"Distribution of checkout delays for late rentals that precede another rental", 
        color_discrete_sequence=["#9A1C93"],
    )
    fig.add_vline(
        x=delay_median_consecutive_late,
        line_dash="dash",
        line_color="#3390FF",
    )
    fig.add_annotation(
        x=delay_median_consecutive_late,
        y=1.08, 
        xref="x", 
        yref="paper",
        text="Median delay",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(color="#3390FF")
    )
    fig.update_traces(xbins=dict(start=-360, end=360, size=30), selector=dict(type="histogram"))    
    fig.update_xaxes(range=[0, 360])
    fig.update_layout(
        xaxis_title='Delay at checkout (min)',
        yaxis_title='Number of rentals',
        title={'x': 0.5,'xanchor': 'center',
            'y': 0.95,
            'yanchor': 'top'},
        margin=dict(b=70),
        xaxis=dict(tick0=0,dtick=60, range=[0, 360])
    )
    st.plotly_chart(fig, use_container_width=True)

mask_conflict = (df_filter['previous_delay_at_checkout'].notnull()) & (df_filter['time_delta_with_previous_rental'].notnull()) & (df_filter['previous_delay_at_checkout']>df_filter['time_delta_with_previous_rental'])
df_filter_conflict = df_filter.loc[mask_conflict]
nb_conflict = len(df_filter_conflict)
prct_conflict = nb_conflict/len(df)*100
median_waiting_time = df_filter_conflict['waiting_time'].median()
df_conflict_agg = df_filter.groupby(['conflict', 'state'], dropna=False).size().reset_index(name='count')
df_conflict_agg['proportion'] = df_conflict_agg['count'] / df_conflict_agg.groupby('conflict')['count'].transform('sum') * 100

col1, col2, col3 = st.columns([1, 2, 2], gap="small")
with col1:
    st.metric(
        "Conflicts", 
        f"{nb_conflict}"
    )

    st.metric(
        "% conflicts", 
        f"{round(prct_conflict, 1):,} %",
        help="Percentage of conflicts matching the applied filter out of the total number of rentals"
    )

    st.metric(
        "Median waiting time if conflict", 
        f"{math.ceil(median_waiting_time)} min",
        help = "Median waiting time for the next driver when the preceding driver is too late"
    )

with col2:
    color_map = {
    'ended': '#D89ACF', 
    'canceled': '#9A1C93',
    }
    
    fig = px.bar(df_conflict_agg,
                x='conflict',
                y='proportion',
                color='state',
                title='Canceled vs completed rentals by conflict status',
                barmode='stack',
                text=df_conflict_agg.apply(lambda row: f"{row['state']}: {row['proportion']:.0f}%", axis=1),
                hover_data={'count': True},
                color_discrete_map=color_map,
                #height=450
                )

    fig.update_layout(
        xaxis_title='Conflict status',
        yaxis_title='Proportion of rentals',
        showlegend=False,
        title={'x': 0.5,'xanchor': 'center',
            'y': 0.95,
            'yanchor': 'top'},
        margin=dict(l=20, r=20, t=50, b=20)
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>State: %{fullData.name}<br>Proportion: %{y:.0f}%<br>Count: %{customdata[0]}<extra></extra>",
        textposition='inside',
        insidetextanchor='middle'
    )
    st.plotly_chart(fig, use_container_width=True)

with col3:
    fig = px.histogram(
        df_filter_conflict, 
        x="waiting_time", 
        marginal="box",
        title=f"Distribution of next driver waiting time in case of conflict", 
        color_discrete_sequence=["#9A1C93"],
    )
    fig.add_vline(
        x=median_waiting_time,
        line_dash="dash",
        line_color="#3390FF",
    )
    fig.add_annotation(
        x=median_waiting_time,
        y=1.08, 
        xref="x", 
        yref="paper",
        text="Median delay",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(color="#3390FF")
    )
    fig.update_traces(xbins=dict(start=0, end=360, size=30), selector=dict(type="histogram"))
    fig.update_xaxes(range=[0, 360])
    fig.update_layout(
        xaxis_title='Waiting time (min)',
        yaxis_title='Number of rentals',
        title={'x': 0.5,'xanchor': 'center',
            'y': 0.95,
            'yanchor': 'top'},
        margin=dict(b=70),
        xaxis=dict(tick0=0,dtick=60, range=[0, 360])
    )
    fig.add_annotation(
        text="Bin size: 30 minutes",
        x=0.5, xref="paper",
        y=-0.26, yref="paper",
        showarrow=False,
        font=dict(size=12, color="gray"),
        align="center"
    )

    st.plotly_chart(fig, use_container_width=True)
    

st.divider()

### --------- SIMULATION OF MIN DELAY BW RENTALS ------------- ###
st.markdown('<div class="section-header">Scenario analysis: buffer time between rentals</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="comment-box">
    <p>Hover over the charts to display the impact of different thresholds for minimum delay between rentals on:
    <li> the number of rental offers affected (i.e. rental offers hidden from search results) and their proportion among all rentals.</li>
    <li> the number and the proportion of conflicts solved.</li>
    <li> the number and the proportion of cancellations which could be avoided.</li>
    <li> the proportion of revenue which could be lost, roughly approximated by the proportion of ended rentals which are hidden from search results among all ended rentals.</li>
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="comment-box">
    <p>Implementing a minimum delay between rentals leads to a revenue loss, but reduces conflictual cases and customer dissatisfaction. </p>
    <p>A <b>30-minute threshold for all check-in types</b> seems interesting:</p>
    <li> GetAround excludes 1.2% of rental offers from search results, sacrificing revenue from 1.4% of potential rentals.</li>
    <li> However this measure reduces conflicts by nearly half and prevents 1.1% of cancellations.</li>
    <li> The ratio of avoided conflicts to hidden offers is most favorable at this threshold. 
    Beyond 30 minutes, the trade-off weakens: the Get Around platform hides substantially more offers than it resolves conflicts.</li>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 4], gap="small")

with col1:
    checkin_filter = st.selectbox("Select the check-in type", options, key="checkin_filter3") 

if checkin_filter != "All":
    df_filter = df.loc[df['checkin_type'] == checkin_filter.lower()].copy()
else:
    df_filter = df.copy()


# Total conflictual cases
mask_conflict_total = (df['previous_delay_at_checkout'].notnull()) & (df['time_delta_with_previous_rental'].notnull()) & (df['previous_delay_at_checkout']>df['time_delta_with_previous_rental'])
nb_conflict = len(df.loc[mask_conflict_total])  # total number of conflicts

# Total canceled rentals
nb_canceled = len(df.loc[df['state']=='canceled'])

# The feature has an impact on consecutive rentals (rentals preceded by another rental)
# Filter on consecutive rentals with non-missing previous delay at checkout
df_hasprevious = df_filter.loc[(df_filter['has_previous_rental']==1) & (df_filter['previous_delay_at_checkout'].notnull())]
nb_total = len(df_hasprevious)

# Conflictual situations with filter
mask_conflict = (df_filter['previous_delay_at_checkout'].notnull()) & (df_filter['time_delta_with_previous_rental'].notnull()) & (df_filter['previous_delay_at_checkout']>df_filter['time_delta_with_previous_rental'])
df_conflict = df_filter.loc[mask_conflict]

# Canceled rentals which are preceded by another rental
df_canceled_previous = df_filter.loc[(df_filter['state']=='canceled') & (df_filter['has_previous_rental']==1)]

# Ended rentals which are preceded by another rental
df_ended_previous = df_filter.loc[(df_filter['state']=='ended') & (df_filter['has_previous_rental']==1)]
nb_ended_previous = len(df_ended_previous)

threshold_range = sorted(df['time_delta_with_previous_rental'].dropna().unique())

results = []

delta = df_ended['time_delta_with_previous_rental']
previous_delay = df_ended['previous_delay_at_checkout']

for threshold in threshold_range:
    
    count_blocked = ((df_hasprevious['time_delta_with_previous_rental'].notnull()) & (df_hasprevious['time_delta_with_previous_rental']<threshold)).sum()
    count_solved = (df_conflict['time_delta_with_previous_rental']<threshold).sum()
    count_blocked_ended = (df_ended_previous['time_delta_with_previous_rental']<threshold).sum()
    count_blocked_canceled = (df_canceled_previous['time_delta_with_previous_rental']<threshold).sum()

    prct_blocked = count_blocked/len(df)*100
    prct_solved = count_solved/nb_conflict*100
    prct_blocked_canceled = count_blocked_canceled/nb_canceled*100
    prct_revenue_loss = count_blocked_ended/nb_ended*100
    
    results.append({
        'threshold': threshold,
        'nb_blocked': count_blocked,
        'nb_solved': count_solved,
        'nb_blocked_canceled': count_blocked_canceled,
        'nb_blocked_ended': count_blocked_ended,
        'prct_blocked': prct_blocked,
        'prct_solved':prct_solved,
        'prct_blocked_canceled':prct_blocked_canceled,
        'prct_revenue_loss':prct_revenue_loss
    })

df_results = pd.DataFrame(results)

x = df_results['threshold']
y1 = df_results['nb_blocked']
y2 = df_results['nb_solved']
y3 = df_results['nb_blocked_canceled']
prct1 = df_results['prct_blocked']
prct2 = df_results['prct_solved']
prct3 = df_results['prct_blocked_canceled']

hover_prct1 = prct1.values.reshape(-1, 1)
hover_prct2 = prct2.values.reshape(-1, 1)
hover_prct3 = prct3.values.reshape(-1, 1)

col1, col2 = st.columns([1, 1], gap="small")
with col1:
        
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y1,
            mode='lines+markers',
            name='Rental offers hidden from search results',
            line=dict(color='#D52B1E'),
            customdata=hover_prct1,
            hovertemplate=(
                "Nb rental offers hidden: %{y:,}<br>" +
                "% rental offers hidden: %{customdata[0]:.1f}%<extra></extra><br>" 
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y2,
            mode='lines+markers',
            name='Conflicts solved',
            line=dict(color='#00997C'),
            customdata=hover_prct2,
            hovertemplate=(
                "Nb conflicts solved: %{y:,}<br>"
                +
                "% conflicts solved: %{customdata[0]:.1f}%<extra></extra><br>"
            )
        )
    )


    fig.add_trace(
        go.Scatter(
            x=x,
            y=y3,
            mode='lines+markers',
            name='Cancellations avoided',
            line=dict(color='#C5C5C5'),
            customdata=hover_prct3,
            hovertemplate=(
                "Nb cancellations avoided: %{y:,}<br>"
                +
                "% cancellations avoided: %{customdata[0]:.1f}%<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title={
            'text': "Impact of a minimum delay threshold",
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.95,
            'yanchor': 'top'
        },
        xaxis_title='Minimum delay threshold (min)',
        yaxis_title='Number of cases',
        margin=dict(b=90),
        hovermode='x unified',
        height=550,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5
    )
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Revenue loss
    fig = px.line(
        df_results,
        x="threshold",
        y="prct_revenue_loss",
        labels={"threshold": "Minimum delay threshold", "prct_revenue_loss": "%"},
        markers=True,
        color_discrete_sequence=["#9A1C93"],
        height=550
    )

    fig.update_traces(
        hovertemplate="Threshold: %{x} min<br>% revenue lost: %{y:.1f}%<extra></extra>"
    )

    fig.update_layout(
        title={
            'text': "Potential revenue loss by threshold",
            'x': 0.5,
            'xanchor': 'center',
            'y': 0.95,
            'yanchor': 'top'
        },
        xaxis_title='Minimum delay threshold (min)',
        yaxis_title='%',
        margin=dict(b=90),
        height=550,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)



### Footer 
empty_space, footer = st.columns([0.85, 0.15])

with empty_space:
    st.write("")

with footer:
    st.markdown("Built with [Streamlit](https://streamlit.io)")