import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from textblob import TextBlob
import helper
import preprocessor


def categorize_sentiment(score):
    if score > 0:
        return 'Positive'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Negative'


st.sidebar.title("WhatsApp Chat Analyzer")
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show analysis"):
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Total Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Total Links Shared")
            st.title(num_links)

        # Sentiment Analysis Section
        st.title("Sentiment Analysis")
        df['sentiment'] = df['message'].apply(lambda message: TextBlob(message).sentiment.polarity)
        df['sentiment_label'] = df['sentiment'].apply(categorize_sentiment)
        df = df[df['user'] != 'group_notification']  # Exclude group_notification user from sentiment analysis

        st.write("Overall Sentiment Distribution:")
        sns.set(style="ticks")
        plt.figure(figsize=(8, 6))
        sns.countplot(x='sentiment_label', data=df, palette="Set2")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        plt.title("Sentiment Analysis")
        st.pyplot(plt)

        st.write("Sentiment Analysis by User:")
        user_sentiment = df.groupby('user')['sentiment'].mean().reset_index()
        st.bar_chart(user_sentiment.set_index('user'))

        # Sentiment Trends
        st.title("Sentiment Trends")
        sentiment_trends = df.groupby(df['only_date'])['sentiment'].mean()
        fig, ax = plt.subplots()
        ax.plot(sentiment_trends.index, sentiment_trends.values, color='blue')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

    # monthly timeline
    st.title("Monthly Timeline")
    timeline = helper.monthly_timeline(selected_user, df)
    fig, ax = plt.subplots()
    ax.plot(timeline['time'], timeline['message'], color='green')
    plt.xticks(rotation='vertical')
    st.pyplot(fig)

    # daily timeline
    st.title("Daily Timeline")
    daily_timeline = helper.daily_timeline(selected_user, df)
    fig, ax = plt.subplots()
    ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='pink')
    plt.xticks(rotation='vertical')
    st.pyplot(fig)

    # activity map
    st.title('Activity Map')
    col1, col2, = st.columns(2)

    with col1:
        st.header("Most busy day")
        busy_day = helper.week_activity_map(selected_user, df)
        fig, ax = plt.subplots()
        ax.bar(busy_day.index, busy_day.values)
        plt.xticks(rotation='vertical')

        st.pyplot(fig)

    with col2:
        st.header("Most busy month")
        busy_month = helper.month_activity_map(selected_user, df)
        fig, ax = plt.subplots()
        ax.bar(busy_month.index, busy_month.values, color='orange')
        plt.xticks(rotation='vertical')

        st.pyplot(fig)

    st.title("Weekly Activity Map")
    user_heatmap = helper.activity_heatmap(selected_user, df)
    fig, ax = plt.subplots()
    sns.heatmap(user_heatmap, ax=ax)  # Use the existing ax object
    st.pyplot(fig)

    # Busiest user
    if selected_user == 'Overall':
        st.title('Most Busy Users')
        x, new_df = helper.most_busy_users(df)
        fig, ax = plt.subplots()

        col1, col2, = st.columns(2)
        with col1:
            ax.bar(x.index, x.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.dataframe(new_df)

    # WordCloud
    df_wc = helper.create_wordcloud(selected_user, df)
    fig, ax = plt.subplots()
    ax.imshow(df_wc)
    st.pyplot(fig)

    # Most common df
    most_common_df = helper.most_common_words(selected_user, df)
    fig, ax = plt.subplots()

    ax.barh(most_common_df[0], most_common_df[1])  # Using barh() for a horizontal bar graph
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.title('Most Common Words')
    plt.tight_layout()  # Ensures the labels fit within the figure area

    st.pyplot(fig)
