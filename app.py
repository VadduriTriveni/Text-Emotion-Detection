import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Load model
pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

# Emoji Dictionary
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱",
    "happy": "🤗", "joy": "😂", "neutral": "😐",
    "sad": "😔", "sadness": "😔", "shame": "😳",
    "surprise": "😮"
}

# Predict Emotion
def predict_emotions(docx):
    return pipe_lr.predict([docx])[0]

# Prediction Probability
def get_prediction_proba(docx):
    return pipe_lr.predict_proba([docx])

# Streamlit App
def main():
    st.title("Text Emotion Detection 😄😢😠")
    st.subheader("Detect emotions from your input text")

    with st.form(key='emotion_form'):
        raw_text = st.text_area("Enter your text here:")
        submit_text = st.form_submit_button(label='Analyze')

    if submit_text:
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Predicted Emotion")
            st.write(f"{prediction} {emotions_emoji_dict.get(prediction, '')}")
            st.write(f"Confidence: {np.max(probability):.2f}")

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["Emotion", "Probability"]

            chart = alt.Chart(proba_df_clean).mark_bar().encode(
                x='Emotion',
                y='Probability',
                color='Emotion'
            )
            st.altair_chart(chart, use_container_width=True)

if __name__ == '__main__':
    main()
