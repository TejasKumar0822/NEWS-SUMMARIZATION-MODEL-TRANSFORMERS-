import streamlit as st
from summarizer import summarize

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Hybrid Multilingual News Summarizer",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Hybrid Multilingual News Summarizer")

st.markdown(
"""
Paste a **news article or URL** below.

### Supported Languages
English • Hindi • Telugu • Tamil • Kannada • Bengali
"""
)

# ================= INPUT =================
user_input = st.text_area(
    "Enter News URL or Article Text",
    height=200,
    placeholder="Paste news article or URL here..."
)

generate = st.button("Generate Summary")

# ================= PROCESSING =================
if generate:

    if user_input.strip() == "":
        st.warning("⚠️ Please enter a URL or article text.")
        st.stop()

    with st.spinner("Generating summary..."):

        try:
            result = summarize(user_input)

            # Extract fields safely
            article = result.get("article", "")
            abstractive = result.get("summary", "")
            lang = result.get("detected_language", "unknown")

            st.success("✅ Summary generated successfully!")

            # ================= LANGUAGE DISPLAY =================
            st.markdown(f"🌐 **Detected Language:** `{lang.upper()}`")

            # ================= SUMMARY =================
            st.subheader("✨ Summary")

            if abstractive and len(abstractive.strip()) > 0:
                st.write(abstractive)
            else:
                st.warning("⚠️ No summary could be generated.")

            st.markdown("---")

            # ================= FULL ARTICLE =================
            if article:
                with st.expander("📄 View Full Article"):
                    st.write(article)

        except Exception as e:
            st.error("❌ Error during summarization")
            st.error(str(e))