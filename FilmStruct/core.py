import streamlit as st
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_mistralai import ChatMistralAI

load_dotenv()


@st.cache_resource
def load_model():
    return ChatMistralAI(model="mistral-small-2506", temperature=0)


model = load_model()


class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str


parser = PydanticOutputParser(pydantic_object=Movie)


prompt = ChatPromptTemplate.from_messages([
    ("system", """
Extract movie details from the given paragraph.

Keep the summary short (2–3 sentences) and rewrite it in your own words.
Do not copy text directly.

{format_instructions}
"""),
    ("human", "{paragraph}")
])


st.set_page_config(page_title="FlimStruct", layout="centered")

st.title("FlimStruct")
st.write("Paste a movie paragraph and get structured data.")


if st.button("Use sample"):
    st.session_state["text"] = """3 Idiots is a comedy-drama film about three engineering students studying at a top college in India. The story questions the traditional education system and focuses on learning with curiosity. It stars Aamir Khan, R. Madhavan, and Sharman Joshi, and is directed by Rajkumar Hirani."""


text = st.text_area(
    "Movie description",
    value=st.session_state.get("text", ""),
    height=160
)


if st.button("Extract"):
    if not text.strip():
        st.warning("Add some text first.")
    else:
        with st.spinner("Working on it..."):
            try:
                formatted = prompt.invoke({
                    "paragraph": text,
                    "format_instructions": parser.get_format_instructions()
                })

                result = model.invoke(formatted)
                data = parser.parse(result.content)

                if data.summary.strip() in text:
                    retry = prompt.invoke({
                        "paragraph": text + "\nRewrite the summary.",
                        "format_instructions": parser.get_format_instructions()
                    })
                    result = model.invoke(retry)
                    data = parser.parse(result.content)

                col1, col2 = st.columns(2)

                with col1:
                    st.write("Title:", data.title)
                    st.write("Year:", data.release_year or "N/A")
                    st.write("Genre:", ", ".join(data.genre))

                with col2:
                    st.write("Director:", data.director or "N/A")
                    st.write("Rating:", data.rating or "N/A")

                st.write("Cast:")
                st.write(", ".join(data.cast) if data.cast else "N/A")

                st.write("Summary:")
                st.write(data.summary)

                with st.expander("Raw output"):
                    st.json(data.dict())

                st.download_button(
                    "Download JSON",
                    data=json.dumps(data.dict(), indent=2),
                    file_name="movie.json",
                    mime="application/json"
                )

            except Exception as e:
                st.error("Something went wrong while parsing.")
                st.exception(e)
