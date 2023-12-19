import streamlit as st
from streamlit_lottie import st_lottie
import requests
from PIL import Image


# --- CONFIG --- #

st.set_page_config(page_title="Feel the genres", page_icon=":sparkles:", layout="wide")

def load_animation(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

## --- IMAGES AND ANIMATIONS --- #
movie_animation = load_animation("https://lottie.host/30a4816a-5b22-400d-92ab-30115ded0ab5/oh3wyo4dat.json")
example_img = Image.open("images/example_img.JPG")

# --- STYLE --- #

def upload_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# --- HEADER --- #
with st.container():
    st.title(":sparkles: Feel the genres :sparkles:, by Badafixm01")
    st.subheader("A data analysis story about movie genres and emotions")
    st.write("Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.")

# --- SIDEBAR --- #
with st.container():
    with st.sidebar:
        st.subheader("About the project")
        st.write("How can a movie genre achieve success in box office revenue? We primarily seek the answer in the realm of emotions. Film enthusiasts and casual viewers alike often base their movie choices on emotions they wish to experience, whether it is thrill, romance, fear, or joy. Utilising the movie plot summaries and their corresponding metadata, we could use NLP techniques to extract sentiment scores from these summaries. Our hypothesis is that certain sentiments, like joy or suspense, may consistently align with higher box office revenues for some genres. By correlating sentiment scores with box office revenues for different genres and potentially factoring in main character gender, release dates, countries and cultural contexts, this analysis aims to reveal the underlying emotional décor that drives moviegoers to the cinema. This investigation could not only provide valuable insights for film producers but also paint a picture of society's emotional needs and preferences at different times, across different countries.")
        st.subheader("Practical information")
        st.write("This project is a data analysis story about feeling genres emotions. Obviously guys, the website is not complete... With everyone working, I'm sure we'll get a hell of a grade.")
        st.write("The [data](https://www.cs.cmu.edu/~ark/personas/) was collected by David Bamman, Brendan O'Connor, and Noah Smith at the Language Technologies Institute and Machine Learning Department at Carnegie Mellon University, and the project is made with [Streamlit](https://streamlit.io/).")
        st.write("The project is hosted on [GitHub](https://ineskahlaoui.github.io/badafixm01/)")
        st.subheader("The team")
        st.write("[Inès Kahlaoui](https://www.instagram.com/ines_kahlaoui/), [Mya Lahjouji](https://www.instagram.com/myalahjouji/), Berta Céspedes, Xiaocheng Zhang and Fernando Meireles")
        st.write("External libraries and frameworks used : ... to be completed ...")

# --- MAIN --- #
with st.container():
    st.subheader("Part XX : if needed, this is how u create double columns & add cool animations")
    col1, col2 = st.columns(2)
    with col1:
        st.header("Column 1")
        st.write(""" 
        Lorem ipsum dolor sit amet, consectetur adipiscing elit.
        Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, 
                 when an unknown printer took a galley of type and scrambled it to make a type specimen book.
        Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
        Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.""")
    with col2:
        st.header("Column 2")
        st_lottie(movie_animation, speed=1, height=400, key="coding")



with st.container():
    st.subheader("Part XX : if needed, this is how u create additional parts")


    st.header("subpart 1")
    st.write(""" 
        Lorem ipsum dolor sit amet, consectetur adipiscing elit.
        Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, 
                 when an unknown printer took a galley of type and scrambled it to make a type specimen book.
        Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
        Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.""")
    
    st.header("subpart 2 - this is how u add an image")
    st.image(example_img, caption="This is an example image", use_column_width=True)


# --- FOOTER --- #
with st.container():
    st.subheader("Part XX : if needed, this is how u add a footer")
    st.write("This is a footer")
    st.write("This is another footer")
    st.write("This is a third footer")
    